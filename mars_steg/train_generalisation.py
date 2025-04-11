"""
Module Name: train_generalisation.py
------------------------


Description:
------------

Created 29.03.2025
Simplified training script for the LanguageTask == ToMTokenBanTask
See Also:
---------

- mars_steg.utils.common.get_dataloaders_and_ref_model
- mars_steg.utils.common.prompt_data.BatchPromptData
- mars_steg.task
- mars_steg.language
"""


from __future__ import annotations
import warnings
import numpy as np
import wandb
import sys

import os

import torch
from tqdm import tqdm
from transformers import set_seed

import trl
import random

from mars_steg.utils.ppo_trainer import PPOConfig, PPOTrainer
import sys
from mars_steg.language.language_aspects.theory_of_mind_aspect import ToMTokenBanTask
from mars_steg.utils.prompt_data import BatchPromptData, log_merged_batch_wandb
from mars_steg.config import ConfigLoader, ExperimentArgs, PromptConfig

from mars_steg.utils.common import (
    get_dataloaders_and_ref_model,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_device_map,
    get_rewards_and_training_datas
)

from mars_steg.utils.answer_extraction import extract_cots


def train(ppo_config, model_config, optimizer_config, train_config, generation_config, experiment_args, prompt_config, device_map):

    # Set random seed for reproducibility
    set_seed(ppo_config.seed)

    output_delimitation_tokens = {
        "start_scratchpad_token": prompt_config.start_scratchpad_token,
        "end_scratchpad_token": prompt_config.end_scratchpad_token,
        "start_output_token": prompt_config.start_output_token,
        "end_output_token": prompt_config.end_output_token,
    }


    tokenizer = get_tokenizer(model_config.model_name)
    generation_config.pad_token_id = tokenizer.eos_token_id 

    #TODO Resolve the problem of the model looping on final answer
    generation_config.eos_token_id = tokenizer.eos_token_id 
    model = get_model(model_config, tokenizer, output_delimitation_tokens, generation_config, device_map["main_model"])
    
    if train_config.load_lora_from_wandb:
        print(f"LOADING LORA WEIGHTS FROM {train_config.load_lora_from_path_wandb}")
        from peft import PeftModel
        artifact = wandb.use_artifact(train_config.load_lora_from_path_wandb, type='model')
        artifact_dir = artifact.download()
        print(f"Artifact downloaded to {artifact_dir}")

        model.model.pretrained_model.base_model.model = PeftModel.from_pretrained(
            model.model.pretrained_model.base_model.model,
            artifact_dir
        )
        
        # Pick these up from wandb artifact! - TODO with Luis!
        #test_nouns = ...

    else:
        test_nouns = None


    optimizer = get_optimizer(optimizer_config=optimizer_config, model=model.model)

    training_generation_kwargs = generation_config.to_training_dict()

    assert experiment_args.dataset_class_name == 'TheoryOfMindTask', \
        "train_generalisation.train requires experiment_args.dataset_class_name == 'TheoryOfMindTask'"

    train_dataset, _, test_dataset, train_loader, _, test_loader, ref_model = get_dataloaders_and_ref_model(
        dataset_class_name = experiment_args.dataset_class_name,
        penalisation_class_name = experiment_args.penalisation_class_name,
        prompt_config = prompt_config,
        dataset_class_kwargs = experiment_args.dataset_class_kwargs,
        penalisation_class_kwargs = train_config.penalisation_class_kwargs,
        train_proportion = train_config.train_proportion,
        validation_proportion = train_config.validation_proportion,
        batch_size = train_config.batch_size,
        training_model = model,
        override_create_ref_model = train_config.create_ref_model,
        number_shared_layers_for_ref_model = train_config.number_shared_layers_for_ref_model,
        test_train_split_kwargs={
            'mode': 'unseen_nouns',
            'test_nouns': test_nouns,
        },
        device_map=device_map
    )

    ppo_trainer = PPOTrainer(
        ppo_config,
        model.model,
        ref_model=ref_model if train_config.create_ref_model else None,
        tokenizer=tokenizer,
        optimizer=optimizer,
    )

    print(ppo_trainer.ref_model)


    #model.model.pretrained_model.load_adapter(train_config.load_lora_from_local_path,adapter_name="lora")

    assert train_config.number_of_evaluations == 0, \
        "train_generalisation.py is intended for train_config.number_of_evaluations == 0, it will automatically evaluate at the end of each epoch"

    assert isinstance(train_dataset.language_aspect, ToMTokenBanTask), \
        "train_generalisation.py is intended to only be used for the LanaguageAspect subclass ToMTokenBanTask"

    wandb_run = wandb.run

    for epoch in range(train_config.num_train_epochs):

        print(f'BEGINING EPOCH {epoch}\n\n')

        wandb.log(
            {
                'train_nouns': train_dataset.language_aspect.penalise_substrings,
                'test_nouns': test_dataset.language_aspect.penalise_substrings,
            }
        )

        batch_ticker = 0
        overall_extracted = 0
        overall_failed = 0

        if not experiment_args.test_only:
            for batch_prompt_datas in tqdm(train_loader):

                batch_prompt_datas: BatchPromptData

                batch_messages = model.batchize_conversation(
                    user_prompts= batch_prompt_datas.cot_prompts, 
                    system_prompt=train_dataset.system_prompt,
                )
                
                transformed_batch_conversation = [model.transform_conversation(conversation, prompt_config.prompt_thinking_helper) for conversation in batch_messages]

                inputs = model.tokenize(transformed_batch_conversation)
                
                # Move the tokenized inputs to the same device the model is on (GPU/CPU)
                inputs = {key: tensor.to(device_map["main_model"]) for key, tensor in inputs.items()}

                # Translate it into ppo.generate format which List[torch.tensors]
                query_tensors = []
                for input in inputs["input_ids"]:
                    query_tensors.append(input)


                # Generate the with-CoT transcript (no-CoT transcript not needed during training)
                transcript_responses = ppo_trainer.generate(
                    query_tensors, **training_generation_kwargs,
                )

                decoded_responses = [
                    tokenizer.decode(r.squeeze()) for r in transcript_responses
                ]

                # extraction of answer and cot (because cot_mode = True) parts of the transcript
                batch_prompt_datas.cot_transcripts = decoded_responses

                # Extract answer and CoT before given to neural overseer and assessor
                # If this fails - we might want to still penalise the batch
                extracted_batch_prompt_datas = extract_cots(batch_prompt_datas, model.output_delimitation_tokens, train_config.skipping_failed_parsing_examples)
                extracted_batch_prompt_datas.extracted_cot_token_lengths = [
                    len(tokenizer(extracted_cot, add_special_tokens=False)["input_ids"]) if extracted_cot is not None else None 
                    for extracted_cot in extracted_batch_prompt_datas.extracted_cots
                ]
            
                # After all that answer/CoT extraction, it might have completely failed, so we can skip everything below entirely
                if len(extracted_batch_prompt_datas) == 0:
                    if train_config.skipping_failed_parsing_examples:
                        warnings.warn("All examples failed to parse by neural assessor and/or overseer. Skipping the optimisation for this batch of examples")
                        continue
                    else:
                        warnings.warn("All examples failed to parse. Penalising for all examples below")

                # Might have partially failed...
                elif len(extracted_batch_prompt_datas) != len(batch_prompt_datas):
                    num_examples_failed = len(decoded_responses) - len(extracted_batch_prompt_datas)
                    warnings.warn(
                        f"Failed to parse {num_examples_failed} examples out of {len(decoded_responses)}"
                    )

                # Might have completely worked!
                else:
                    print("All examples parsed successfully")

            
                if train_config.use_temporal_penalization:
                    composite_reward_list, task_score_list, language_score_list, train_query_tensors, train_transcript_responses, penalisation_tensor_list, task_score_list_no_failed, language_score_list_no_failed, composite_reward_list_no_failed = \
                    get_rewards_and_training_datas(
                        batch_prompt_datas=extracted_batch_prompt_datas,
                        query_tensors=query_tensors,
                        transcript_responses=transcript_responses, 
                        train_dataset=train_dataset,
                        skipping_failed_parsing_examples = train_config.skipping_failed_parsing_examples,
                        t_weight=train_config.t_weight,
                        l_weight=train_config.l_weight,
                        tokenizer=tokenizer
                    )
                else:
                    composite_reward_list, task_score_list, language_score_list, train_query_tensors, train_transcript_responses, _, task_score_list_no_failed, language_score_list_no_failed, composite_reward_list_no_failed = \
                    get_rewards_and_training_datas(
                        batch_prompt_datas=extracted_batch_prompt_datas,
                        query_tensors=query_tensors,
                        transcript_responses=transcript_responses, 
                        train_dataset=train_dataset,
                        skipping_failed_parsing_examples = train_config.skipping_failed_parsing_examples,
                        t_weight=train_config.t_weight,
                        l_weight=train_config.l_weight,
                    )
                    

                
                log_merged_batch_wandb([batch_prompt_datas], epoch = epoch, batch_in_epoch = batch_ticker)

                if len(composite_reward_list) == 0:
                    warnings.warn("All feedbacks (get_task_score and/or get_language_score) failed to parse. Skipping the optimisation for this batch of examples")
                    continue
                
                overall_extracted += len(composite_reward_list)
                overall_failed += len(composite_reward_list) - len(task_score_list_no_failed)
                overall_ratio_failed = overall_failed/ overall_extracted
                print("--------")
                print(f"Rewarding example with {composite_reward_list}")
                print(f"Task score: {task_score_list}")
                print(f"Language score: {language_score_list}")
                print(f"Rewarding no failes examples with {composite_reward_list_no_failed}")
                print(f"No failed Task score: {task_score_list_no_failed}")
                print(f"No failed Language score: {language_score_list_no_failed}")
                print(f"Percent of failed: {overall_ratio_failed}%")
                print("--------")
                # Set this flag to False to skip optimization
                run_optimization = train_config.run_optimization

                # If running optimization, adjust the batch size based on the available rewards.
                if run_optimization:
                    if len(composite_reward_list) < ppo_trainer.config.batch_size:

                        print('Some items failed - decreasing batch size for security - reset shortly')

                        ppo_trainer.config.batch_size = len(composite_reward_list)
                        ppo_trainer.config.mini_batch_size = 1
                        ppo_trainer.config.gradient_accumulation_steps = len(composite_reward_list)
                        ppo_trainer.config.backward_batch_size = len(composite_reward_list)

                    stats = ppo_trainer.step(
                        train_query_tensors, train_transcript_responses, composite_reward_list, penalization_tensors=penalisation_tensor_list, tokeniser=tokenizer
                    )
                else:
                    # When skipping optimization, we can set stats to an empty dict or
                    # some placeholder information.
                    stats = {"info": "Optimization skipped (logging only)"}

                # IMPORTANT: Ensure the inputs contain the expected keys for logging.
                inputs["query"] = batch_prompt_datas.cot_prompts
                inputs["response"] = batch_prompt_datas.cot_transcripts

                # Log the stats along with the rewards and scores.
                ppo_trainer.log_stats(
                    stats,
                    inputs,
                    [composite_reward_list, task_score_list, language_score_list],
                )

                # If we ran an optimization step, restore the original batch size.
                if run_optimization:
                    ppo_trainer.config.batch_size = train_config.batch_size
                    ppo_trainer.config.mini_batch_size = train_config.mini_batch_size
                    ppo_trainer.config.gradient_accumulation_steps = train_config.gradient_accumulation_steps
                    ppo_trainer.config.backward_batch_size = train_config.batch_size

                # Log stats to wandb.
                # import pdb; pdb.set_trace(header = "2025.02.25: associate this with a epoch and batch ticker")
                wandb.log(stats)            
                epsilon = 1e-8  # Small value to prevent division by zero

                wandb.log({
                    'epoch': epoch,
                    'batch_in_epoch': batch_ticker,
                    'mean_cot_token_length': batch_prompt_datas.mean_length_cot(),
                    'task_score': sum(task_score_list) / (len(composite_reward_list) + epsilon),
                    'language_score': sum(language_score_list) / (len(language_score_list) + epsilon),
                    'no_failed_task_score': sum(task_score_list_no_failed) / (len(composite_reward_list_no_failed) + epsilon),
                    'no_failed_language_score': sum(language_score_list_no_failed) / (len(language_score_list_no_failed) + epsilon),
                    'no_failed_reward': sum(composite_reward_list_no_failed) / (len(composite_reward_list_no_failed) + epsilon),
                    'reward': sum(composite_reward_list) / (len(composite_reward_list) + epsilon),
                    'overall_ratio_failed': overall_ratio_failed,
                })

                # Clear CUDA cache if needed.
                if "cuda" in device_map["main_model"]:
                    torch.cuda.empty_cache() 


                print(batch_ticker % train_config.save_frequency)
                if batch_ticker % train_config.save_frequency == 0:
                    if ppo_trainer.accelerator.is_main_process:
                        save_path = f"experiment_lora_cache"
                        run_name = f"{run_wandb.id}"
                        ppo_trainer.model.pretrained_model.save_pretrained(save_path)
                        artifact = wandb.Artifact(name=f"{run_name}_model_{epoch}_step_{batch_ticker}", type=f"model")
                        artifact.add_dir(save_path)  # Adds the entire folder
                        wandb.log_artifact(artifact)

                batch_ticker += 1


    

        print(f'BEGINING TESTING EPOCH {epoch}\n\n')

        test_batch_ticker = 0
        test_overall_extracted = 0
        test_overall_failed = 0
        with torch.no_grad():
            # START EVALUATION
            for test_batch_prompt_datas in tqdm(test_loader):

                test_batch_prompt_datas: BatchPromptData

                test_batch_messages = model.batchize_conversation(
                    user_prompts= test_batch_prompt_datas.cot_prompts, 
                    system_prompt= test_dataset.system_prompt,
                )

                
                transformed_batch_conversation = [model.transform_conversation(conversation, prompt_config.prompt_thinking_helper) for conversation in test_batch_messages]

                inputs = model.tokenize(transformed_batch_conversation)
                
                # Move the tokenized inputs to the same device the model is on (GPU/CPU)
                inputs = {key: tensor.to(device_map["main_model"]) for key, tensor in inputs.items()}

                # Translate it into ppo.generate format which List[torch.tensors]
                query_tensors = []
                for input in inputs["input_ids"]:
                    query_tensors.append(input)


                # Generate the with-CoT transcript (no-CoT transcript not needed during training)
                transcript_responses = ppo_trainer.generate(query_tensors, **training_generation_kwargs,)
                decoded_responses = [tokenizer.decode(r.squeeze()) for r in transcript_responses]
                
                # extraction of answer and cot (because cot_mode = True) parts of the transcript
                test_batch_prompt_datas.cot_transcripts = decoded_responses

                # Extract answer and CoT before given to neural overseer and assessor
                # If this fails - we might want to still penalise the batch
                extracted_test_batch_prompt_datas = extract_cots(test_batch_prompt_datas, model.output_delimitation_tokens, train_config.skipping_failed_parsing_examples)
                extracted_test_batch_prompt_datas.extracted_cot_token_lengths = [
                    len(tokenizer(extracted_cot, add_special_tokens=False)["input_ids"]) if extracted_cot is not None else None 
                    for extracted_cot in extracted_test_batch_prompt_datas.extracted_cots
                ]
                print("-----")
                for cot in decoded_responses:
                    print(f"CoT: {cot}")
                print("-----")
                
                # After all that answer/CoT extraction, it might have completely failed, so we can skip everything below entirely
                if len(extracted_test_batch_prompt_datas) == 0 and train_config.skipping_failed_parsing_examples:
                    warnings.warn("All test examples failed to parse by neural assessor and/or overseer. Skipping the test-time evaluation for this batch of examples")
                    continue

                # Extract rewards
                if train_config.use_temporal_penalization:
                    
                    test_composite_reward_list, test_task_score_list, test_language_score_list, test_query_tensors, test_transcript_responses, _, test_task_score_list_no_failed, test_language_score_list_no_failed, test_composite_reward_list_no_failed = \
                    get_rewards_and_training_datas(
                        batch_prompt_datas=extracted_test_batch_prompt_datas,
                        query_tensors=query_tensors,
                        transcript_responses=transcript_responses, 
                        train_dataset=train_dataset,
                        skipping_failed_parsing_examples = train_config.skipping_failed_parsing_examples,
                        t_weight=train_config.t_weight,
                        l_weight=train_config.l_weight,
                        tokenizer=tokenizer
                    )
                else:
                    test_composite_reward_list, test_task_score_list, test_language_score_list, test_query_tensors, test_transcript_responses,  _, test_task_score_list_no_failed, test_language_score_list_no_failed, test_composite_reward_list_no_failed = \
                    get_rewards_and_training_datas(
                        batch_prompt_datas=extracted_test_batch_prompt_datas,
                        query_tensors=query_tensors,
                        transcript_responses=transcript_responses, 
                        train_dataset=train_dataset,
                        skipping_failed_parsing_examples = train_config.skipping_failed_parsing_examples,
                        t_weight=train_config.t_weight,
                        l_weight=train_config.l_weight
                    )

                log_merged_batch_wandb([test_batch_prompt_datas], epoch = epoch, batch_in_epoch = test_batch_ticker, name="_test")

                test_overall_extracted += len(test_composite_reward_list)
                test_overall_failed += len(test_composite_reward_list) - len(test_task_score_list_no_failed)
                if test_overall_failed == 0:
                    test_overall_ratio_failed = 0
                else:
                    test_overall_ratio_failed = test_overall_extracted/test_overall_failed

                print("--------")
                print(f"Test examples rewarded with {test_composite_reward_list}")
                print(f"Task score: {test_task_score_list}")
                print(f"Language score: {test_language_score_list}")
                print("--------")

                epsilon = 1e-8  # Small value to prevent division by zero

                wandb.log({
                    'test_epoch': epoch,
                    'test_batch_in_epoch': test_batch_ticker,
                    'test_mean_cot_token_length': test_batch_prompt_datas.mean_length_cot(),
                    'test_task_score': sum(test_task_score_list) / (len(test_composite_reward_list) + epsilon),
                    'test_language_score': sum(test_language_score_list) / (len(test_language_score_list) + epsilon),
                    'test_reward': sum(test_composite_reward_list) / (len(test_composite_reward_list) + epsilon),
                    'non_failed_test_task_score': sum(test_task_score_list_no_failed) / (len(test_task_score_list_no_failed) + epsilon),
                    'non_failed_test_language_score': sum(test_language_score_list_no_failed) / (len(test_language_score_list_no_failed) + epsilon),
                    'non_failed_test_reward': sum(test_composite_reward_list_no_failed) / (len(test_composite_reward_list_no_failed) + epsilon),
                    'test_overall_ratio_failed': test_overall_ratio_failed
                })
                test_batch_ticker += 1




if __name__ == "__main__":

    # TODO: argsparsify

    print(sys.argv)
    experiment_args_path = sys.argv[1]
    default_config_path = sys.argv[2]   # e.g. "mars_steg/configs/config.yaml"
    try:
        overwrite_config_path = sys.argv[3]
        prompt_config_path = sys.argv[4]
    except IndexError:
        warnings.warn("No overwrite_config_path has been given, setting it to None by default !")
        overwrite_config_path = None  # e.g. "mars_steg/configs/config.yaml"
        prompt_config_path = sys.argv[3]
       # e.g. "mars_steg/configs/prompt/math_training_deepseek.yaml"


    device_map = get_device_map()


    # seed = 16
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # # Ensures deterministic behavior for cudnn
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ["PYTHONHASHSEED"] = str(seed)


    ####################################################################################################################
    # GET CONFIGS 
    ####################################################################################################################

    if overwrite_config_path is None:
        model_config, optimizer_config, train_config, generation_config = ConfigLoader.load_config(default_config_path)
    else:
        model_config, optimizer_config, train_config, generation_config = ConfigLoader.load_config(default_config_path, overwrite_config_path)

    prompt_config = PromptConfig.from_file(prompt_config_path)

    generation_config.batch_size = train_config.batch_size

    ppo_config = PPOConfig(
        model_name=model_config.model_name,
        learning_rate=optimizer_config.learning_rate,
        log_with=train_config.log_with,
        ppo_epochs=train_config.ppo_epochs,
        mini_batch_size=train_config.mini_batch_size,
        batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        is_peft_model=train_config.is_peft_model,
        init_kl_coef=train_config.init_kl_coef,
        adap_kl_ctrl=train_config.adap_kl_ctrl,
    )


    experiment_args = ExperimentArgs.from_yaml(experiment_args_path)

    ####################################################################################################################
    # SET UP WANDB LOGGING
    ####################################################################################################################

    wandb.init(
        project="MARS-STEGO",
        entity="luisoso",
        config={
            'ppo_config': ppo_config,
            'model_config': model_config,
            'optimizer_config': optimizer_config,
            'train_config': train_config,
            'experiment_args': experiment_args,
            'generation_config': generation_config,
        }
    )


    train(
        ppo_config = ppo_config, 
        model_config = model_config,
        optimizer_config = optimizer_config,
        train_config = train_config,
        generation_config = generation_config,
        experiment_args = experiment_args,
        prompt_config = prompt_config,
        device_map = device_map
    )

