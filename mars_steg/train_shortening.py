"""
Module Name: train_shortening.py
------------------------


Description:
------------

Created 11.03.2025
Simplified training script for the LanguageTask == ShortReasoning

The core file used to train an LLM to perform steganography on a task.
Configs are loaded to set up the training environment, and the model is given a task to perform.
The model is trained using here PPO, and is rewarded according to his performance on the task and some language aspects tahtw e want to penalize.

Logs are saved on wandb, and the model is saved every `save_frequency` batchs.
Models are also evaluated on the CoT gap every `number_of_evaluations` epochs, to check the effectiveness of the CoT in improving the model's performance.

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

import torch
from tqdm import tqdm
from transformers import set_seed

import trl

from mars_steg.utils.ppo_trainer import PPOConfig, PPOTrainer
import sys
from mars_steg.language.base_language_aspect import CoTLengthPenalisation
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
    optimizer = get_optimizer(optimizer_config=optimizer_config, model=model.model)

    training_generation_kwargs = generation_config.to_training_dict()


    train_dataset, _, _, train_loader, _, _, ref_model = get_dataloaders_and_ref_model(
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
        device_map=device_map
    )

    ppo_trainer = PPOTrainer(
        ppo_config,
        model.model,
        ref_model=ref_model if train_config.create_ref_model else None,
        tokenizer=tokenizer,
        optimizer=optimizer,
    )


    assert train_config.number_of_evaluations == 0, \
        "train_shortening.py is intended for train_config.number_of_evaluations == 0"

    assert isinstance(train_dataset.language_aspect, CoTLengthPenalisation), \
        "train_shortening.py is intended to only be used for the LanaguageAspect subclass CoTLengthPenalisation"


    for epoch in range(train_config.num_train_epochs):

        print(f'BEGINING EPOCH {epoch}\n\n')

        batch_ticker = 0

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

            if train_config.lora_unload_measurement:
                batch_messages_no_lora = model.batchize_conversation(
                    user_prompts= batch_prompt_datas.cot_prompts, 
                    system_prompt=train_dataset.system_prompt,
                    assistant_prompts = extracted_batch_prompt_datas.extracted_cots
                )

                transformed_batch_conversation_lora = [model.transform_conversation(conversation_no_lora) for conversation_no_lora in batch_messages_no_lora]
                inputs_no_lora = model.tokenize(transformed_batch_conversation_lora)

                # Move the tokenized inputs to the same device the model is on (GPU/CPU)
                inputs_no_lora = {key: tensor.to(device_map["main_model"]) for key, tensor in inputs_no_lora.items()}

                # Translate it into ppo.generate format which List[torch.tensors]
                query_tensors_no_lora = []
                for input_no_lora in inputs_no_lora["input_ids"]:
                    query_tensors_no_lora.append(input_no_lora)

                transcript_responses_no_lora = ppo_trainer.generate_no_lora(
                    query_tensors_no_lora, **training_generation_kwargs,
                )
                

                decoded_responses_no_lora = [
                    tokenizer.decode(r.squeeze()) for r in transcript_responses_no_lora
                ]
                batch_prompt_datas_no_lora = batch_prompt_datas
                batch_prompt_datas_no_lora.cot_transcripts = decoded_responses_no_lora

                
                extracted_batch_prompt_datas_no_lora = extract_cots(batch_prompt_datas_no_lora, model.output_delimitation_tokens, train_config.skipping_failed_parsing_examples)
                extracted_batch_prompt_datas_no_lora.extracted_cot_token_lengths = [
                    len(tokenizer(extracted_cot, add_special_tokens=False)["input_ids"]) if extracted_cot is not None else None 
                    for extracted_cot in extracted_batch_prompt_datas_no_lora.extracted_cots
                ]
                
                print("__non Lora model__")
                for i in range(len(transformed_batch_conversation_lora)):
                    print(f"Input: {transformed_batch_conversation_lora[i]}")

                    print(f"Output: {decoded_responses_no_lora[i]}")

            # extract the cot from decoded_responses_no_lora and give an score

           
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

            # Extract rewards
            composite_reward_list, task_score_list, language_score_list, train_query_tensors, train_transcript_responses, _, _, _, _ = \
            get_rewards_and_training_datas(
                batch_prompt_datas=extracted_batch_prompt_datas,
                query_tensors=query_tensors,
                transcript_responses=transcript_responses, 
                train_dataset=train_dataset,
                skipping_failed_parsing_examples = train_config.skipping_failed_parsing_examples,
                t_weight=train_config.t_weight,
                l_weight=train_config.l_weight,
            )

            if train_config.lora_unload_measurement:

                composite_reward_list_no_lora, task_score_list_no_lora, language_score_list_no_lora, _, _, _, _, _, _  = \
                get_rewards_and_training_datas(
                    batch_prompt_datas=extracted_batch_prompt_datas_no_lora,
                    query_tensors=query_tensors_no_lora,
                    transcript_responses=transcript_responses_no_lora, 
                    train_dataset=train_dataset,
                    skipping_failed_parsing_examples = train_config.skipping_failed_parsing_examples,
                )

            log_merged_batch_wandb([batch_prompt_datas], epoch = epoch, batch_in_epoch = batch_ticker)

            if len(composite_reward_list) == 0:
                warnings.warn("All feedbacks (get_task_score and/or get_language_score) failed to parse. Skipping the optimisation for this batch of examples")
                continue

            if sum(composite_reward_list)/len(composite_reward_list) >= train_config.curriculum_threshold:
                train_dataset.language_aspect.max_new_tokens_threshold = (
                    train_dataset.language_aspect.max_new_tokens_threshold * (1 - train_config.max_length_decreasing_rate)
                )
                print("--------------------------------")
                print(f"Decreasing Max token Length of {train_config.max_length_decreasing_rate}")
                print(f"Current max token length : {train_dataset.language_aspect.max_new_tokens_threshold}")
                print("--------------------------------")

            print("--------")
            print(f"Rewarding example with {composite_reward_list}")
            print(f"Task score: {task_score_list}")
            print(f"Language score: {language_score_list}")
            print("--------")

            if train_config.lora_unload_measurement:
                print("NO LORA")
                print("--------")
                print(f"Rewarding example with {composite_reward_list_no_lora}")
                print(f"Task score: {task_score_list_no_lora}")
                print(f"Language score: {language_score_list_no_lora}")
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
                    train_query_tensors, train_transcript_responses, composite_reward_list,
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

            ppo_trainer.model.pretrained_model.save_pretrained( "./_lora")  # Saves only LoRA            exit()            
            wandb.log(
                {   
                    'epoch': epoch,
                    'batch_in_epoch': batch_ticker,
                    'mean_cot_token_length': batch_prompt_datas.mean_length_cot(),
                    'task_score': sum(task_score_list) / len(composite_reward_list),
                    'language_score': sum(language_score_list) / len(language_score_list),
                    'reward' : sum(composite_reward_list) / len(composite_reward_list)
                }
            )

            # Clear CUDA cache if needed.
            if "cuda" in device_map["main_model"]:
                torch.cuda.empty_cache() 

            # Save the model every 'save_frequency' batches.
            print(batch_ticker % train_config.save_frequency)
            if batch_ticker % train_config.save_frequency == 0:
                if ppo_trainer.accelerator.is_main_process:
                    save_path = f"experiment_lora_cache"
                    ppo_trainer.model.pretrained_model.save_pretrained(save_path)
                    artifact = wandb.Artifact(name=f"model_{batch_ticker}", type="model")
                    artifact.add_dir(save_path)  # Adds the entire folder
                    wandb.log_artifact(artifact)

            batch_ticker += 1



if __name__ == "__main__":

    # TODO: argsparsify

    print(sys.argv)
    experiment_args_path = sys.argv[1]
    default_config_path = sys.argv[2]   # e.g. "mars_steg/configs/config.yaml"
    try:
        prompt_config_path = sys.argv[4]
        overwrite_config_path = sys.argv[3]
    except IndexError:
        warnings.warn("No overwrite_config_path has been given, setting it to None by default !")
        overwrite_config_path = None  # e.g. "mars_steg/configs/config.yaml"
        prompt_config_path = sys.argv[3]
       # e.g. "mars_steg/configs/prompt/math_training_deepseek.yaml"


    device_map = get_device_map()

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
        seed=train_config.seed
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

