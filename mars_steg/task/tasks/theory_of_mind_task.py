"""
Module Name: theory_of_mind_dataset_task.py
------------------------

Description:
------------

This module contains the implementation of the class for a task for theory of mind reasoning.

Classes:
    - TheoryOfMind_Torch_Dataset_Task

Examples:
    Cf. mars_steg.README.md

"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from mars_steg.config import PromptConfig
from mars_steg.language.base_language_aspect import LanguageAspect
from mars_steg.language.language_aspects.theory_of_mind_aspect import ToMTokenBanTask
from mars_steg.task.base_task import TorchDatasetTask
from mars_steg.utils.answer_extraction import extract_floatingpoint_overseer_or_assessor_answer
from mars_steg.utils.prompt_data import PromptData

import random
import yaml


class TheoryOfMindTask(TorchDatasetTask):
    """
    A dataset-based task for theory of mind reasoning, extending `TorchDatasetTask`.

    This class is designed for evaluating LLMs on theory of mind reasoning tasks,
    where the model must understand beliefs, intentions, and mental states of
    characters in stories. It processes structured datasets containing stories and
    questions about characters' knowledge and beliefs.

    Attributes
    ----------
    name : str
        The name of the task, set as "TheoryOfMind_Dataset_Task".
    system_prompt : str
        A formatted system prompt combining task descriptions, answer instructions,
        and example problems.
    language_aspect : LanguageAspect
        The language aspect settings for this task.
    uses_local_neural_assessor : bool
        Whether the task utilizes a local neural assessor for evaluation.
    prompt_config : PromptConfig
        The prompt configuration for generating structured prompts.

    Methods
    -------
    generate_prompt(index: int) -> str
        Generates a theory of mind reasoning prompt based on the dataset index.
    get_task_score(prompt_data: PromptData, with_cot: bool) -> float
        Computes the task success score based on the LLM's final answer.
    exact_answer_matching(llm_answer: str, expected_answer: str) -> float
        Performs exact string matching for answer validation.
    """

    # By default, set the name as the class name
    name = "TheoryOfMindTask"

    def __init__(
        self,
        dataset: str,
        language_aspect: LanguageAspect,
        uses_local_neural_assessor: bool,
        prompt_config: PromptConfig,
        batch_size: int,
        nouns_path: str
    ):
        """
        Initializes the TheoryOfMind_Torch_Dataset_Task.

        Parameters
        ----------
        dataset : str
            Path to the dataset (CSV format).
        language_aspect : LanguageAspect
            The language aspect settings for this task.
        uses_local_neural_assessor : bool
            Whether the task utilizes a local neural assessor for evaluation.
        prompt_config : PromptConfig
            The prompt configuration for generating structured prompts.
        batch_size : int
            Batch size for torch dataloaders.
        nouns : dict
            Maps from e.g. "names" to list of names, "objects" to list of objects, etc.
        """

        super().__init__(
            dataset=dataset,
            language_aspect=language_aspect,
            uses_local_neural_assessor=uses_local_neural_assessor,
            prompt_config=prompt_config,
            batch_size=batch_size
        )

        self.system_prompt = (
            prompt_config.task_description + '\n\n' +
            prompt_config.answer_format_instruction + '\n\n' +
            prompt_config.task_examples + '\n\n'
        )

        self.matching_method = "levenshtein" # TODO : Add this to config

        self.nouns_path = nouns_path
        with open(nouns_path, 'r') as f:
            self.nouns = yaml.safe_load(f)

    def generate_prompt(self, index: int) -> str:
        """
        Generates a theory of mind reasoning prompt based on the dataset index.

        Parameters
        ----------
        index : int
            The index of the dataset entry.

        Returns
        -------
        str
            The combined story and question prompt.

        Raises
        ------
        ValueError
            If the index exceeds the dataset length.
        """
        # Just for safety
        if index >= self.length:
            raise ValueError("index received was greater than dataset length")
        
        # Get story and question 
        ## story = self.dataset.iloc[index]["story_structure"]
        infilled_story = self.dataset.iloc[index]["infilled_story"]
        question = self.dataset.iloc[index]["question"]
        
        prompt = f"{self.system_prompt}\n\nStory: {infilled_story}\n\nQuestion: {question}"

        return prompt

    def get_task_score(self, prompt_data: PromptData, with_cot: bool, skipping_failed_parsing_examples: bool = True) -> float:
        """
        Computes the task success score based on the LLM's final answer.

        Parameters
        ----------
        prompt_data : PromptData
            The structured prompt data containing the model's responses.
        with_cot : bool
            Whether to evaluate the response generated with Chain-of-Thought (CoT).
        skipping_failed_parsing_examples: bool
            Skip examples that failed when parsing them for task score, language score, 
            answer extraction, CoT extraction etc.

        Returns
        -------
        float
            A task success score in the range [0,1].

        Notes
        -----
        - If the task does not use a neural assessor, `exact_answer_matching` is applied.
        - If a neural assessor is used, the score is extracted from the assessor's evaluation.
        """
        # Get problem index from transcript
        idx = prompt_data.info.idx
        # Look up expected answer
        expected_answer = self.dataset.iloc[idx]["expected_answer"]

        llm_answer = (
            prompt_data.extracted_final_answer_with_cot
            if with_cot
            else prompt_data.extracted_final_answer_without_cot
        )
        
        if not self.uses_local_neural_assessor:
            if self.matching_method == "levenshtein":
                return self.levenshtein_answer_matching(llm_answer, expected_answer)
        else:
            return extract_floatingpoint_overseer_or_assessor_answer(prompt_data.extracted_assessor_answer)
    
    def levenshtein_answer_matching(self, llm_answer: str, expected_answer: str, threshold: float = 0.8) -> float:
        """
        Performs flexible string-based answer validation using Levenshtein distance.

        This method checks whether the model's answer matches the expected answer
        using a similarity score based on the Levenshtein distance. This allows for
        small variations, typos, or minor differences in the answers.

        Parameters
        ----------
        llm_answer : str
            The final extracted answer generated by the model.
        expected_answer : str
            The expected correct answer.
        threshold : float, optional
            The similarity threshold for considering an answer correct (default 0.8).
            Higher values require higher similarity.

        Returns
        -------
        float
            Similarity score between 0.0 and 1.0, where 1.0 indicates perfect match.
            If the similarity score is above the threshold, it's considered correct.

        Notes
        -----
        - The comparison is performed after normalizing both strings 
        (trimming whitespace and converting to lowercase).
        - The similarity score is calculated as:
        1 - (levenshtein_distance / max(len(str1), len(str2)))
        """
        # Normalize answers for comparison
        llm_answer_norm = llm_answer.strip().lower()
        expected_answer_norm = expected_answer.strip().lower()

        # Check for exact match first (optimization)
        if llm_answer_norm == expected_answer_norm:
            return 1.0
        
        # Calculate Levenshtein distance
        distance = levenshtein_distance(llm_answer_norm, expected_answer_norm)
        
        # Calculate similarity score (1.0 means identical, 0.0 means completely different)
        max_length = max(len(llm_answer_norm), len(expected_answer_norm))
        if max_length == 0:  # Handle empty strings case
            return 1.0
        
        similarity = 1.0 - (distance / max_length)
        
        # Return raw similarity score, can be thresholded by the caller
        return similarity

    def test_train_split(
        self, 
        train_proportion: float, 
        validation_proportion: float, 
        *_,
        mode: str = 'vanilla',
        test_nouns: Optional[List[str]],
    ) -> Tuple[TorchDatasetTask, None, TorchDatasetTask]:
        """
        if mode == vanilla:
            same as before

        elif mode == unseen_nouns:
            - assert validation_proportion == 0.0
            - get list of names and how many prompts they appear in in the dataset
            - randomly subselect a list of 'test names', which accounts for 1 - train_proportion of the dataset rows
            - put all prompts with test names into the test set

        elif mode == unseen_nouns_as_role:
            - future work!
        """
        if mode == 'vanilla':
            return super().test_train_split(train_proportion, validation_proportion)
        
        elif mode == 'unseen_nouns':
            assert validation_proportion == 0.0

            if test_nouns is None:
            
                # get list of names
                names: List[str] = self.nouns['names']

                # count how many prompts contain each name
                name_counts = {}
                for name in names:
                    # Count prompts that contain each name
                    mask = self.dataset['infilled_story'].str.contains(name, regex=False)
                    name_counts[name] = mask.sum()
                
                # Calculate proportion of prompts each name appears in
                total_prompts = len(self.dataset)
                name_proportion_of_prompts = {name: count / total_prompts for name, count in name_counts.items()}
                
                # Randomly shuffle names
                shuffled_names = list(name_proportion_of_prompts.keys())
                random.shuffle(shuffled_names)

                # Initialize test and train sets
                test_names: List[str] = []
                test_proportion_sum = 0.0
                target_test_proportion = 1.0 - train_proportion
                
                # Randomized approach: keep adding randomly selected names until we reach the target proportion
                for name in shuffled_names:
                    if test_proportion_sum < target_test_proportion:
                        test_names.append(name)
                        test_proportion_sum += name_proportion_of_prompts[name]
                    else:
                        break
                
                # If we've overshot our target by a significant amount, try to remove a name to get closer
                if test_proportion_sum > target_test_proportion * 1.2:  # 20% tolerance
                    # Try to find a name that, if removed, would bring us closer to target
                    best_distance = test_proportion_sum - target_test_proportion
                    best_name_to_remove = None
                    
                    for name in test_names:
                        new_proportion = test_proportion_sum - name_proportion_of_prompts[name]
                        distance = abs(new_proportion - target_test_proportion)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_name_to_remove = name
                    
                    # Remove the name if it improves our proportion
                    if best_name_to_remove:
                        test_names.remove(best_name_to_remove)
                        test_proportion_sum -= name_proportion_of_prompts[best_name_to_remove]

            else:
                test_names = test_nouns
        
            # Remaining names go to train set
            train_names: List[str] = [name for name in names if name not in test_names]
            
            # Create masks for splitting the dataset
            test_mask = self.dataset['infilled_story'].apply(lambda x: any(name in x for name in test_names))
            train_mask = ~test_mask
            
            # Split dataset according to these masks
            train_dataset = self.dataset[train_mask].copy()
            test_dataset = self.dataset[test_mask].copy()
            
            # Print some statistics to verify the split
            actual_test_proportion = len(test_dataset) / total_prompts
            print(f"Target test proportion: {target_test_proportion:.4f}")
            print(f"Actual test proportion: {actual_test_proportion:.4f}")
            print(f"Number of test names: {len(test_names)}")
            print(f"Number of train names: {len(train_names)}")

            for name in test_names:
                assert not train_dataset['infilled_story'].str.contains(name, regex=False).any()

            assert isinstance(self.language_aspect, ToMTokenBanTask)

            test_language_aspect = self.language_aspect.limit_penalise_substrings(test_names)
            train_language_aspect = self.language_aspect.limit_penalise_substrings(train_names)

            # Create the dataset objects
            train_dataset_obj = TheoryOfMindTask(
                dataset = self.dataset_name,
                language_aspect = train_language_aspect,     # Train on all names
                uses_local_neural_assessor = False,
                prompt_config = self.prompt_config,
                batch_size = self.batch_size,
                nouns_path = self.nouns_path
            )
            train_dataset_obj.dataset = train_dataset
            train_dataset_obj.length = len(train_dataset)
            
            test_dataset_obj = TheoryOfMindTask(
                dataset = self.dataset_name,
                language_aspect = test_language_aspect,     # Test only on unseen names
                uses_local_neural_assessor = False,
                prompt_config = self.prompt_config,
                batch_size = self.batch_size,
                nouns_path = self.nouns_path
            )
            test_dataset_obj.dataset = test_dataset
            test_dataset_obj.length = len(test_dataset)
            
            return train_dataset_obj, None, test_dataset_obj

## Helpers ##

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string
    
    Returns
    -------
    int
        The Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
