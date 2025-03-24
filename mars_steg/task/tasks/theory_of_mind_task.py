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

from typing import List, Dict, Any
from mars_steg.config import PromptConfig
from mars_steg.language.base_language_aspect import LanguageAspect
from mars_steg.task.base_task import TorchDatasetTask
from mars_steg.utils.answer_extraction import extract_floatingpoint_overseer_or_assessor_answer
from mars_steg.utils.prompt_data import PromptData


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
        batch_size: int
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
        story = self.dataset.iloc[index]["story_structure"]
        infilled_story = self.dataset.iloc[index]["infilled_story"]
        question = self.dataset.iloc[index]["question"]
        
        prompt = f"{self.system_prompt}\n\nStory: {story}\n\nQuestion: {question}"

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