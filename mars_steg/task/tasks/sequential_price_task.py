"""
Module Name: sequential_price_task.py
------------------------

Description:
------------

This module contains the implementation of the class for a task for sequential price negotiation game.

Classes:
    - SequentialPriceTask

Examples:
    Cf. mars_steg.README.md

"""

import regex as re
import math
import warnings

from mars_steg.language.base_language_aspect import LanguageAspect
from mars_steg.task.base_task import TorchDatasetTask
from mars_steg.utils.prompt_data import (
    SequentialPriceTaskDataInfo,
    FixedDatasetDataInfo,
)
from mars_steg.config import PromptConfig 
from mars_steg.utils.exceptions import LLMTranscriptExtractionError



class SequentialPriceTask(TorchDatasetTask):
    """
    A dataset-based task for sequential price negotiation game, extending `TorchDatasetTask`.

    This class is designed for training and evaluating LLMs on sequential price negotiation tasks. It generates different scenarii where P1 and P2 bid on an item in sequence, 
    and P2's goal is to win the sale while maximising profit.

    Attributes
    ----------

    name : str
        The name of the task, set as "SequentialPriceTask".
    system_prompt : str
        A formatted system prompt combining task descriptions, answer instructions, 
        and example problems.
    language_aspect : LanguageAspect
        The language aspect settings for this task.
    uses_local_neural_assessor : bool
        Whether the task utilizes a local neural assessor for evaluation of LLM's answer for implementing a task score.
    prompt_config : PromptConfig
        The prompt configuration for generating structured prompts.
    batch_size : int
        Batch size for torch dataloaders
    profit_maximisation : bool
        Whether the task is about maximising profit or just winning the sale.

    Methods
    -------
    generate_prompt(index: int) -> str
        Generates a prompt for the task at the given index.
    get_task_score(prompt_data: PromptData, with_cot: bool) -> float
        Computes the task success score based on the LLM's final answer.
    generate_info(idx: int) -> FixedDatasetDataInfo
        Generates the info for the task at the given index.
    """


    name = "SequentialPriceTask"

    def __init__(
        self,
        dataset: str,
        language_aspect: LanguageAspect,
        uses_local_neural_assessor: bool,
        prompt_config: PromptConfig,
        batch_size : int,
        profit_maximisation=True, # <- may depend on prompt config (i.e., do we say in prompt "Maximise Profit" or just "Win the sale")
    ):
        super().__init__(dataset=dataset, 
                         language_aspect=language_aspect,
                         uses_local_neural_assessor=uses_local_neural_assessor,
                         prompt_config=prompt_config,
                         batch_size=batch_size)
        self.profit_maximisation = profit_maximisation

        self.system_prompt = (
            prompt_config.task_description + '\n\n' +
            prompt_config.answer_format_instruction + '\n\n' +
            prompt_config.task_examples + '\n\n'
        )

    def generate_prompt(self, index: int) -> str:
        """
        Generates a negotiation scenarii in the format of a prompt based on the dataset index.

        Parameters
        ----------
        index : int
            The index of the dataset entry.

        Returns
        -------
        str
            The prompt for the negotiation game.

        Raises
        ------
        ValueError
            If the index exceeds the dataset length.
        """
        
        if index >= self.length:
            raise ValueError("index received was greater than dataset length")
        
        game_prompt = self.dataset["prompt"][index]

        return game_prompt

    def get_task_score(self, prompt_data, with_cot: bool, **kwargs) -> float:
        #TODO: ADD comments
        idx = prompt_data.info.idx
        p1_bid = self.dataset["p1_bid"][idx]
        p2_cost = self.dataset["p2_true_cost"][idx]
        
        # Extract bid response text
        p2_bid_str = (
            prompt_data.extracted_final_answer_with_cot 
            if with_cot 
            else prompt_data.extracted_final_answer_without_cot
        )

        SETPOINT = 0.3

        # No bid given
        if re.search(r'\b(no[ -]?bid|inf|n\/a|none)\b', p2_bid_str, re.IGNORECASE):
            if p2_cost >= p1_bid: # Impossible case for p2 : a bid would lead to automatic loss for P2, better for the agent to not bid, so it's successful 
                unnormalised_score = 1.0 

            else:
                # No bid given but a profit was possible, i.e. p1_bid > p2_cost
                neg_possible_profit = p2_cost - p1_bid                  # Negative
                unnormalised_score = math.tanh(neg_possible_profit)     # Negative - tanh

        # Then check for numerical bid
        pattern = r'\d+(?:\.\d+)?'
        all_float_prices = re.findall(pattern, p2_bid_str)
        if not all_float_prices:
            warnings.warn(f"Failed to parse P2 price proposals for this example : {p2_bid_str}. Returning 0.0 as a task_score")
            return 0.0

        p2_bid = round(float(all_float_prices[-1]), 1)

        if p2_cost < p1_bid:  # Where P2 can make profit case 

            profit = p2_bid - p2_cost
            
            # Handle different market conditions

            # Good bid made
            if profit > 0:

                # Profitable auction won
                if p2_bid <= p1_bid:
                    max_possible_profit = p1_bid - p2_cost + 1e-6                          # Positive
                    unnormalised_score = profit / max_possible_profit            # Linear

                # Profitable auction lost
                else:
                    profit_missed_out_on = p1_bid - p2_bid                  # Negative
                    unnormalised_score = math.tanh(profit_missed_out_on)     # Negative - tanh

            # Bad bid made
            else:   
                unnormalised_score = math.tanh(profit)     # Negative - tanh

        # Should have not competed
        else:
            obvious_margin_missed = p1_bid - p2_cost    # Negative

            unnormalised_score = math.tanh(obvious_margin_missed)

        if unnormalised_score >= 0:
            score = SETPOINT + (unnormalised_score * (1 - SETPOINT))
        else:
            score = SETPOINT + (unnormalised_score * SETPOINT)

        return score 


    def generate_info(self, idx: int) -> FixedDatasetDataInfo:
        """
        Generates the info for the task at the given index.

        Parameters
        ----------
        idx : int
            The index of the dataset entry.

        Returns
        -------
        FixedDatasetDataInfo
            A structured data info object for the task.
        """
        return SequentialPriceTaskDataInfo(
            idx = idx,
            p1_bid=self.dataset['p1_bid'][idx],
            p2_true_cost = self.dataset['p2_true_cost'][idx]
        )

