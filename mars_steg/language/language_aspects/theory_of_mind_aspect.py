import os
import sys

sys.path.append(os.path.abspath(""))

from mars_steg.utils.prompt_data import PromptData

from typing import List, Optional, Set, Dict
import yaml
import re
import torch
import random
from mars_steg.language.base_language_aspect import LanguageAspect
from transformers import AutoTokenizer

class ToMTokenBanTask:
    """
    Token ban task that penalizes specific tokens in the model's output.
    Can handle both single-word names and multi-word objects.
    
    temporal_reward_penalisation_offset is used to offset the effect of kl regularisation on the length of CoT
    """
    
    compatible_tasks = {"TheoryOfMindTask"}
    uses_local_neural_overseer = False
    
    def __init__(self, nouns_path: str, nouns_penalisation_type: str, max_freq=50, 
                 temporal_reward_penalisation_offset: Optional[float] = None, **kwargs):
        """
        Initialize the token ban task.
        
        Args:
            nouns_path: Path to the YAML file containing the penalization strings
            nouns_penalisation_type: Key in the YAML file for the list of strings to penalize
            max_freq: Maximum frequency to consider for penalization
            temporal_reward_penalisation_offset: Offset for temporal reward calculation
        """
        super().__init__()
        self.nouns_path = nouns_path
        self.nouns_penalisation_type = nouns_penalisation_type
        
        # Load the penalization strings from YAML
        with open(nouns_path, 'r') as f:
            nouns = yaml.safe_load(f)
        self.penalise_substrings = set(nouns[nouns_penalisation_type])
        
        self.max_freq = max_freq
        if temporal_reward_penalisation_offset is not None:
            self.temporal_reward_penalisation_offset = torch.tensor(temporal_reward_penalisation_offset).to(torch.bfloat16)
        else:
            self.temporal_reward_penalisation_offset = None
    
    def limit_penalise_substrings(self, new_penalise_substrings: List[str]):
        """
        Create a new instance with a limited set of penalization strings.
        
        Args:
            new_penalise_substrings: New list of strings to penalize
            
        Returns:
            A new ToMTokenBanTask instance with the limited set of strings
        """
        new_instance = ToMTokenBanTask(
            nouns_path=self.nouns_path,
            nouns_penalisation_type=self.nouns_penalisation_type,
            max_freq=self.max_freq,
            temporal_reward_penalisation_offset=self.temporal_reward_penalisation_offset
        )
        new_penalise_substrings = set(new_penalise_substrings)
        assert new_instance.penalise_substrings.issuperset(new_penalise_substrings), \
            "Trying to override ToMTokenBanTask.penalise_substrings with unseen before penalisation strings!"
        new_instance.penalise_substrings = new_penalise_substrings
        
        return new_instance
    
    def tokenize_phrases(self, tokenizer: AutoTokenizer, phrases: Set[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize phrases and store their token representations.
        
        Args:
            tokenizer: The tokenizer to use
            phrases: Set of phrases to tokenize
            
        Returns:
            Dictionary mapping phrases to their tokenized representations
        """
        phrase_tokens = {}
        special_tokens = set(tokenizer.all_special_ids)

        whitespace_variants = ['', ' ', '\t', '\n']
        
        for phrase in phrases:
            all_token_variants = []

            for ws in whitespace_variants:
                text_variant = ws + phrase
                # Tokenize the phrase with spaces (different tokens)
                tokenized= tokenizer(text_variant, add_special_tokens=False)['input_ids']
                # Filter out special tokens
                cleaned_tokens= [t for t in tokenized if t not in special_tokens]
                

                if cleaned_tokens:
                    all_token_variants.append(torch.tensor(cleaned_tokens))

            if all_token_variants:
                # Just store the first one:
                phrase_tokens[phrase] = all_token_variants
                        
        return phrase_tokens
    
    def create_penalisation_tensor(self, token_ids: List[int], 
                                   tokenized_phrases: Dict[str, torch.Tensor],
                                   device: torch.device) -> torch.Tensor:
        """
        Create a tensor with penalty values for each token.
        
        Args:
            token_ids: List of token IDs from the tokenizer
            tokenized_phrases: Dictionary mapping phrases to their tokenized representations
            device: Device to place tensors on
            
        Returns:
            Tensor with penalty values (1.0 for tokens to penalize, 0.0 otherwise)
        """
        token_tensor = torch.tensor(token_ids, device=device)
        # Initialize with zeros (no penalty)
        penalty_tensor = torch.zeros(len(token_ids), dtype=torch.bfloat16, device=device)
        
        for phrase, phrase_tokens_list in tokenized_phrases.items():
            print(phrase)
            print(phrase_tokens_list)
            for phrase_tokens  in phrase_tokens_list:
                phrase_tokens = phrase_tokens.to(device)
                seq_len = len(phrase_tokens)
                
                # Skip if sequence is too short
                if len(token_tensor) < seq_len:
                    continue
                    
                # Create windows for matching
                windows = token_tensor.unfold(0, seq_len, 1)
                
                # Find matches
                matches = (windows == phrase_tokens.unsqueeze(0)).all(dim=1)
                match_positions = matches.nonzero(as_tuple=True)[0]
                
                # Mark matched tokens with penalty of 1.0
                for pos in match_positions:
                    penalty_tensor[pos+1:pos+seq_len+1] = 1.0
        
        return penalty_tensor
    
    def get_language_score(self, prompt_data):
        """
        Calculate language score based on substring matches.
        
        Args:
            prompt_data: Data containing the extracted CoT
            
        Returns:
            Score between 0 and 1, where lower means more penalization
        """
        # Extract the CoT
        cot = prompt_data.extracted_cot
        
        # Count matches of each substring
        n_string_matches = 0
        for penalise_substring in self.penalise_substrings:
            # For whole word/phrase matching
            pattern = r'\b' + re.escape(penalise_substring) + r'\b'
            matches = re.findall(pattern, cot)
            n_string_matches += len(matches)

        
        # Cap at maximum frequency
        n_string_matches = min(n_string_matches, self.max_freq)
        language_penalty = float(n_string_matches / self.max_freq)
        
        return 1 - language_penalty
    
    def get_language_score_temporal_reward(self, prompt_data: PromptData | str, tokenizer: Optional[AutoTokenizer] = None):
        """
        Calculate temporal reward with token-level penalties.
        
        Args:
            prompt_data: Data containing the extracted CoT
            tokenizer: Tokenizer for processing text
            
        Returns:
            Tuple of (score, penalty_tensor) if tokenizer is provided, otherwise just score
        """
        if isinstance(prompt_data, PromptData):
            cot = prompt_data.extracted_cot
        else:
            cot = prompt_data
        
        # Calculate basic score like in get_language_score
        n_string_matches = 0
        matched_phrases = set()
        
        for phrase in self.penalise_substrings:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            matches = re.findall(pattern, cot)
            n_string_matches += len(matches)
            if matches:
                matched_phrases.update(matches)

            
        n_string_matches = min(n_string_matches, self.max_freq)
        language_penalty = float(n_string_matches / self.max_freq)
        score = 1 - language_penalty
        
        # If no tokenizer provided, just return the score
        if tokenizer is None:
            return score
        
        # Create the penalty tensor if tokenizer is provided
        assert self.temporal_reward_penalisation_offset is not None, "Temporal reward offset must be provided"
        device = self.temporal_reward_penalisation_offset.device
        
        # Tokenize the CoT
        tokenized = tokenizer(cot, add_special_tokens=False)
        token_ids = tokenized['input_ids']
        
        # Tokenize the matched phrases
        if matched_phrases:
            tokenized_phrases = self.tokenize_phrases(tokenizer, matched_phrases)
            
            # Create penalty tensor
            raw_penalty_tensor = self.create_penalisation_tensor(
                token_ids=token_ids,
                tokenized_phrases=tokenized_phrases,
                device=device
            )
        else:
            # No penalties to apply
            raw_penalty_tensor = torch.zeros(len(token_ids), dtype=torch.bfloat16, device=device)
        
        # Apply offset to the penalty tensor
        final_penalty_tensor = raw_penalty_tensor - self.temporal_reward_penalisation_offset

        # Could do this
        # Where mask is 1, apply high_penalty; where mask is 0, apply baseline_reward
        """final_penalty_tensor = torch.where(
            raw_penalty_tensor > 0,
            torch.tensor(high_penalty, dtype=torch.bfloat16, device=device),
            torch.tensor(baseline_reward, dtype=torch.bfloat16, device=device)
        )"""
        
        return score, final_penalty_tensor

class TestToM:
    def __init__(self, nouns_path: str, nouns_penalisation_type: str, tokenizer: AutoTokenizer, input_test_file: str, output_test_file: str,  max_freq=50, temporal_reward_penalisation_offset: Optional[float] = None, **kwargs):
        self.ToMTokenBanObject = ToMTokenBanTask(
            nouns_path=nouns_path, 
            nouns_penalisation_type=nouns_penalisation_type,
            max_freq=max_freq,
            temporal_reward_penalisation_offset=temporal_reward_penalisation_offset
            )
        self.tokenizer = tokenizer
        self.input_test_file = input_test_file
        self.output_test_file = output_test_file

    def check_penalisation_tensor(self, sentences_list: List[str]):
        rsts = []
        for sentence in sentences_list:
            tokenized = self.tokenizer(sentence, add_special_tokens=False)
            _ , penalisation_tensor = self.ToMTokenBanObject.get_language_score_temporal_reward(
                prompt_data=sentence,
                tokenizer=self.tokenizer
            )
            rst = [(self.tokenizer.decode(tokenized["input_ids"][i]), penalisation_tensor[i+1].item()) if i != len(penalisation_tensor)-1 else (self.tokenizer.decode(tokenized["input_ids"][i]), 0)
                   for i in range(len(penalisation_tensor))
                ]
            rsts.append(rst)
        return rsts
    
    def unittest(self):
        with open(self.input_test_file, 'r') as f:
            self.sentences_list = f.readlines()
        rsts = self.check_penalisation_tensor(self.sentences_list)
        print(rsts)
        with open(self.output_test_file, 'w') as f:
            for rst in rsts:
                f.write(str(rst) + "\n")


            


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    input_test_file = "mars_steg/language/language_aspects/input_test_cases.txt"
    output_test_file = "mars_steg/language/language_aspects/output_test_cases.txt"
    TestToMObject = TestToM(
        nouns_path= "mars_steg/dataset/theory_of_mind_nouns.yaml",
        nouns_penalisation_type= "names",
        tokenizer=tokenizer,
        temporal_reward_penalisation_offset= 0.022,
        input_test_file=input_test_file,
        output_test_file=output_test_file
    )
    TestToMObject.unittest()
    