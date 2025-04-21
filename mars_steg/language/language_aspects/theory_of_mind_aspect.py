
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
    
    def get_language_score_temporal_reward(self, prompt_data, tokenizer: Optional[AutoTokenizer] = None):
        """
        Calculate temporal reward with token-level penalties.
        
        Args:
            prompt_data: Data containing the extracted CoT
            tokenizer: Tokenizer for processing text
            
        Returns:
            Tuple of (score, penalty_tensor) if tokenizer is provided, otherwise just score
        """
        cot = prompt_data.extracted_cot
        
        # Calculate basic score like in get_language_score
        n_string_matches = 0
        matched_phrases = set()
        
        for phrase in self.penalise_substrings:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            matches = re.findall(pattern, cot)
            n_string_matches += len(matches)
            if matches:
                matched_phrases.update(matches)

            # pattern_with_space = r'(?<=\s)' + re.escape(phrase) + r'\b'
            # matches_with_space = re.findall(pattern_with_space, cot)
            # n_string_matches += len(matches_with_space)
            # if matches_with_space:
            #     matched_phrases.update(matches_with_space)
        
        print(matched_phrases)

            
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

class ToMTokenBanTaskOld(LanguageAspect):

    """
    temporal_reward_penalisation_offset is used to offset the effect of kl regularisation on the length of CoT
    """

    compatible_tasks = {"TheoryOfMindTask"}

    uses_local_neural_overseer = False

    def __init__(self, chance_penalisation: float, nouns_path: str, nouns_penalisation_type: str, max_freq=50, temporal_reward_penalisation_offset: Optional[float] = None, **kwargs):
        super().__init__()
        self.chance_penalisation = chance_penalisation
        self.nouns_path = nouns_path
        self.nouns_penalisation_type = nouns_penalisation_type
        with open(nouns_path, 'r') as f:
            nouns = yaml.safe_load(f)
        self.penalise_substrings = set(nouns[nouns_penalisation_type])

        self.max_freq = max_freq
        if temporal_reward_penalisation_offset is not None:
            self.temporal_reward_penalisation_offset = torch.tensor(temporal_reward_penalisation_offset).to(torch.bfloat16)
        else:
            self.temporal_reward_penalisation_offset = None

    def limit_penalise_substrings(self, new_penalise_substrings: List[str]):
        new_instance = ToMTokenBanTask(
            chance_penalisation=self.chance_penalisation,
            nouns_path=self.nouns_path,
            nouns_penalisation_type=self.nouns_penalisation_type,
            max_freq=self.max_freq,
            temporal_reward_penalisation_offset = self.temporal_reward_penalisation_offset
        )
        new_penalise_substrings = set(new_penalise_substrings)
        assert new_instance.penalise_substrings.issuperset(new_penalise_substrings), \
            "Trying to override ToMTokenBanTask.penalise_substrings with unseen before penalisation strings!"
        new_instance.penalise_substrings = new_penalise_substrings

        return new_instance

    def create_penalisation_tensor(self, token_ids:torch.Tensor, lists_of_tokens_to_penalize: List[torch.Tensor])-> torch.Tensor:

        token_ids = torch.tensor(token_ids).to("cpu")
        penalisation_tensor = torch.zeros_like(token_ids, dtype=torch.bool)
        lists_of_tokens_to_penalize = [t.to("cpu") for t in lists_of_tokens_to_penalize]
        for list_of_tokens_to_penalize in lists_of_tokens_to_penalize:

            if random.random() > self.chance_penalisation:
                # Only penalize with probability chance_penalised
                continue

            # Get sequence length
            seq_len = list_of_tokens_to_penalize.shape[0]

            # Use unfold to create a rolling window view (efficiently without extra memory copies)
            windows = token_ids.unfold(0, seq_len, 1)  # Shape: [num_windows, seq_len]

            # Compare each window with the target sequence
            matches = (windows == list_of_tokens_to_penalize).all(dim=1)
            start_indices = matches.nonzero(as_tuple=True)[0]

            # Create a mask for penalization
            penalized_one_sequence_token_tensor = torch.zeros_like(token_ids, dtype=torch.bool)

            # Penalize entire matched windows
            for idx in start_indices:
                penalized_one_sequence_token_tensor[idx: idx + seq_len] = True  # Set mask for all elements in the window
            penalisation_tensor+=penalized_one_sequence_token_tensor
        return penalisation_tensor

    def get_language_score(self, prompt_data):

        if self.chance_penalisation == 1.0:
            raise ValueError("chance_penalise cannot be 1.0")

        # Count equals signs in cot
        cot = prompt_data.extracted_cot

        # TODO: 2025.03.29 - make string matches work with multiple strings
        n_string_matches = 0
        for penalise_substring in self.penalise_substrings:
            n_string_matches += cot.count(penalise_substring)

        # We want to drive down the number of equals signs
        # We will punish based on use up to a maximum number of times
        n_string_matches = min(n_string_matches, self.max_freq)
        language_penality = float(n_string_matches / self.max_freq)
        return  1 - language_penality

    def get_language_score_temporal_reward(self, prompt_data, tokenizer: Optional[AutoTokenizer] = None):

        # Count equals signs in cot
        if self.chance_penalisation == 1.0:
            raise ValueError("chance_penalise cannot be 1.0")
        
        cot = prompt_data.extracted_cot

        if tokenizer is not None:
            
            assert self.temporal_reward_penalisation_offset is not None

            tokenized = tokenizer(cot)
            tokenized_cot = tokenized['input_ids']
            matches_token = [re.findall(keyword, cot) for keyword in self.penalise_substrings]
            list_word_penalise = []
            for words_to_penalise in matches_token:
                list_word_penalise+=words_to_penalise
            list_word_penalise = list(set(list_word_penalise))

            special_tokens = list(tokenizer.all_special_ids)
            token_to_penalise = [tokenizer(" "+ word)['input_ids'] for word in list_word_penalise]
            token_to_penalise = [torch.tensor([token for token in list_token if token not in special_tokens]).to("cuda") for list_token in token_to_penalise]

            penalisation_tensor = self.create_penalisation_tensor(
                token_ids=tokenized_cot, 
                lists_of_tokens_to_penalize=token_to_penalise
            ).to(torch.bfloat16) - self.temporal_reward_penalisation_offset # Offset removed = small reward for each token
        
        n_string_matches = 0
        for penalise_substring in self.penalise_substrings:
            n_string_matches += cot.count(penalise_substring)

        n_string_matches = min(n_string_matches, self.max_freq)
        language_penality = float(n_string_matches / self.max_freq)

        if tokenizer is not None:
            return 1 - language_penality, penalisation_tensor
        else:
            return 1 - language_penality


    