from typing import List
import yaml

from mars_steg.language.base_language_aspect import LanguageAspect




class ToMTokenBanTask(LanguageAspect):

    compatible_tasks = {"TheoryOfMindTask"}

    uses_local_neural_overseer = False

    def __init__(self, nouns_path: str, nouns_penalisation_type: str, max_freq=50, **kwargs):
        super().__init__()
        self.nouns_path = nouns_path
        self.nouns_penalisation_type = nouns_penalisation_type
        with open(nouns_path, 'r') as f:
            nouns = yaml.safe_load(f)
        self.penalise_substrings = set(nouns[nouns_penalisation_type])
        self.max_freq = max_freq

    def limit_penalise_substrings(self, new_penalise_substrings: List[str]):
        new_instance = ToMTokenBanTask(
            nouns_path=self.nouns_path,
            nouns_penalisation_type=self.nouns_penalisation_type,
            max_freq=self.max_freq
        )
        new_penalise_substrings = set(new_penalise_substrings)
        assert new_instance.penalise_substrings.issuperset(new_penalise_substrings), \
            "Trying to override ToMTokenBanTask.penalise_substrings with unseen before penalisation strings!"
        new_instance.penalise_substrings = new_penalise_substrings

        return new_instance

    def get_language_score(self, prompt_data):

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

