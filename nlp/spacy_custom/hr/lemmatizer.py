from typing import List, Dict, Tuple

from ...pipeline import Lemmatizer
from ...tokens import Token

""" This lemmatizer was adapted from the Italian one (version ..) and the Polish one (version of April 2021).
    It implements lookup lemmatization based on the inflectional lexicon hrLex 1.3 ().
    The lookup tables were built using the CroatianLemmatizerLookupsBuilder class
    defined in the file lemmatizer_lookups_builder_hr.py (see example code in that same file).
"""

class CroatianLemmatizer(Lemmatizer):

    @classmethod
    def get_lookups_config(cls, mode: str) -> Tuple[List[str], List[str]]:
        if mode == "pos_lookup":
            required = [ 
                "lemma_lookup_adj", "lemma_lookup_adp", "lemma_lookup_adv", "lemma_lookup_aux",
                "lemma_lookup_conj", "lemma_lookup_cconj", "lemma_lookup_det", "lemma_lookup_intj",
                "lemma_lookup_noun", "lemma_lookup_num", "lemma_lookup_part", "lemma_lookup_pron",
                "lemma_lookup_propn",  "lemma_lookup_punct", "lemma_lookup_sconj", "lemma_lookup_verb", 
            ]
            return (required, [])
        else:
            return super().get_lookups_config(mode)

    def pos_lookup_lemmatize(self, token: Token) -> List[str]:
        string = token.text
        univ_pos = token.pos_
        lookup_pos = univ_pos.lower()
        lookup_table = self.lookups.get_table("lemma_lookup_" + lookup_pos, {})
        if string in lookup_table:
            return [lookup_table[string]]
        elif string != string.lower() and string.lower() in lookup_table:
            return [lookup_table[string.lower()]]
        return [string]
