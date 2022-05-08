"""
The code below aims to document how the lookup tables for the Croatian POS-aware Lemmatizer
have been derived from the morphological lexicon hrLex 1.3
(see https://www.clarin.si/repository/xmlui/handle/11356/1232).
The tag_map in the creator method of the CroatianLemmatizerLookupsBuilder class specifies a mapping
from the universal tag-set used by spaCy to the tag-set used by hrLex 1.3.
It is a one-to-many mapping; overlaps aim to tackle possible misalignments in tag assignment.
"""

from collections import OrderedDict
import os
import json
from spacy.lookups import Lookups

class CroatianLemmatizerLookupsBuilder():
    
    def __init__(self, in_path='/_Lingue/Croatian', out_path='', in_name='hrLex_v1.3', tag_map={}):
        self.in_path = in_path
        self.out_path = out_path or in_path or '.'
        self.in_name = in_name
        self.tag_map = tag_map or {
           'ADJ': ['ADJ', 'DET',],
           'ADP': ['ADP',],
           'ADV': ['ADV',],
           'AUX': ['AUX',],
           'CONJ': ['CONJ', 'CCONJ', 'SCONJ',],
           'CCONJ': ['CCONJ',],
           'DET': ['DET',],
           'INTJ': ['INTJ',],
           'NOUN': ['NOUN', 'PROPN',],
           'NUM': ['NUM',],
           'PART': ['PART',],
           'PRON': ['PRON',],
           'PROPN': ['PROPN',],
           'PUNCT': ['PUNCT',],        
           'SCONJ': ['SCONJ',],
           'VERB': ['VERB', 'AUX',],
        }

    def lexicon_preview(self, n=100):
        """ This method just prints the first N lines of the lexicon
        >>> from nlp.spacy_custom.hr.lemmatizer_lookups_builder import CroatianLemmatizerLookupsBuilder as builder 
        >>> builder = CroatianLemmatizerLookupsBuilder()
        >>> builder().lexicon_preview()
        """
        lexicon_path = os.path.join(self.in_path, self.in_name)
        i = 0
        with open(lexicon_path) as infile:
            line = infile.readline()
            while line:
                print(line)
                i += 1
                if i > n:
                    break
                line = infile.readline()
        infile.close()

    def lexicon_pos_set(self):
        """ This method just returns the tag-set used by the inflectional lexicon hrLex 1.3:
        >>> builder = CroatianLemmatizerLookupsBuilder()
        >>> builder.lexicon_pos_set()
        {'SCONJ', 'PART', 'NOUN', 'CCONJ', 'INTJ', 'ADJ', 'PROPN', 'DET', 'ADV', 'PRON', 'AUX', 'VERB', 'NUM', 'PUNCT', 'ADP'}
        """
        lexicon_path = os.path.join(self.in_path, self.in_name)
        pos_set = set()
        with open(lexicon_path, encoding='utf8') as infile:
            line = infile.readline()
            while line:
                wordform, lemma, MSD, MSD_features, UPOS, morphological_features, frequency, per_million_frequency = line.replace('\n','').split('\t')
                pos_list = UPOS.split(':')[0].split('-')
                for pos in pos_list:
                    if pos.isupper() and len(pos)>1:
                        pos_set.add(pos)
                line = infile.readline()
        infile.close()
        return pos_set

    def extract_pos_from_lexicon(self, lexicon_pos):
        """ Scans the entire lexicon and
            builds a lookup dict corresponding to an entry in the tag_map. """
        out_dict = OrderedDict()
        line = self.infile.readline()
        while line:
            try:
                wordform, lemma, MSD, MSD_features, UPOS, morphological_features, frequency, per_million_frequency = line.replace('\n','').split('\t')
                for pos in lexicon_pos:
                    if pos == UPOS:
                        out_dict[wordform] = lemma
            except:
                print(len(line.split()), line.split())
            line = self.infile.readline()
        return out_dict

    def extract_lookup_tables(self, filename_pattern='hr_lemma_lookup_{}.json'):
        """ Builds a lookup file (json) for each entry in the tag_map. """
        for lookup_pos, lexicon_pos in self.tag_map.items():
            self.infile = open(os.path.join(self.in_path, self.in_name), 'r', encoding='utf8')
            filename = filename_pattern.format(lookup_pos.lower())
            out_dict = self.extract_pos_from_lexicon(lexicon_pos)
            with open(os.path.join(self.out_path, filename), 'w') as outfile:
                outfile.write(json.dumps(out_dict, indent=2))
            self.infile.close()

    def make_lookups_bin(self, lookup_name_pattern='lemma_lookup_{}', filename_pattern='hr_lemma_lookup_{}.json'):
        """ Merges the tables corresponding to the lookup files created by method "extract_lookup_tables" """
        lookups = Lookups()
        lookup_keys = list(self.tag_map.keys())
        for lookup_pos in lookup_keys:
            lookup_name = lookup_name_pattern.format(lookup_pos.lower())
            filename = filename_pattern.format(lookup_pos.lower())
            with open(os.path.join(self.out_path, filename)) as json_file:
                lookup_dict = json.load(json_file)
            lookups.add_table(lookup_name, lookup_dict)
        lookups.to_disk(self.out_path, 'lookups.bin')

"""
THE BINARY LOOKUPS FILE lookups.bin MUST BE MOVED/COPIED TO ALL CROATIAN LANGUAGE MODELS
AND THE [components.lemmatizer] SECTION OF THE config.cfg FILE MUST BE UPDATED (mode = "pos_lookup").

from nlp.spacy_custom.hr.lemmatizer_lookups_builder_hr import CroatianLemmatizerLookupsBuilder
in_name = 'hrLex_v1.3'
in_path = <path to hrLex 1.3>
out_path = <local github directory for a fork of the spacy_lookups_data repository>
builder = CroatianLemmatizerLookupsBuilder(in_path=in_path, out_path=out_path, in_name=in_name)
builder.extract_lookup_tables()
builder.make_lookups_bin()
"""
