from collections import OrderedDict
import os
import json
from spacy.lookups import Lookups


"""
import nlp
from nlp.spacy_custom.it.scripts import lemmatizer_lookups_builder
path = '\\_Tecnica\\AI\\CL\\spacy\\lemmatizer\\it'
in_name = 'morph-it.txt'
builder = lemmatizer_lookups_builder(in_path=path, out_path=path, in_name=in_name)
# >>> builder.morphit_pos_set()
# {'ABL', 'SENT', 'NUM', 'SYM', 'NPR', 'POSS', 'PON', 'ADV', 'VER', 'COM', 'SI', 'NOUN', 'MOD', 'AUX', 'TALE', 'PERS', 'INT', 'SMI', 'ASP', 'CARD', 'DEMO', 'CAU', 'ADJ', 'ARTPRE', 'CI', 'PRE', 'DET', 'ART', 'CON', 'INDEF', 'CE', 'CLI', 'PRO', 'CHE', 'WH', 'NE'}
builder.extract_lookup_tables()
builder.compute_legacy()
builder.make_lookups_bin()
"""


class lemmatizer_lookups_builder():
    
    def __init__(self, in_path='.', out_path='.', in_name='morph-it.txt', tag_map={}):
        self.in_path = in_path
        self.out_path = out_path
        self.in_name = in_name
        self.tag_map = tag_map or {
           'ADJ': ['ADJ', 'DET', 'TALE',],
           'ADV': ['ADV',],
           'ART': ['ART', 'ARTPRE',],
           'AUX': ['AUX', 'ASP', 'CAU', 'MOD',],
           'DET': ['DET',],
           'NOUN': ['NOUN', 'NPR',],
           'PRON': ['PRO',],
           'ADP': ['PRE', 'ARTPRE'],
           'VERB': ['VER', 'AUX', 'ASP', 'CAU', 'MOD',],
           'NUM': ['NUM',],
           'OTHER': ['ABL', 'SENT', 'SYM', 'PON', 'SI', 'INT', 'SMI', 'CON', 'WH', 'NE',],
        }

    def morphit_pos_set(self):
        morph_it_path = os.path.join(self.in_path, 'morph-it.txt')
        pos_set = set()
        with open(morph_it_path) as infile:
            line = infile.readline()
            while line:
                word, lemma, morph = line.replace('\n','').split('\t')
                pos_list = morph.split(':')[0].split('-')
                for pos in pos_list:
                    if pos.isupper() and len(pos)>1:
                        pos_set.add(pos)
                line = infile.readline()
        return pos_set

    def extract_pos_from_morphit(self, lookup_pos, lexicon_pos):
        out_dict = OrderedDict()
        line = self.infile.readline()
        while line:
            try:
                word, lemma, morph = line.split('\t')
                for pos in lexicon_pos:
                    if pos in morph:
                        out_dict[word] = lemma
            except:
                print(len(line.split()), line.split())
            line = self.infile.readline()
        return out_dict

    def extract_lookup_tables(self, filename_pattern='it_lemma_lookup_{}.json'):
        for lookup_pos, lexicon_pos in self.tag_map.items():
            self.infile = open(os.path.join(self.in_path, self.in_name), 'r')
            filename = filename_pattern.format(lookup_pos.lower())
            out_dict = self.extract_pos_from_morphit(lookup_pos, lexicon_pos)
            with open(os.path.join(self.out_path, filename), 'w') as outfile:
                outfile.write(json.dumps(out_dict, indent=2))
            self.infile.close()

    def compute_legacy(self):
        it_lookup_path = os.path.join(self.out_path, 'it_lemma_lookup.json')
        with open(it_lookup_path) as infile:
            it_lookup_dict = json.load(infile)
        morph_it_dict = OrderedDict()
        morph_it_path = os.path.join(self.in_path, 'morph-it.txt')
        with open(morph_it_path) as infile:
            line = infile.readline()
            while line:
                word, lemma, morph = line.split('\t')
                morph_it_dict[word] = lemma
                line = infile.readline()
        legacy_dict = OrderedDict()
        for word, lemma in it_lookup_dict.items():
            if word not in morph_it_dict:
                legacy_dict[word] = lemma
        outfile = open(os.path.join(self.out_path, 'it_lemma_lookup_legacy.json'), 'w')
        outfile.write(json.dumps(legacy_dict, indent=2))
        outfile.close()

    def make_lookups_bin(self, lookup_name_pattern='lemma_lookup_{}', filename_pattern='it_lemma_lookup_{}.json'):
        lookups = Lookups()
        lookup_keys = list(self.tag_map.keys())
        lookup_keys.append('LEGACY')
        for lookup_pos in lookup_keys:
            lookup_name = lookup_name_pattern.format(lookup_pos.lower())
            filename = filename_pattern.format(lookup_pos.lower())
            with open(os.path.join(self.out_path, filename)) as json_file:
                lookup_dict = json.load(json_file)
            lookups.add_table(lookup_name, lookup_dict)
        lookups.to_disk(self.out_path, 'lookups.bin')
