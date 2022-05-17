import os
from collections import defaultdict, OrderedDict
from random import randint

def settimes_to_iob(in_path):
    """ From the conllx (extended conllu) format of the SETimes.HR treebank,
        extract only the form and iob fields to be used as NER training data
        and split them between train, eval and test files in the 3:2:1 ratio.
        see: https://universaldependencies.org/docs/format.html 
             http://nlp.ffzg.hr/resources/corpora/setimes-hr/
             https://www.clarin.si/repository/xmlui/handle/11356/1183 """
    train_path = in_path+'-train.iob'
    eval_path = in_path+'-eval.iob'
    test_path = in_path+'-test.iob'
    choices = []
    empty = True
    with open(in_path, encoding='utf8') as infile:
        with open(train_path, 'w', encoding='utf8') as trainfile:
            with open(eval_path, 'w', encoding='utf8') as evalfile:
                with open(test_path, 'w', encoding='utf8') as testfile:
                    if not choices:
                        choices = [trainfile, trainfile, trainfile, evalfile, evalfile, testfile]
                    line = infile.readline()
                    while(line):
                        if empty:
                            outfile = choices[randint(0, 5)]
                        fields = line.replace('\n','').split('\t')
                        if len(fields) > 1:
                            id, form, lemma, upos, xpos, iob, head, deprel, deps, misc = fields 
                            line = '\t'.join([form, iob])+'\n'
                            empty = False
                        else:
                            empty = True
                        outfile.write(line)
                        line = infile.readline()
        trainfile.close()
        evalfile.close()
        testfile.close()
    infile.close()

CONLLU_DICT = {
    'ID': 0, # Word index
    'FORM': 1, # Word form or punctuation symbol.
    'LEMMA': 2, # Lemma or stem of word form.
    'UPOS': 3, # Universal part-of-speech tag.
    'XPOS': 4, # Language-specific part-of-speech tag; underscore if not available.
    'FEATS': 5, # List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    'HEAD': 6, # Head of the current word, which is either a value of ID or zero (0).
    'DEPREL': 7, # Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    'DEPS': 8, # Enhanced dependency graph in the form of a list of head-deprel pairs.
    'MISC': 9, # Any other annotation.
    'IOB': 9, # annotation for NER.
}

def normalize_split_hr500k(in_path):
    """ From the conll format of the hr500k.conllu treebank,
        remove the comment lines and keep only sentences with UD annotation, 
        split them between train, eval and test files in the 3:2:1 ratio,
        keep only IOB data from MISC field
        see: https://github.com/reldi-data/hr500k/blob/master/hr500k.conllu """
    file_ext = '.'+in_path.split('.')[-1]
    file_name = in_path.replace(file_ext, '')
    train_path = file_name+'-train'+file_ext
    eval_path = file_name+'-eval'+file_ext
    test_path = file_name+'-test'+file_ext
    choices = []
    empty = True
    skip = False
    with open(in_path, encoding='utf8') as infile:
        with open(train_path, 'w', encoding='utf8') as trainfile:
            outfile = trainfile
            with open(eval_path, 'w', encoding='utf8') as evalfile:
                with open(test_path, 'w', encoding='utf8') as testfile:
                    if not choices:
                        choices = [trainfile, trainfile, trainfile, evalfile, evalfile, testfile]
                    line = infile.readline()
                    while(line):
                        if line.startswith('#'):
                            if line.startswith('# sent_id'):
                                outfile = choices[randint(0, 5)]
                            elif not line.startswith('# text'):
                                pass
                            line = infile.readline()
                            continue
                        else:
                            fields = line.replace('\r','').replace('\n','').split('\t')
                            if len(fields) >= 10:
                                if fields[CONLLU_DICT['HEAD']]=='_' or fields[CONLLU_DICT['DEPREL']]=='_':
                                    skip = True
                                else:
                                    misc_list = fields[CONLLU_DICT['MISC']].split('|')
                                    iob = 'O'
                                    for item in misc_list:
                                        if item.startswith('NER='):
                                            iob = item.replace('NER=', '').upper()
                                            break
                                    fields = fields[:CONLLU_DICT['MISC']]+[iob]+fields[CONLLU_DICT['MISC']+1:]
                                    line = '\t'.join(fields)+'\n'
                                empty = False
                            else:
                                empty = True
                        if empty or not skip:
                            outfile.write(line)
                        if empty:
                            skip = False
                        line = infile.readline()
        trainfile.close()
        evalfile.close()
        testfile.close()
    infile.close()

LEXICON_COLS = {
    'FORM': 0, # Word form or punctuation symbol
    'LEMMA': 1, # Lemma or stem of word form
    'MSD': 3, # Morpho-syntactic description (MSD)
    'MSD_FEATS': 3, # MSD features.
    'UPOS': 4, # Universal part-of-speech tag.
    'MORPHO': 5, # Morphological features.
    'FREQUENCY': 6, # Any other annotation.
    'FREQUENCY_PER_MILLION': 7, # Any other annotation.
}

FREQUENCY_INTERVALS = { # level codes are case-insensitive
    'NOUN': [1200, 1200, 1200,],
    'VERB': [500, 500, 500],
    'ADJ': [500, 500, 500],
    'ADV': [120, 120, 120],
}
CEFR_LEVELS = { # level codes are case-insensitive
    0: 'a',
    1: 'b',
    2: 'c',
}

LEX_TRAILER = """voc_hr = [
"""
LEX_TAIL = """]
"""

def frequency_to_level(index, intervals):
    level = 0
    max_index = 0
    for interval in intervals:
        max_index += interval
        if index > max_index:
            level += 1
            if level == len(CEFR_LEVELS):
                return None
    return CEFR_LEVELS[level]  

# see class CroatianLemmatizerLookupsBuilder():in nlp/spacy_custom_hr (repository commons-language)
# from the HR Inflectional Lexicon hrLex v1.3,
# builds a basic lexicon with noun, verb, adjective and adverb lemmas
# tagged with a difficulty level (a, b, c) based on their absolute frequency in the source
class CroatianInflectionalLexicon():
    
    def __init__(self, in_path='/_Lingue/Croatian', in_name='hrLex_v1.3', out_path='', out_name='hrLex_v1.3.txt', max_entries=None, min_frequency=None):
        self.in_path = in_path
        self.in_name = in_name
        self.out_path = out_path or in_path or '.'
        self.out_name = out_name
        self.max_entries = max_entries or 10000
        self.min_frequency = min_frequency or 1000
        self.frequency_lexicon = defaultdict(lambda: defaultdict(int))

    def build_frequency_lexicon(self):
        """ builds in memory an unsorted frequency lexicon structured as follows:
            top-level is a Python dict, where each sub-lexicons is keyed with an UPOS;
            a sub-lexicon is a Python dict where an absolute frequency is keyed by lemma
            
        """
        lexicon_path = os.path.join(self.in_path, self.in_name)
        n_cols = LEXICON_COLS['FREQUENCY_PER_MILLION'] + 1
        with open(lexicon_path, encoding='utf8') as infile:
            line = infile.readline()
            while line:
                fields = line.split('\t')
                if len(fields) >= n_cols:
                    upos = fields[LEXICON_COLS['UPOS']]
                    if upos == 'PROPN':
                        upos = 'NOUN'
                    lemma = fields[LEXICON_COLS['LEMMA']]
                    frequency = int(fields[LEXICON_COLS['FREQUENCY']])
                    self.frequency_lexicon[upos][lemma] += frequency
                line = infile.readline()
        infile.close()
  
    def write_frequency_lexicon(self):
        """ sorts by UPOS the top-level dict, whose entries represent sub-lexicons;
            sorts each sub-lexicon by frequency and truncates it by max_entries and min_frequency;
            writes a flat file whose entries are sorted by UPOS first and by descending frequency:
            in the output lines, the fields 'UPOS', 'LEMMA' and FREQUENCY are separated by tabs.
            """
        lexicon_path = os.path.join(self.out_path, self.out_name)
        with open(lexicon_path, 'w', encoding='utf8') as outfile:
            for upos in sorted(self.frequency_lexicon.keys()):
                upos_lexicon = self.frequency_lexicon[upos]
                for lemma, frequency in sorted(upos_lexicon.items(), key=lambda x: x[1], reverse=True)[:self.max_entries]:
                    if frequency < self.min_frequency:
                        break
                    line = "{}\t{}\t{}\n".format(lemma, upos, frequency)
                    outfile.write(line)
        outfile.close()
      
    def write_basic_vocabulary(self):
        """ splits the frequency lexicon in 4 sorted sub-lexicons
            and prints them as a text file containing only comments (#) 
            and lines of 4 tab-separated fields: lemma, upos, level, frequency """
        vocabulary_path = os.path.join(self.out_path, self.out_name)
        with open(vocabulary_path, 'w', encoding='utf8') as outfile:
            outfile.write(LEX_TRAILER)
            for tag in ['NOUN', 'VERB', 'ADJ', 'ADV',]:
                tag_levels = FREQUENCY_INTERVALS[tag]
                tag_lexicon = self.frequency_lexicon[tag]
                index = 0
                for lemma, frequency in sorted(tag_lexicon.items(), key=lambda x: x[1], reverse=True)[:self.max_entries]:
                    level = frequency_to_level(index, tag_levels)
                    if not level:
                        break
                    index += 1
                    line = "'{}'\t{}\t{}\t{}\n".format(lemma, tag, level, frequency)
                    outfile.write(line)
            outfile.write(LEX_TAIL)
        outfile.close()
