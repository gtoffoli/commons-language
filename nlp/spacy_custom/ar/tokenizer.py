# define and replace the Arabic tokenizer

import os
import re
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
 
from .camel_tools.utils.charsets import UNICODE_LETTER_CHARSET
from .camel_tools.utils.dediac import dediac_ar

TATWEEL = u'\u0640' # 'ـ' Tatweel/Kashida character (esthetic character elongation for improved layout)
ALEF_SUPER = u'\u0670' # ' ' Arabic Letter superscript Alef

class MsaTokenizer:

    def __init__(self):
        print('--- init MsaTokenizer ---')
        self.count = 0
        from .camel_tools.disambig.mle import MLEDisambiguator
        from .camel_tools.tokenizers.morphological import MorphologicalTokenizer
        self.nlp = spacy.blank("ar")
        mle_msa = MLEDisambiguator.pretrained('calima-msa-r13')
        self.msa_atb_tokenizer = MorphologicalTokenizer(disambiguator=mle_msa, scheme='atbtok', split=True)

    def __call__(self, text):
        self.count += 1
        # doc = self.nlp(text)
        doc = self.nlp(dediac_ar(text)) # is dediac_ar really needed?
        raw_tokens = [t.text for t in doc if t.text]
        n_raw_tokens = len(raw_tokens)
        raw_tokens_text = ''.join(raw_tokens)
        words = []
        spaces = []
        morphos = self.msa_atb_tokenizer.tokenize(raw_tokens)
        n_morphos = len(morphos)
        tat_weel_spans = []
        i_raw = 0 # index of token in simple tokenization
        raw_token = doc[i_raw]
        raw_text = raw_token.text
        raw_idx = raw_token.idx
        raw_len = len(raw_text)
        raw_space = raw_token.whitespace_

        morphos_chars = 0
        i_morpho = 0 # morpho index
        l_morphos = 0
        for morpho in morphos:
            assert len(morpho) > 0
            if morpho and len(morpho) > 1:
                if morpho[0] == '+' and not raw_text[l_morphos] == '+':
                    morpho = morpho[1:]
                elif morpho[-1] == '+' and not raw_text[l_morphos+len(morpho)-1] == '+':
                    morpho = morpho[:-1]
            l_morpho = len(morpho)
            try:
                assert l_morpho <= raw_len
            except:
                print('!', morphos_chars, l_morphos, raw_len, i_raw, raw_text, morpho)
            morpho_source = raw_tokens_text[morphos_chars : morphos_chars+l_morpho]
            assert l_morpho > 0
            words.append(morpho_source)
            morphos_chars += l_morpho
            l_morphos += l_morpho
            i_morpho += 1
            if l_morphos == raw_len:
                spaces.append(raw_space)
            else:
                spaces.append('')

            if l_morphos > raw_len:
                print('!!!', morphos_chars, l_morphos, raw_len, i_raw, raw_text, morpho)
                break
                     
            if l_morphos == raw_len:
                l_morphos = 0
                i_raw += 1
                if i_raw < n_raw_tokens:
                    raw_token = doc[i_raw]
                    raw_text = raw_token.text
                    raw_idx = raw_token.idx
                    raw_len = len(raw_text)
                    raw_space = raw_token.whitespace_
        if False: # self.count == 6221:
            tokens_chars = 0
            token_list = []
            for token in doc:
                token_list.append([tokens_chars, len(token.text), token.text])
                tokens_chars += len(token.text)
            print(token_list)
            morphos_chars = 0
            morpho_list = []
            for morpho in morphos:
                morpho_list.append([morphos_chars, len(morpho), morpho])
                morphos_chars += len(morpho)
            print(morpho_list)
            words_chars = 0
            word_list = []
            for word in words:
                word_list.append([words_chars, len(word), word])
                words_chars += len(word)
            print(word_list)
        morpho_doc = Doc(Vocab(), words=words, spaces=spaces)
        if False: # self.count == 6221:
            print([[token.idx, len(token.text), token.text] for token in morpho_doc])
        print('---', self.count, len(doc.text), len(morpho_doc.text))
        return morpho_doc

@spacy.registry.tokenizers("msa_tokenizer")
def make_msa_tokenizer():

    def create_msa_tokenizer(nlp):
        return MsaTokenizer()

    return create_msa_tokenizer

print(spacy.registry.tokenizers.get("msa_tokenizer"))

def msa_filter_pattern(in_file, out_file, pattern):
    assert in_file and pattern
    char = None
    n_matches = n_removed = 0
    while 1:
        prev = char
        char = in_file.read(1)          
        if not char: 
            break
        if char==pattern:
            n_matches += 1
            if prev in UNICODE_LETTER_CHARSET or pattern==ALEF_SUPER:
                n_removed += 1
                continue
        if out_file:
            out_file.write(char)
    return n_matches, n_removed

def msa_filter(folder='/_Tecnica/AI/CL/spacy/training/ar', filename='ar_padt-ud-train.conllu', remove=False):
    in_path = os.path.join(folder, filename)
    in_file = open(in_path, 'r', encoding='utf-8')
    i = 1
    for pattern, pat_name in ((TATWEEL, 'TATWEEL'), (ALEF_SUPER, 'ALEF_SUPER')):
        if remove:
            out_path = os.path.join(folder, filename+'.'+str(i))
            out_file = open(out_path, 'w', encoding='utf-8')
        n_matches, n_removed = msa_filter_pattern(in_file, remove and out_file, pattern)
        print(pat_name, '- found:', n_matches, '- removed:', n_removed)
        if pat_name != 'ALEF_SUPER': # check it wasn't the last iteration
            if remove:
                out_file.close()
                in_path = out_path
                in_file = open(in_path, 'r', encoding='utf-8')
                i += 1
            else:
                in_file.seek(0)
    in_file.close()
    if remove:
        out_file.close()
  