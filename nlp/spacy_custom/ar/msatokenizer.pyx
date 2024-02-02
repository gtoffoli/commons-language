# cython: embedsignature=True, binding=True
# distutils: language=c++
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport cython

import os
import re
import spacy
# from spacy.tokens import Doc
from spacy.vocab import Vocab
# from spacy.tokenizer import Tokenizer

from .camel_tools.utils.charsets import UNICODE_LETTER_CHARSET
from .camel_tools.utils.dediac import dediac_ar
from .camel_tools.disambig.mle import MLEDisambiguator
from .camel_tools.tokenizers.morphological import MorphologicalTokenizer

from cymem.cymem cimport Pool
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc
from libc.string cimport memcpy, memset
from libcpp.set cimport set as stdset
from preshed.maps cimport PreshMap

# import re
from spacy.lexeme cimport EMPTY_LEXEME
from spacy.strings cimport hash_string
from spacy.tokens.doc cimport Doc

from spacy import util
from spacy.attrs import intify_attrs
from spacy.errors import Errors
from spacy.scorer import Scorer
from spacy.symbols import NORM, ORTH
from spacy.tokens import Span
from spacy.training import validate_examples
from spacy.util import get_words_and_spaces

# define and replace the Arabic tokenizer

TATWEEL = u'\u0640' # 'ـ' Tatweel/Kashida character (esthetic character elongation for improved layout)
ALEF_SUPER = u'\u0670' # ' ' Arabic Letter superscript Alef

# class cdef MsaTokenizer(Tokenizer):
cdef class MsaTokenizer:

    def __init__(self, Vocab vocab):
        self.vocab = vocab
        self.count = 0
        self.nlp = spacy.blank("ar")
        mle_msa = MLEDisambiguator.pretrained('calima-msa-r13')
        self.atb_tokenizer = MorphologicalTokenizer(disambiguator=mle_msa, scheme='atbtok', split=True)

    def __call__(self, text):
        self.count += 1
        doc = self.nlp(text)
        raw_tokens = [t.text for t in doc if t.text]
        n_raw_tokens = len(raw_tokens)
        raw_tokens_text = ''.join(raw_tokens)
        words = []
        spaces = []
        morphos = self.atb_tokenizer.tokenize(raw_tokens)
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
        doc_text = doc.text
        morpho_doc_text = morpho_doc.text
        # print('---', self.count, len(text), len(doc_text), len(morpho_doc_text))
        if morpho_doc_text != text:
            print(text)
            print(doc_text)
            print(morpho_doc_text)
        return morpho_doc

    def pipe(self, texts, batch_size=1000):
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts.
        batch_size (int): Number of texts to accumulate in an internal buffer.
        Defaults to 1000.
        YIELDS (Doc): A sequence of Doc objects, in order.

        DOCS: https://spacy.io/api/tokenizer#pipe
        """
        for text in texts:
            yield self(text)

    def score(self, examples, **kwargs):
        validate_examples(examples, "Tokenizer.score")
        return Scorer.score_tokenization(examples)

    def to_disk(self, path, **kwargs):
        """Save the current state to a directory.

        path (str / Path): A path to a directory, which will be created if
            it doesn't exist.
        exclude (list): String names of serialization fields to exclude.

        DOCS: https://spacy.io/api/tokenizer#to_disk
        """
        path = util.ensure_path(path)
        with path.open("wb") as file_:
            file_.write(self.to_bytes(**kwargs))

    def from_disk(self, path, *, exclude=tuple()):
        """Loads state from a directory. Modifies the object in place and
        returns it.

        path (str / Path): A path to a directory.
        exclude (list): String names of serialization fields to exclude.
        RETURNS (Tokenizer): The modified `Tokenizer` object.

        DOCS: https://spacy.io/api/tokenizer#from_disk
        """
        path = util.ensure_path(path)
        with path.open("rb") as file_:
            bytes_data = file_.read()
        self.from_bytes(bytes_data, exclude=exclude)
        return self

    def to_bytes(self, *, exclude=tuple()):
        """Serialize the current state to a binary string.

        exclude (list): String names of serialization fields to exclude.
        RETURNS (bytes): The serialized form of the `Tokenizer` object.

        DOCS: https://spacy.io/api/tokenizer#to_bytes
        """
        """
        serializers = {
            "vocab": lambda: self.vocab.to_bytes(exclude=exclude),
            "prefix_search": lambda: _get_regex_pattern(self.prefix_search),
            "suffix_search": lambda: _get_regex_pattern(self.suffix_search),
            "infix_finditer": lambda: _get_regex_pattern(self.infix_finditer),
            "token_match": lambda: _get_regex_pattern(self.token_match),
            "url_match": lambda: _get_regex_pattern(self.url_match),
            "exceptions": lambda: dict(sorted(self._rules.items())),
            "faster_heuristics": lambda: self.faster_heuristics,
        }
        """
        serializers = {
            "vocab": lambda: self.vocab.to_bytes(exclude=exclude),
        }
        return util.to_bytes(serializers, exclude)

    def from_bytes(self, bytes_data, *, exclude=tuple()):
        """Load state from a binary string.

        bytes_data (bytes): The data to load from.
        exclude (list): String names of serialization fields to exclude.
        RETURNS (Tokenizer): The `Tokenizer` object.

        DOCS: https://spacy.io/api/tokenizer#from_bytes
        """
        data = {}
        """
        deserializers = {
            "vocab": lambda b: self.vocab.from_bytes(b, exclude=exclude),
            "prefix_search": lambda b: data.setdefault("prefix_search", b),
            "suffix_search": lambda b: data.setdefault("suffix_search", b),
            "infix_finditer": lambda b: data.setdefault("infix_finditer", b),
            "token_match": lambda b: data.setdefault("token_match", b),
            "url_match": lambda b: data.setdefault("url_match", b),
            "exceptions": lambda b: data.setdefault("rules", b),
            "faster_heuristics": lambda b: data.setdefault("faster_heuristics", b),
        }
        """
        deserializers = {
            "vocab": lambda b: self.vocab.from_bytes(b, exclude=exclude),
        }
        # reset all properties and flush all caches (through rules),
        # reset rules first so that _reload_special_cases is trivial/fast as
        # the other properties are reset
        self.rules = {}
        self.prefix_search = None
        self.suffix_search = None
        self.infix_finditer = None
        self.token_match = None
        self.url_match = None
        util.from_bytes(bytes_data, deserializers, exclude)
        """
        if "prefix_search" in data and isinstance(data["prefix_search"], str):
            self.prefix_search = re.compile(data["prefix_search"]).search
        if "suffix_search" in data and isinstance(data["suffix_search"], str):
            self.suffix_search = re.compile(data["suffix_search"]).search
        if "infix_finditer" in data and isinstance(data["infix_finditer"], str):
            self.infix_finditer = re.compile(data["infix_finditer"]).finditer
        if "token_match" in data and isinstance(data["token_match"], str):
            self.token_match = re.compile(data["token_match"]).match
        if "url_match" in data and isinstance(data["url_match"], str):
            self.url_match = re.compile(data["url_match"]).match
        if "faster_heuristics" in data:
            self.faster_heuristics = data["faster_heuristics"]
        # always load rules last so that all other settings are set before the
        # internal tokenization for the phrase matcher
        if "rules" in data and isinstance(data["rules"], dict):
            self.rules = data["rules"]
        """
        return self

@spacy.registry.tokenizers("msa_tokenizer")
def make_msa_tokenizer():

    def create_msa_tokenizer(nlp):
        return MsaTokenizer(nlp.vocab)

    return create_msa_tokenizer

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
  