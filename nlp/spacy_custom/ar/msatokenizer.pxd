from cymem.cymem cimport Pool
from libcpp.vector cimport vector
from preshed.maps cimport PreshMap

from spacy.matcher.phrasematcher cimport PhraseMatcher
from spacy.strings cimport StringStore
from spacy.structs cimport LexemeC, SpanC, TokenC
from spacy.tokens.doc cimport Doc
from spacy.typedefs cimport hash_t
from spacy.vocab cimport LexemesOrTokens, Vocab, _Cached

cdef class MsaTokenizer:
    cdef readonly Vocab vocab
    cdef int count
    cdef object nlp
    cdef object atb_tokenizer
