# module derived from:
# https://github.com/asterbini/spacy-babelnet/blob/master/spacy_babelnet/babelnet_annotator.py

from operator import itemgetter
import json

from spacy.tokens.doc      import Doc
from spacy.tokens.token    import Token

import babelnet as bn
from babelnet.pos import POS
from babelnet.data.domain import BabelDomain
from babelnet.data.source import BabelSenseSource
from babelnet.data.lemma import BabelLemmaType
from babelnet.synset import SynsetType
from babelnet.resources import BabelSynsetID
from babelnet.api import OnlineAPI

api = OnlineAPI()

HIGH_QUALITY = 1

from django.conf import settings

class BabelnetAnnotator:
    __FIELD = 'babelnet'

    def __init__(self, nlp, domains=[]):
        Token.set_extension(BabelnetAnnotator.__FIELD, default=None, force=True)
        self.__bn_lang = bn.language.Language.from_iso(nlp.lang)
        self.__bn_domains = set([BabelDomain[d] for d in domains])

    def __call__(self, doc: Doc):
        for token in doc:
            babelnet = Babelnet(token=token, lang=self.__bn_lang, domains=self.__bn_domains)
            # token._.set(BabelnetAnnotator.__FIELD, babelnet)
            annotation = [
                [s.id.id, s.main_gloss().gloss, s.main_sense().source.source_name, \
                 sorted([[k.domain_string, v] for k,v in s.domains.items()], key=itemgetter(1), reverse=True)] \
                for s in babelnet.synsets()]
            if settings.DEBUG:
                print('---', token, annotation)
            token._.set(BabelnetAnnotator.__FIELD, annotation)
        return doc

class Babelnet():
    # FIXME: check if correct
    __SPACY_BN_POS_MAPPING = {
            'ADJ'   : POS.ADJ, # bn.BabelPOS.ADJECTIVE,
    #       ADP   :
            'ADV'   : POS.ADV, # bn.BabelPOS.ADVERB,
            'AUX'   : POS.VERB, # bn.BabelPOS.VERB,
    #       CCONJ :
    #       CONJ  :
    #       DET   : POS.NOUN, # bn.BabelPOS.NOUN,
    #       EOL   :
    #       IDS   :
    #       INTJ  :
    #       NO_TAG:
            'NOUN'  : POS.NOUN, # bn.BabelPOS.NOUN,
            'NUM'   : POS.NOUN, # bn.BabelPOS.NOUN,
    #       PART  :
    #       PRON  : POS.NOUN, # bn.BabelPOS.NOUN,
            'PROPN' : POS.NOUN, # bn.BabelPOS.NOUN,
    #       PUNCT :
    #       SCONJ :
    #       SPACE :
    #       SYM   :
            'VERB'  : POS.VERB, # bn.BabelPOS.VERB,
    #       X     :
            }

    @classmethod
    def spacy2babelnet_pos(cls, pos):
        # return cls.__SPACY_BN_POS_MAPPING.get(pos)
        return cls.__SPACY_BN_POS_MAPPING.get(pos, '')

    def __init__(self, token: Token, lang: bn.Language = bn.Language.EN, domains=None):
        self.__token = token
        self.__bn_lang = lang
        self.__bn_domains = domains
        self.__synsets = self.__find_synsets(token)    # we store only the IDs
        self.__lemmas = None # computed only on demand

    def synsets(self):
        # retrieve the BabelnetSynset from their IDs
        return self.__synsets

    # we cache the synsets for a (word, pos) pair at class level
    cached_synsets = {}

    def domains_ok(self, synset):
        for k, v in synset.domains.items():
            if v >= 4.0:
                return True
        return False

    def __word_synsets(self, pos, words, poses):
        '''retrieve the sysnsets for a given (word, pos)'''
        lemma = words[0]
        key = '{}_{}'.format(lemma, pos)
        if key not in self.cached_synsets:
            filter = { 'words': words, 'poses': poses, 'from_langs': [self.__bn_lang]}
            synset_filters = []
            synsets_1 = api._get_synsets(**filter)
            synsets_2 = [ s for s in synsets_1 if set(s.domains).intersection(self.__bn_domains) ]
            if synsets_2:
                print('- filtered by domain:', [[s.id, s] for s in synsets_2])
            synsets = []
            for s in synsets_2:
                # print('- s: ', key, s.id, s.type, s.main_sense().source.source_name)
                s_ok = False
                lemma_objects = s.lemmas(self.__bn_lang)
                for lo in lemma_objects:
                    # print('- lo:', lo.lemma, lo.lemma_type)
                    #? if lo.lemma.lower() == lemma.lower() and lo.lemma_type == BabelLemmaType.HIGH_QUALITY and \
                    if lo.lemma_type == BabelLemmaType.HIGH_QUALITY and \
                       ((pos in ['VERB', 'NOUN', 'ADJ',] and s.type == SynsetType.CONCEPT) or \
                        (pos in ['PROPN'] and s.type == SynsetType.NAMED_ENTITY \
                                          and not s.main_sense().source.is_from_wikipedia)) \
                       and self.domains_ok(s):
                        s_ok = True
                        break
                if s_ok:
                    synsets.append(s)
            self.cached_synsets[key] = synsets
        return self.cached_synsets[key]

    def __find_synsets(self, token: Token):
        '''Retrieves the IDs of the token synsets. POS and source are used to restrict the search.'''
        if not token.pos_ in ['VERB', 'NOUN', 'PROPN', 'ADJ',]:
            return []
        word_variants = [token.lemma_]
        if token.pos_ in ['VERB', 'NOUN', 'ADJ']:
            # extend synset coverage using original text
            if not token.lemma_ == token.text:
                word_variants.append(token.text)

        token_synsets = set()
        pos = self.spacy2babelnet_pos(token.pos_)
        if pos is not None:
            token_synsets = self.__word_synsets(token.pos_, word_variants, [pos])
        return token_synsets

    def __str__(self):
        return f"Babelnet({self.__token}, {self.__token.pos_}, {self.__synsets})"

tourism_domains = [
    'ART_ARCHITECTURE_AND_ARCHAEOLOGY',
    'BIOLOGY',
    'ENVIRONMENT_AND_METEOROLOGY',
    'FOOD_DRINK_AND_TASTE',
    'GEOGRAPHY_GEOLOGY_AND_PLACES',
    'NAVIGATION_AND_AVIATION',
    'TRANSPORT_AND_TRAVEL',
]
    
def test(text='', domains=tourism_domains):
    from nlp.utils import text_to_language, make_docbin, addto_docbin
    if not text:
        text = "En 1496 Caboto partió de Bristol con un buque, pero no logró ir más allá de Islandia y se vio obligado a regresar a causa de disputas con la tripulación."
    language, confidence = text_to_language(text)
    model = settings.LANGUAGE_MODELS[language]
    doc = model(text)
    annotator = BabelnetAnnotator(model, domains=domains)
    annotator(doc)
    file_key, docbin = make_docbin('test_es', language=doc.lang_)
    file_key, docbin = addto_docbin(docbin, doc, file_key)
    return doc
