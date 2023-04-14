# module derived from:
# https://github.com/asterbini/spacy-babelnet/blob/master/spacy_babelnet/babelnet_annotator.py

from spacy.tokens.doc      import Doc
from spacy.tokens.token    import Token
from spacy.parts_of_speech import *
# from spacy.language        import Language
import json

import babelnet as bn
"""
bn.initVM()
DEBUG=True
DEBUG=False
"""
from babelnet.pos import POS
from babelnet.data.domain import BabelDomain
from babelnet.data.source import BabelSenseSource
from babelnet.resources import BabelSynsetID
from babelnet.api import OnlineAPI
api = OnlineAPI()

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
            token._.set(BabelnetAnnotator.__FIELD, babelnet)
            # if DEBUG:
            if settings.DEBUG:
                print(token, babelnet)
        return doc

class Babelnet():
    # FIXME: check if correct
    __SPACY_BN_POS_MAPPING = {
            ADJ   : POS.ADJ, # bn.BabelPOS.ADJECTIVE,
    #       ADP   :
            ADV   : POS.ADV, # bn.BabelPOS.ADVERB,
            AUX   : POS.VERB, # bn.BabelPOS.VERB,
    #       CCONJ :
    #       CONJ  :
    #       DET   : POS.NOUN, # bn.BabelPOS.NOUN,
    #       EOL   :
    #       IDS   :
    #       INTJ  :
    #       NO_TAG:
            NOUN  : POS.NOUN, # bn.BabelPOS.NOUN,
            NUM   : POS.NOUN, # bn.BabelPOS.NOUN,
    #       PART  :
    #       PRON  : POS.NOUN, # bn.BabelPOS.NOUN,
            PROPN : POS.NOUN, # bn.BabelPOS.NOUN,
    #       PUNCT :
    #       SCONJ :
    #       SPACE :
    #       SYM   :
            VERB  : POS.VERB, # bn.BabelPOS.VERB,
    #       X     :
            }

    @classmethod
    def spacy2babelnet_pos(cls, pos):
        return cls.__SPACY_BN_POS_MAPPING.get(pos)
    @classmethod
    def pos2babelnet_pos(cls, pos):
        return cls.__SPACY_BN_POS_MAPPING.get(IDS[pos])

    def __init__(self, token: Token, lang: bn.Language = bn.Language.EN, domains=None):
        self.__token = token
        self.__bn_lang = lang
        self.__bn_domains = domains
        self.__synsets = self.__find_synsets(token)    # we store only the IDs
        self.__lemmas = None # computed only on demand

    def synsets(self):
        # retrieve the BabelnetSynset from their IDs
        # return [ self.__bn.getSynset(bn.BabelSynsetID(s)) for s in self.__synsets ]
        return [ api.getSynset(BabelSynsetID(s)) for s in self.__synsets ]

    def synset_IDs(self):
        return self.__synsets

    def lemmas(self):
        if self.__lemmas is None:
            self.__lemmas = self.__find_lemmas()
        return self.__lemmas

    # we cache the synsets for a (word, pos) pair at class level
    cached_synsets = {}

    @classmethod
    def dump_json(cls, filename):
        '''Dump the cached synsets to json file'''
        with open(filename, mode='w') as F:
            diz = { "|".join([w,p.toString()]) : list(v)
                    for (w,p),v in  cls.cached_synsets.items()}
            return json.dump(diz, F)
    @classmethod
    def load_json(cls, filename):
        '''reload the synsets cache from json file'''
        def rebuild_key(k):
            w,p = k.split('|')
            return w, cls.pos2babelnet_pos(p)
        with open(filename, mode='r') as F:
            diz = json.load(F)
            cls.cached_synsets = { rebuild_key(k) : set(v) for k,v in diz.items() }

    # TODO: define serialization methods
    def to_disk(self):
        # save only:
        # __synsets
        # __token?
        # __bn_source?
        # __bn_lang as string
        pass
    def from_disk(self):
        pass

    def __word_synsets(self, word, pos):
        '''retrieve the sysnsets for a given (word, pos)'''
        """
        if (word,pos) not in self.cached_synsets:
            # we use LKBQuery to be able to select the main SenseSource
            qb = bn.BabelNetQuery.Builder(word)
            qb.POS(pos)
            getattr(qb, 'from')(self.__bn_lang)  # from is a reserved word
            if self.__bn_source:
                qb.source(self.__bn_source)
            if self.__bn_domain:
                qb.tag(self.__bn_domain)
            q = qb.build()
            q = bn.LKBQuery.cast_(q)
            self.cached_synsets[word, pos] = set(bn.BabelSynset.cast_(s).getID().toString()
                                                 for s in self.__lkb.getSynsets(q))
            if DEBUG:
                print(word, pos, self.cached_synsets[word, pos])
        return self.cached_synsets[word, pos]
        """
        key = '{}_{}'.format(word, pos)
        if key not in self.cached_synsets:
            synsets = api._get_synsets(words=[word], poses=[pos], from_langs=[self.__bn_lang])
            print('__word_synsets:', word, pos, key, synsets)
            synsets = [ s for s in synsets if set(s.domains).intersection(self.__bn_domains) ]
            print('filtered_synsets:', synsets)
            self.cached_synsets[key] = set([s.id for s in synsets])
            """
            if settings.DEBUG:
                print(word, pos, self.cached_synsets[key])
            """
        return self.cached_synsets[key]

    def __find_synsets(self, token: Token):
        '''Retrieves the IDs of the token synsets. POS and source are used to restrict the search.'''
        word_variants = [token.text]
        if token.pos in [VERB, NOUN, ADJ]:
            # extend synset coverage using lemmas
            if not token.lemma_ in word_variants:
                word_variants.append(token.lemma_)

        token_synsets = set()
        pos = self.spacy2babelnet_pos(token.pos)
        if pos is not None:
            for word in word_variants:
                token_synsets |= self.__word_synsets(word, pos)
        print('__find_synsets:', word_variants, pos, token_synsets)
        if token_synsets:
            return list(token_synsets)  # sorted?
        return []

    def __find_lemmas(self):
        return list({lemma for synset in self.synsets() for lemma in synset.getLemmas(self.__bn_lang)})

    def __str__(self):
        return f"Babelnet({self.__token}, {self.__token.pos_}, {self.__synsets})"

tourism_domains = [
    'TRANSPORT_AND_TRAVEL',

    'GEOGRAPHY_GEOLOGY_AND_PLACES',
    'NAVIGATION_AND_AVIATION',

    'ART_ARCHITECTURE_AND_ARCHAEOLOGY',
    'ENVIRONMENT_AND_METEOROLOGY',
    'FOOD_DRINK_AND_TASTE',
    'SPORT_GAMES_AND_RECREATION',
]
    
def test():
    from nlp.utils import text_to_language
    text = "En 1496 Caboto partió de Bristol con un buque, pero no logró ir más allá de Islandia y se vio obligado a regresar a causa de disputas con la tripulación."
    language, confidence = text_to_language(text)
    model = settings.LANGUAGE_MODELS[language]
    doc = model(text)
    annotator = BabelnetAnnotator(model, domains=tourism_domains)
    annotator(doc)
    return doc
