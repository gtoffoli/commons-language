from operator import itemgetter

from django.conf import settings
# from py_babelnet.calls import BabelnetAPI
import babelnet as bn
from babelnet import Language
from babelnet.pos import POS
from babelnet.synset import SynsetType
from babelnet.resources import BabelSynsetID
from babelnet.data.domain import BabelDomain
from babelnet.data.source import BabelSenseSource
from babelnet.data.lemma import BabelLemmaType
from babelnet.data.relation import BabelSynsetRelation
from babelnet.data.relation import BabelPointer
from babelnet.api import OnlineAPI

HIGH_QUALITY = 1

""" A few SynsetID representing "meta" concepts:
    their instances are candidate terms in a domain glossary. """
_TERM = BabelSynsetID('bn:15290634n') # (termine, terminologia) instances: prezzo 
_JARGON = BabelSynsetID('bn:07399267n') # (termine tecnico, gergo) instances: carrucola, ..
_SPECIALTY = BabelSynsetID('bn:14265130n') # (specialità) instances: politica monetaria
_TERM_LIKE_CONCEPTS = [_TERM, _JARGON, _SPECIALTY,]

# FIXME: check if correct
SPACY_BN_POS_MAPPING = {
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

tourism_domains = [
    'ART_ARCHITECTURE_AND_ARCHAEOLOGY',
    'BIOLOGY',
    'ENVIRONMENT_AND_METEOROLOGY',
    'FOOD_DRINK_AND_TASTE',
    'GEOGRAPHY_GEOLOGY_AND_PLACES',
    'NAVIGATION_AND_AVIATION',
    'TRANSPORT_AND_TRAVEL',
]

economy_domains = [
    "BUSINESS_INDUSTRY_AND_FINANCE",
    "COMMUNICATION_AND_TELECOMMUNICATION",
    "CRAFT_ENGINEERING_AND_TECHNOLOGY",
    "MATHEMATICS_AND_STATISTICS",
    "POLITICS_GOVERNMENT_AND_NOBILITY",
    "POSSESSION",
    "TASKS_JOBS_ROUTINE_AND_EVALUATION",
]

def isTerm(synset):
    """ return the term-like meta-concept of which a synset is an instance, if any, or None """
    relations = synset.outgoing_edges(ANY_HYPERNYM)
    for relation in relations:
        if relation.id_target in _TERM_LIKE_CONCEPTS:
            return relation.id_target
    return None

def spacy2babelnet_pos(pos):
    return SPACY_BN_POS_MAPPING.get(pos, '')

def domains_ok(synset):
    for k, v in synset.domains.items():
        if v >= 4.0:
            return True
    return False

def getSynsets():
    synsets = api._get_synsets()

def synsets_to_annotation(synsets):
    """ convert synsets to annotation compatible with json storage """
    return [[s.id.id, s.main_gloss().gloss, s.main_sense().source.source_name, \
              sorted([[k.domain_string, v] for k,v in s.domains.items()], key=itemgetter(1), reverse=True)] \
            for s in synsets]

def query(filter={}, word='', words=[], pos='NOUN', poses=[], lang='EN', from_langs=[], domains=[], sources_in=[], sources_out=[]):
    api = OnlineAPI()
    if not words:
        words = [word]
    if not poses:
        poses = [spacy2babelnet_pos(pos)]
    if lang and not from_langs:
        bn_lang = bn.language.Language.from_iso(lang)
        from_langs = [bn_lang]
    if not filter:
        filter = { 'words': words, 'poses': poses, 'from_langs': from_langs}
    synsets_1 = api._get_synsets(**filter)
    bn_domains = set([BabelDomain[d] for d in domains])
    synsets_2 = [ s for s in synsets_1 if not bn_domains or set(s.domains).intersection(bn_domains) ]
    synsets = []
    for s in synsets_2:
        s_ok = False
        lemma_objects = s.lemmas(bn_lang)
        for lo in lemma_objects:
            if lo.lemma_type == BabelLemmaType.HIGH_QUALITY and \
               ((pos in ['VERB', 'NOUN', 'ADJ',] and s.type == SynsetType.CONCEPT) or \
                (pos in ['PROPN'] and s.type == SynsetType.NAMED_ENTITY)) \
               and domains_ok(s):
                s_ok = True
                break
        if s_ok:
            synsets.append(s)
    print(synsets_to_annotation(synsets))
    return synsets