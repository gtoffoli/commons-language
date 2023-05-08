# module derived from:
# https://github.com/asterbini/spacy-babelnet/blob/master/spacy_babelnet/babelnet_annotator.py

from operator import itemgetter
import json

from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from spacy.tokens import Span

import babelnet as bn
from babelnet.pos import POS
from babelnet.data.domain import BabelDomain
from babelnet.data.source import BabelSenseSource
from babelnet.data.lemma import BabelLemmaType
from babelnet.synset import SynsetType
from babelnet.resources import BabelSynsetID
from babelnet.api import OnlineAPI
from _tracemalloc import start

api = OnlineAPI()

HIGH_QUALITY = 1

from django.conf import settings

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

def spacy2babelnet_pos(pos):
    return SPACY_BN_POS_MAPPING.get(pos, '')

def extended_noun_chuncks(doc):
    """ try to extend noun chunks, mostly by including prepositional clauses """
    noun_chunks = []
    for nc in doc.noun_chunks:
        start_token = doc[nc.start]
        if start_token.pos_ == 'PRON' and (start_token.tag_ in ['PR'] or start_token.tag_.startswith('W')):
            continue
        end = nc.end
        extended = nc
        subtree = nc.subtree
        if subtree:
            subtree = sorted(subtree, key=lambda token: token.i)
            for token in subtree:
                if token.i >= end:
                    if token.pos_ not in ['ADJ', 'ADP', 'ADV', 'DET', 'NOUN', 'NUM', 'PRON', 'PROPN',]:
                        break
                    if token.pos_ in ['ADV', 'PRON',] and (token.tag_ in ['PR'] or token.tag_.startswith('W')):
                        break
                    end = token.i + 1
            if end > nc.end:
                extended = Span(doc, nc.start, end)
            if  settings.DEBUG:
                print('extended', nc.text, extended.text )
        noun_chunks.append(extended)
    return noun_chunks

def domains_ok(synset):
    for k, v in synset.domains.items():
        if v >= 4.0:
            return True
    return False

class BabelnetAnnotator():

    def __init__(self, model, domains=[]):
        self.bn_lang = bn.language.Language.from_iso(model.lang)
        self.bn_domains = set([BabelDomain[d] for d in domains])
        self.cached_synsets = {}
        self.spans = []
        self.requests = 0
        Span.set_extension('babelnet', default=None, force=True)

    def synsets_to_annotation(self, synsets):
        """ convert synsets to annotation compatible with json storage """
        return [[s.id.id, s.main_gloss().gloss, s.main_sense().source.source_name, \
                  sorted([[k.domain_string, v] for k,v in s.domains.items()], key=itemgetter(1), reverse=True)] \
                for s in synsets]

    def __call__(self, doc: Doc):
        self.doc = doc

        for noun_chunk in extended_noun_chuncks(doc):
            spans = self.noun_chunk_annotated_spans(noun_chunk)
            if spans:
                for span, synsets in spans:
                    annotation = self.synsets_to_annotation(synsets)
                    if annotation:
                        span._.set('babelnet', annotation)
                        self.spans.append(span)

        for token in doc:
            if not token.pos_ in ['VERB', 'NOUN', 'PROPN', 'ADJ',]:
                continue
            if token.pos_ == 'PRON' and (token.tag_ in ['PR'] or token.tag_.startswith('W')):
                continue
            if token.pos_ == 'AUX' or token.lemma_ in ['essere', 'avere',]:
                continue
            synsets = self.token_synsets(token)
            annotation = self.synsets_to_annotation(synsets)
            if annotation:
                span = Span(doc, token.i, token.i+1)
                span._.set('babelnet', annotation)
                self.spans.append(span)

        if self.spans:
            doc.spans['BABELNET'] = self.spans
        return doc

    def noun_chunk_annotated_spans(self, noun_chunk):
        """ could return several annotated spans for a single noun_chunk
        """
        spans = []
        start = noun_chunk.start
        end = noun_chunk.end
        if len(noun_chunk) == 1:
            pos = self.doc[start].pos_
        else:
            max_children = 0
            for token in noun_chunk:
                n_children = len(list(token.children))
                if n_children > max_children:
                    pos = token.pos_
                    max_children = n_children
        while start < end:
            key = '_'.join(['{}_{}'.format(self.doc[i].lemma_, self.doc[i].pos_) for i in range(start, end)])
            if key in self.cached_synsets:
                synsets = self.cached_synsets[key]
            else:
                synsets = []
                span = Span(self.doc, start, end)
                filter = { 'words': [span.text], 'poses': [spacy2babelnet_pos(pos)], 'from_langs': [self.bn_lang]}
                if not span.text.startswith(span[0].lemma_):
                    filter['words'].append(span.text.replace(span[0].text, span[0].lemma_, 1))
                if  settings.DEBUG:
                    print('filter = ', filter)
                synsets_1 = api._get_synsets(**filter)
                self.requests += 1
                if  settings.DEBUG:
                    print('+', self.requests, filter['words'])
                if synsets_1 and settings.DEBUG:
                    print('+ unfiltered:', [[s.id, s] for s in synsets_1])
                synsets_2 = [ s for s in synsets_1 if set(s.domains).intersection(self.bn_domains) ]
                if synsets_2 and settings.DEBUG:
                    print('+ filtered by domain:', [[s.id, s] for s in synsets_2])
                for s in synsets_2:
                    s_ok = False
                    lemma_objects = s.lemmas(self.bn_lang)
                    for lo in lemma_objects:
                        if lo.lemma_type == BabelLemmaType.HIGH_QUALITY and \
                           ((pos == 'NOUN' and s.type == SynsetType.CONCEPT) or \
                            (pos == 'PROPN' and s.type == SynsetType.NAMED_ENTITY \
                                              and not s.main_sense().source.is_from_wikipedia)) \
                           and domains_ok(s):
                            s_ok = True
                            break
                    if s_ok:
                        synsets.append(s)
                if synsets:
                    spans.append([span, synsets])
                    self.cached_synsets[key] = synsets
                    break
                self.cached_synsets[key] = []
            start += 1
        return spans

    def token_synsets(self, token):
        pos = token.pos_
        lemma = token.lemma_
        words = [lemma]
        if lemma != token.text:
            words.append(token.text)
        key = '{}_{}'.format(lemma, pos)
        if key in self.cached_synsets:
            synsets = self.cached_synsets[key]
        else:
            filter = { 'words': words, 'poses': [spacy2babelnet_pos(pos)], 'from_langs': [self.bn_lang]}
            synsets_1 = api._get_synsets(**filter)
            self.requests += 1
            if settings.DEBUG:
                print('-', self.requests, filter['words'])
            if synsets_1 and settings.DEBUG:
                print('- unfiltered:', [[s.id, s] for s in synsets_1])
            synsets_2 = [ s for s in synsets_1 if set(s.domains).intersection(self.bn_domains) ]
            if synsets_2 and settings.DEBUG:
                print('- filtered by domain:', [[s.id, s] for s in synsets_2])
            synsets = []
            for s in synsets_2:
                s_ok = False
                lemma_objects = s.lemmas(self.bn_lang)
                for lo in lemma_objects:
                    if lo.lemma_type == BabelLemmaType.HIGH_QUALITY and \
                       ((pos in ['VERB', 'NOUN', 'ADJ',] and s.type == SynsetType.CONCEPT) or \
                        (pos in ['PROPN'] and s.type == SynsetType.NAMED_ENTITY \
                                          and not s.main_sense().source.is_from_wikipedia)) \
                       and domains_ok(s):
                        s_ok = True
                        break
                if s_ok:
                    synsets.append(s)
            self.cached_synsets[key] = synsets
        return synsets

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
    
def test2(text='', domains=economy_domains):
    from nlp.utils import text_to_language, make_docbin, addto_docbin
    if not text:
        text = "L'analisi dettagliata dei dati aggregati relativi all’inflazione post-pandemica ha messo in luce una significativa crescita dei margini di profitto."
    language, confidence = text_to_language(text)
    model = settings.LANGUAGE_MODELS[language]
    doc = model(text)
    annotator = BabelnetSpanAnnotator(model, domains=domains)
    annotator(doc)
    return doc

def test3():
    return test2(text="Il progresso tecnologico e la moderazione salariale avevano consentito una riduzione dei costi di produzione che, senza l’aumento dei margini di profitto, avrebbero molto probabilmente condotto a un’inflazione ancora più bassa, se non negativa.", domains=economy_domains)