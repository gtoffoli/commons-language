# module derived from:
# https://github.com/asterbini/spacy-babelnet/blob/master/spacy_babelnet/babelnet_annotator.py

from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from spacy.tokens import Span

from nlp.babelnet import *

def extended_noun_chuncks(doc):
    """ try to extend noun chunks, mostly by including prepositional clauses;
        for each original noun chunk, return at most itself or one of its possible extensions """
    noun_chunks = []
    for nc in doc.noun_chunks:
        start = nc.start
        start_token = doc[start]
        # Wh-type or personal or possessive pronoun?
        if start_token.pos_ in ['ADV', 'DET', 'PRON',] and (start_token.tag_ in ['PRP', 'PRP$', 'REL', 'INT',] or start_token.tag_.startswith('W')):
            continue
        end = nc.end
        extended = nc
        subtree = list(nc.subtree)
        if len(subtree) > (end - start):
            subtree = sorted(subtree, key=lambda token: token.i)
            n_adps = 0
            for token in subtree:
                if token.i >= end: # consider only tokens following the root
                    if token.pos_ == 'ADP':
                        if n_adps == 2: # limit the number of nested prepositional phrases
                            break
                        n_adps += 1
                    if token.pos_ not in ['ADJ', 'ADP', 'ADV', 'DET', 'NOUN', 'NUM', 'PRON', 'PROPN',]:
                        break
                    # Wh-type adverb, determiner or pronouns? personal or possessive pronoun?
                    if (token.pos_ in ['ADV', 'DET', 'PRON',] and token.tag_.startswith('W')) \
                    or (token.pos_ in ['ADV', 'DET', 'PRON',] and token.tag_ in  ['PRP', 'PRP$', 'REL', 'INT',]):
                        break
                    end = token.i + 1
            if end > nc.end:
                extended = Span(doc, start, end)
            if  settings.DEBUG:
                print('extended', nc.text, extended.text )
        noun_chunks.append(extended)
    return noun_chunks

class BabelnetAnnotator():

    # def __init__(self, model, domains=[], common_nouns=[]):
    def __init__(self, model, domains=[], common_nouns=[], max_vector_norm=30.0):
        self.bn_lang = bn.language.Language.from_iso(model.lang)
        self.bn_domains = set([BabelDomain[d] for d in domains])
        self.common_nouns = common_nouns
        self.max_vector_norm = max_vector_norm
        self.cached_synsets = {}
        self.spans = []
        self.requests = 0
        Span.set_extension('babelnet', default=None, force=True)
        self.api = OnlineAPI()

    def __call__(self, doc: Doc):
        self.doc = doc

        # process multi-token nounchunks
        for noun_chunk in extended_noun_chuncks(doc):
            if len(noun_chunk) == 1 and doc[noun_chunk.start].vector_norm > self.max_vector_norm:
                continue # to minimize the number of api calls
            spans = self.noun_chunk_annotated_spans(noun_chunk)
            if spans:
                for span, synsets in spans:
                    annotation = synsets_to_annotation(synsets)
                    if annotation:
                        span._.set('babelnet', annotation)
                        self.spans.append(span)

        # process single-token nounchunks
        for token in doc:
            if not token.pos_ in ['VERB', 'NOUN', 'PROPN', 'ADJ', 'ADV',]:
                continue
            if token.pos_ in ['ADJ', 'ADV',] and token.tag_ in ['REL', 'INT',]:
                continue
            if len(token.text) <= 3:
                continue
            if token.vector_norm > self.max_vector_norm:
               continue # to minimize the number of api calls
            synsets = self.token_synsets(token)
            annotation = synsets_to_annotation(synsets)
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
        pos = None
        noun_lemma = None
        for token in noun_chunk:
            if token.pos_ == 'PROPN':
                pos = 'PROPN' # to match SynsetType.NAMED_ENTITY
            elif token.pos_ == 'NOUN' and not noun_lemma:
                noun_lemma = token.lemma_
        if not pos == 'PROPN':
            pos = 'NOUN' # default, to match SynsetType.CONCEPT
            if noun_lemma and noun_lemma in self.common_nouns:
                return [] # to minimize the number of api calls
        while start < end:
            if self.doc[start].pos_ in ['ADP']: # filters cannot start with a preposition
                start += 1
                continue
            key = '_'.join(['{}_{}'.format(self.doc[i].lemma_, self.doc[i].pos_) for i in range(start, end)])
            if key in self.cached_synsets:
                synsets = self.cached_synsets[key]
            else:
                synsets = []
                if spacy2babelnet_pos(pos):
                    span = Span(self.doc, start, end)
                    filter = { 'words': [span.text], 'poses': [spacy2babelnet_pos(pos)], 'from_langs': [self.bn_lang]}
                    if not span.text.startswith(span[0].lemma_):
                        filter['words'].append(span.text.replace(span[0].text, span[0].lemma_, 1))
                    if  settings.DEBUG:
                        print('pos = ', pos, 'filter = ', filter)
                    synsets_1 = self.api._get_synsets(**filter)
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
            if not spacy2babelnet_pos(pos):
                return []
            filter = { 'words': words, 'poses': [spacy2babelnet_pos(pos)], 'from_langs': [self.bn_lang]}
            synsets_1 = self.api._get_synsets(**filter)
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
                       ((pos in ['VERB', 'NOUN', 'ADJ', 'ADV',] and s.type == SynsetType.CONCEPT) or \
                        (pos in ['PROPN'] and s.type == SynsetType.NAMED_ENTITY \
                        #                  and not s.main_sense().source.is_from_wikipedia)) \
                       )) and domains_ok(s):
                        s_ok = True
                        break
                if s_ok:
                    synsets.append(s)
            self.cached_synsets[key] = synsets
        return synsets

from nlp.babelnet import tourism_domains, economy_domains, argumentation_domains

def test1(text='', domains=tourism_domains):
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
    annotator = BabelnetAnnotator(model, domains=domains)
    annotator(doc)
    return doc

def test3():
    return test2(text="Il progresso tecnologico e la moderazione salariale avevano consentito una riduzione dei costi di produzione che, senza l’aumento dei margini di profitto, avrebbero molto probabilmente condotto a un’inflazione ancora più bassa, se non negativa.", domains=economy_domains)

def test4():
    return test2(text="La retórica es un arte muy olvidado e incomprensiblemente poco utilizado, mientras que la argumentación se ha puesto de moda como señal de legitimación del estado de derecho.", domains=argumentation_domains)
