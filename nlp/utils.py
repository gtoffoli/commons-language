import os
import math
from pathlib import Path
import time
import json
import glob
import tempfile
from subprocess import check_output
from collections import defaultdict

from django.conf import settings
# import spacy 
from spacy import displacy
from spacy.tokens import Doc, DocBin, Token, Span
from spacy.attrs import ORTH
# from gensim.summarization import summarize
import pandas as pd
import operator
import re
from langid.langid import LanguageIdentifier, model
try:
    from tmtoolkit.preprocess._docfuncs import _init_doc
except:
    def _init_doc(doc):
        return doc
from nlp.entity_grid import EntityGrid, get_local_coherence

def text_to_list(text):
    if not text:
        return []
    list = text.splitlines()
    list = [item.strip() for item in list]
    return [item for item in list if len(item)]

fasttext_path = '/opt/demo-app/fastText/fasttext'

def text_to_language(text):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    language, confidence = identifier.classify(text)
    if language == 'ur' and not settings.LANGUAGE_MODELS.get('ur', None) and settings.LANGUAGE_MODELS.get('ar', None):
        language = 'ar'
    print('language_detection: ', language, confidence)
    return language, confidence

# uncomment for debugging purporses
import logging
fmt = getattr(settings, 'LOG_FORMAT', None)
lvl = getattr(settings, 'LOG_LEVEL', logging.DEBUG)
logging.basicConfig(format=fmt, level=lvl)


MODEL_MAPPING = {
    'el': '/opt/demo-app/demo/el_classiffier.bin'
}

ENTITIES_MAPPING = {
    'PERSON': 'person',
    'LOC': 'location',
    'GPE': 'location',
    'ORG': 'organization',
}

POS_MAPPING = {
    'NOUN': 'nouns',
    'VERB': 'verbs',
    'ADJ': 'adjectives',
}

SEMANTIC_POS_LIST = ['VERB', 'ADJ', 'NOUN', 'ADV', 'PROPN']

def load_greek_lexicon():
    indexes = {}
    df = pd.read_csv(
        'datasets/sentiment_analysis/greek_sentiment_lexicon.tsv', sep='\t')
    df = df.fillna('N/A')
    for index, row in df.iterrows():
        df.at[index, "Term"] = row["Term"].split(' ')[0]
        indexes[df.at[index, "Term"]] = index
    subj_scores = {
        'OBJ': 0,
        'SUBJ-': 0.5,
        'SUBJ+': 1,
    }

    emotion_scores = {
        'N/A': 0,
        '1.0': 0.2,
        '2.0': 0.4,
        '3.0': 0.6,
        '4.0': 0.8,
        '5.0': 1,
    }

    polarity_scores = {
        'N/A': 0,
        'BOTH': 0,
        'NEG': -1,
        'POS': 1
    }
    return df, subj_scores, emotion_scores, polarity_scores, indexes


df, subj_scores, emotion_scores, polarity_scores, indexes = load_greek_lexicon()

def customize_model(model):
    def set_custom_boundaries(doc):
        """ end sentence at colon and semicolon """
        for token in doc[:-2]:
            if token.text in [':', ';',]:
                doc[token.i+1].is_sent_start = True
        return doc
    try:
        model.add_pipe(set_custom_boundaries, before="parser")
    except:
        pass

def doc_to_json(doc, language):
    json = doc.to_json(underscore=['obj_type', 'obj_id', 'label', 'url'])
    json['language'] = language
    # create a dict of json tokens keyed by character position in doc (idx)
    start_dict = {}
    for token_data in json['tokens']:
        start_dict[token_data['start']] = token_data
    # augment each token in json with additional attributes
    for token in doc:
        token_data = start_dict[token.idx]
        sentence = token.sent
        token_data['sent_start'] = sentence.start
        token_data['sent_end'] = sentence.end
        token_data['stop'] = token.is_stop
        token_data['lemma'] = token.lemma_
        token_data['oov'] = token.is_oov
        token_data['num'] = token.like_num
        token_data['email'] = token.like_email
        token_data['url'] = token.like_url
        token_data['text'] = token.text
    return json # some keys: ['text', 'ents', 'sents', 'tokens', 'language'])

# as a side-effect, extends the language model
def text_to_doc(text, return_json=False, doc_key=None):
    # language = settings.LANG_ID.classify(text)[0]
    language, confidence = text_to_language(text)
    model = settings.LANGUAGE_MODELS[language]
    customize_model(model) # 191111 GT
    doc = model(text)
    _init_doc(doc)
    doc.user_data.update({'doc_key': doc_key})
    if return_json:
        json = doc_to_json(doc, language)
        return json
    else:
        return doc

# see https://github.com/kamal2230/text-summarization/blob/master/Summarisation_using_spaCy.ipynb
def spacy_summarize(doc, sorted_frequencies, max_frequency, limit=None):
    if not sorted_frequencies:
        return ''
    sents = list(doc.sents)
    if len(sents) <= 2:
        summarized_sentences = sents
    else:
        limit = limit or int(math.log(len(sents), 1.8))
        normal_frequencies = {}
        for lemma, frequency in sorted_frequencies:
            normal_frequencies[lemma] = frequency / max_frequency
        sent_strength = defaultdict(float)
        for i, sent in enumerate(sents):
            sent_length = 0
            proper_nouns = []
            for token in sent:
                lemma = token.lemma_
                if lemma in normal_frequencies:
                    # count only once proper nouns
                    if token.pos_ == 'PROPN':
                        if lemma in proper_nouns:
                            continue
                        else:
                            proper_nouns.append(lemma)
                            if sent_length == 0: # proper noun at sentence start counts double
                                sent_strength[i] += normal_frequencies[lemma]
                    sent_strength[i] += normal_frequencies[lemma]
                    sent_length += 1
            if sent_length > 1:
                long_sentence_corrector = math.log(sent_length, 1.4)
                sent_strength[i] = sent_strength[i]/long_sentence_corrector
        sent_keys = [k for k, v in sorted(sent_strength.items(), key=lambda item: item[1], reverse=True)][:limit]
        summarized_sentences = [sent for i, sent in enumerate(sents) if i in sent_keys]
    summary = ' '.join([sent.text for sent in summarized_sentences])
    return summary

def get_doc_attributes(doc):
    # return {'language': doc.lang_, 'n_tokens': doc.__len__(), 'n_words': len(doc.count_by(ORTH))}
    return {'language': doc.lang_, 'n_tokens': doc.__len__(), 'n_words': len(doc.count_by(ORTH)), 'n_synsets': len(doc.spans.get('BABELNET', [])), 'n_terms': len(doc.spans.get('GLOSSARY', []))}

def predict_category(text, language):
    "Loads FastText models and predicts category"
    text = text.lower().replace('\n', ' ')
    # fastText expects a file here
    fp = tempfile.NamedTemporaryFile(delete=False)
    fp.write(str.encode(text))
    fp.close()

    model = MODEL_MAPPING[language]
    cmd = [fasttext_path, 'predict', model, fp.name]
    result = check_output(cmd).decode("utf-8")
    category = result.split('__label__')[1]

    # remove file
    try:
        os.remove(fp.name)
    except Exception:
        pass

    return category

def merge_docs(language=None, docs=[], docbin=None, text=None):
    if docbin:
        assert language is not None
        model = settings.LANGUAGE_MODELS[language]
        docs = docs + list(docbin.get_docs(model.vocab))
    if text:
        docs.append(text_to_doc(text))
    if not docs:
        return -1, None
    if not all(doc[0].lang_ == doc.lang_ for doc in docs):
        return -2, None
    if len(docs) == 1:
        doc = docs[0]
    else:
        doc = Doc.from_docs(docs)
        _init_doc(doc)
    return 0, doc

def add_paragraph_spans(doc):
    """ annotate the doc with multi-sentence spans representing paragraphs;
        a paragraph starts with a sentence whose first token has SPACE pos,
        which presumably derives from one or more newline characters """
    spans = []
    start = 0
    end = 0
    for sent in doc.sents:
        # if doc[sent.start].pos_ == 'SPACE':
        start_token = doc[sent.start]
        end_token = doc[sent.end - 1] 
        if start_token.text.startswith('\n') or start_token.text.startswith('\r'):
            # append span ending with the previous sentence
            if (start_token.pos_=='SPACE' and (end-start) > 1) or (start_token.pos_!='SPACE' and (end-start) > 0):
                spans.append(Span(doc, start, end))
            start = sent.start
        # elif end_token.pos_ == 'SPACE' and ('\n' in end_token.text or '\r' in end_token.text):
        elif end_token.text.endswith('\n') or end_token.text.endswith('\r'):
            # append span ending with the current sentence
            if (end_token.pos_=='SPACE' and (end-start) > 1) or (end_token.pos_!='SPACE' and (end-start) > 0):
                spans.append(Span(doc, start, end))
            start = end
            end = sent.end
        else:
            end = sent.end
    if (end-start) > 1:
        spans.append(Span(doc, start, end))
    doc.spans['PARA'] = spans
    return spans

def analyze_doc(doc=None, text='', glossary=[], keys=[], return_text=True):
    assert doc or text
    if doc:
        text = doc.text
    else:
        doc = text_to_doc(text)
    language = doc.lang_

    ret = {'doc': doc_to_json(doc, language), 'language': language}
    paragraphs = add_paragraph_spans(doc)

    if not keys or 'text' in keys:
        ret['text'] = text

    if not keys or 'analyzed_text' in keys:
        analyzed_text = ''
        for token in doc:
            if token.ent_type_:
                analyzed_text += '<span class="tooltip" data-content="POS: {0}<br> LEMMA: {1}<br> DEP: {2}" style="color: red;" >{3} </span>'.format(
                    token.pos_, token.lemma_, token.dep_, token.text)
            else:
                analyzed_text += '<span class="tooltip" data-content="POS: {0}<br> LEMMA: {1}<br> DEP: {2}" >{3} </span>'.format(
                    token.pos_, token.lemma_, token.dep_, token.text)
    
        ret['analyzed_text'] = analyzed_text

    # Text category. Only valid for Greek text for now
    if (not keys or 'sentiment' in keys) and language == 'el':
        ret.update(sentiment_analysis(doc))
        try:
            ret['category'] = predict_category(text, language)
        except Exception:
            pass

    if not keys or 'keywords' in keys:
        # top 10 most frequent keywords, based on tokens lemmatization
        frequency = defaultdict(int)
        lemma_forms = defaultdict(set)
        lexical_attrs = {
            'urls': [],
            'emails': [],
            'nums': [],
        }
        for token in doc:
            if (token.like_url):
                lexical_attrs['urls'].append(token.text)
            if (token.like_email):
                lexical_attrs['emails'].append(token.text)
            if (token.like_num or token.is_digit):
                lexical_attrs['nums'].append(token.text)
            # if not token.is_stop and token.pos_ in ['VERB', 'ADJ', 'NOUN', 'ADV', 'AUX', 'PROPN']:
            if not token.is_stop and token.pos_ in SEMANTIC_POS_LIST:
                lemma = token.lemma_
                frequency[lemma] += 1
                if 'lemma_forms' in keys:
                    lemma_forms[lemma].add(token.text)
        frequency_items = sorted(frequency.items(), key=lambda k_v: k_v[1], reverse=True)
        if return_text:
            keywords = [item[0] for item in frequency_items[:10]]
            ret['keywords'] = ', '.join(keywords)
        else:
            ret['keywords'] = frequency_items
            if 'lemma_forms' in keys:
                ret['lemma_forms'] = lemma_forms

        sorted_frequencies = sorted(frequency.items(), key=lambda k_v: k_v[1], reverse=True)
        keywords = [keyword for keyword, frequency in sorted_frequencies][:10]
        max_frequency = len(sorted_frequencies) and sorted_frequencies[0][1]

    if not keys or 'summary' in keys:
        try:
            # see: https://stackoverflow.com/questions/69064948/how-to-import-gensim-summarize
            from gensim.summarization import summarize
            ret['summary'] = summarize(text)
        except:
            ret['summary'] = spacy_summarize(doc, sorted_frequencies, max_frequency)

    if not keys or 'entities' in keys:
        # Named Entities
        entities = {label: [] for key, label in ENTITIES_MAPPING.items()}
        for ent in doc.ents:
            # noticed that these are found some times
            if ent.text.strip() not in ['\n', '', ' ', '.', ',', '-', '–', '_']:
                mapped_entity = ENTITIES_MAPPING.get(ent.label_)
                if mapped_entity and ent.text not in entities[mapped_entity]:
                    entities[mapped_entity].append(ent.text)
        ret['named_entities'] = entities

    if not keys or 'sentences' in keys:
        # Sentences splitting
        ret['sentences'] = [sentence.text for sentence in doc.sents]

    if not keys or 'entities' in keys:
        # Lemmatized sentences splitting
        ret['lemmatized_sentences'] = [sentence.lemma_ for sentence in doc.sents]

    if not keys or 'tokenized' in keys:
        # Text tokenization
        ret['text_tokenized'] = [token.text for token in doc]

    if not keys or 'pos' in keys:
        # Parts of Speech
        part_of_speech = {label: [] for key, label in POS_MAPPING.items()}
    
        for token in doc:
            mapped_token = POS_MAPPING.get(token.pos_)
            if mapped_token and token.text not in part_of_speech[mapped_token]:
                part_of_speech[mapped_token].append(token.text)
        ret['part_of_speech'] = part_of_speech

    if not keys or 'lexical_attrs' in keys:
        ret['lexical_attrs'] = lexical_attrs

    if not keys or 'noun_chunks' in keys:
        try:
            noun_chunks = [[span.start, span.end] for span in doc.noun_chunks]
        except:
            noun_chunks = []
        ret['noun_chunks'] = noun_chunks
        ret['paragraphs'] = [{'i': i, 'text': span.text, 'start': span.start, 'end': span.end} for i, span in enumerate(paragraphs)]

    if 'text_cohesion' in keys:
        ret['cohesion_by_similarity'] = local_cohesion_by_similarity(doc, paragraphs=paragraphs)
        local_cohesion, repeated_lemmas = local_cohesion_by_repetitions(doc, paragraphs=paragraphs)
        ret['cohesion_by_repetitions'] = local_cohesion
        ret['repeated_lemmas'] = sorted(repeated_lemmas.items(), key=lambda x: x[1], reverse=True)
        ret['cohesion_by_entity_graph'] = local_cohesion_by_entity_graph(doc)
        ret['paragraphs'] = [{'i': i, 'text': span.text, 'start': span.start, 'end': span.end} for i, span in enumerate(paragraphs)]

    if glossary:
        # serialize annotations with glossary terms
        from .terms_annotator import TermsAnnotator
        model = settings.LANGUAGE_MODELS[language]
        annotator = TermsAnnotator(model, glossary)
        doc = annotator(doc)

    if not keys or 'noun_chunks' in keys:
        spans = doc.spans.get('GLOSSARY', [])
        if spans:
            glossary_matches = []
            for span in spans:
                glossary_matches.append({'concept_id': doc.vocab.strings[span.label], 'start': span.start, 'end': span.end})
            ret['glossary_matches'] = glossary_matches

    if not keys or 'entities' in keys:
        # serialize BabelNet annotations
        bn_terms = doc.spans.get('BABELNET', [])
        if bn_terms:
            Span.set_extension('babelnet', default=None, force=True)
            ret['bn_terms'] = [{'i': i, 'babelnet': span._.babelnet, 'start': span.start, 'end': span.end} for i, span in enumerate(bn_terms)]
 
    return ret

def get_sorted_keywords(language=None, doc=None, docs=[], text=None):
    assert doc or docs or text
    if text:
        pass
    if docs:
        doc = docs[0]
    results = analyze_doc(doc, keys=['keywords', 'lemma_forms'], return_text=False)
    keywords = results['keywords']
    lemma_forms = results['lemma_forms']
    if docs and len(docs)>1:
        kw_dict = defaultdict(int, keywords)
        for doc in docs[1:]:
            results = analyze_doc(doc, keys=['keywords', 'lemma_forms'], return_text=False)
            for lemma, frequency in results['keywords']:
                kw_dict[lemma] += frequency
            for lemma, tokens in results['lemma_forms'].items():
                lemma_forms[lemma] = lemma_forms[lemma].union(tokens)
        keywords = sorted(kw_dict.items(), key=lambda x: x[1], reverse=True)
    return keywords, lemma_forms

def visualize_text(text):
    # language = settings.LANG_ID.classify(text)[0]
    language, confidence = text_to_language(text)
    lang = settings.LANGUAGE_MODELS[language]
    doc = lang(text)
    return displacy.parse_deps(doc)


def sentiment_analysis(doc):

    subjectivity_score = 0
    anger_score = 0
    disgust_score = 0
    fear_score = 0
    happiness_score = 0
    sadness_score = 0
    surprise_score = 0
    matched_tokens = 0
    for token in doc:
        lemmatized_token = token.lemma_
        if (lemmatized_token in indexes):
            indx = indexes[lemmatized_token]
            pos_flag = False
            for col in ["POS1", "POS2", "POS3", "POS4"]:
                if (token.pos_ == df.at[indx, col]):
                    pos_flag = True
                    break
            if (pos_flag):
                match_col_index = [int(s) for s in col if s.isdigit()][0]
                subjectivity_score += subj_scores[df.at[indx,
                                                        'Subjectivity' + str(match_col_index)]]
                anger_score += emotion_scores[str(
                    df.at[indx, 'Anger' + str(match_col_index)])]
                disgust_score += emotion_scores[str(
                    df.at[indx, 'Disgust' + str(match_col_index)])]
                fear_score += emotion_scores[str(
                    df.at[indx, 'Fear' + str(match_col_index)])]
                happiness_score += emotion_scores[str(
                    df.at[indx, 'Happiness' + str(match_col_index)])]
                sadness_score += emotion_scores[str(
                    df.at[indx, 'Sadness' + str(match_col_index)])]
                surprise_score += emotion_scores[str(
                    df.at[indx, 'Surprise' + str(match_col_index)])]
                matched_tokens += 1
    try:
        subjectivity_score = subjectivity_score / matched_tokens * 100
        emotions = {'anger': anger_score, 'disgust': disgust_score, 'fear': fear_score,
                    'happiness': happiness_score, 'sadness': sadness_score, 'surprise': surprise_score}
        emotion_name = max(emotions.items(), key=operator.itemgetter(1))[0]
        emotion_score = emotions[emotion_name] * 100 / matched_tokens
        ret = {'subjectivity': round(subjectivity_score, 2),
               'emotion_name': emotion_name, 'emotion_score': round(emotion_score, 2)}
        # logging.debug(subjectivity_score)
        return ret
    except ZeroDivisionError:
        return {}
    except Exception:
        return {}

def compare_docs(doc_dicts):
    doc_dict_0 = doc_dicts[0]
    doc_0 = doc_dict_0['doc']
    similarities = []
    for doc_dict in doc_dicts[1:]:
        doc = doc_dict['doc']
        similarities.append(doc.similarity(doc_0))
    ret = {'n_docs': len(doc_dicts), 'similarity': similarities[0]}
    return ret

def local_cohesion_by_similarity(doc, paragraphs=None, distance=3):
    """ describe local cohesion of a doc as a list of similarity scores
        computed by spaCy between contiguous or neighbouring sentences;
        the underlying model is highly related to collocations """
    assert distance >= 1
    paras = doc.spans['PARA']
    local_cohesion = []
    n_paras = len(paras)
    mean_cohesion = 0
    if n_paras >= 2:
        cohesion_sum = 0
        for i in range(1, n_paras):
            para_similarity = 0
            window = range(min(i, distance))
            weight = 0
            for j in window:
                dist = j+1
                para_similarity += paras[i].similarity(paras[i-dist]) / dist
                weight += 1/dist
            # para_similarity /= len(window)
            para_similarity /= weight
            local_cohesion.append(para_similarity)
            cohesion_sum += para_similarity
            i += 1
        mean_cohesion = cohesion_sum / (n_paras-1)
    # return [0] + local_cohesion
    return [mean_cohesion] + local_cohesion

def local_cohesion_by_repetitions(doc, paragraphs=None, distance=3):
    """ describes local cohesion of a doc as a list of frequencies of lemmas
        repeated between contiguous or neighboring sentences;
        paragraph weight is inversely proportional to distance;
        a first normalization effect aims to bring values below 1;
        further heuristic normalization aims to avoid too small values; 
        this algorithm could be improved by taking into account synonyms
        and other semantic relationship between words """
    assert distance >= 1
    local_cohesion = []
    paras = doc.spans['PARA']
    n_paras = len(paras)
    mean_cohesion = 0
    repeated_lemmas = defaultdict(int)
    if n_paras >= 2:
        lemmas_list = []
        count_list = []
        for para in paras:
            lemmas = set()
            for token in para:
                if token.pos_ in ['NOUN', 'PROPN',]:
                    lemmas.add(token.lemma_)
            lemmas_list.append(lemmas)
            count_list.append(len(lemmas))
        cohesion_sum = 0
        for i in range(1, n_paras):
            n_repetitions = 0
            window = range(min(i, distance))
            for j in window:
                dist = j+1
                repeated = lemmas_list[i].intersection(lemmas_list[i-dist])
                n_repetitions += len(repeated)/dist
                if n_repetitions:
                    try:
                        weighted_repetitions = n_repetitions / min(count_list[i], count_list[i-dist])
                    except:
                        weighted_repetitions = 1
                    normalized_repetitions = math.pow(weighted_repetitions, 1/3)
                else:
                    weighted_repetitions = 0
                    normalized_repetitions = 0
                for lemma in repeated:
                    repeated_lemmas[lemma] += 1
            local_cohesion.append(normalized_repetitions)
            cohesion_sum += weighted_repetitions
        mean_cohesion = cohesion_sum / (n_paras-1)
        if mean_cohesion:
            mean_cohesion = math.pow(mean_cohesion, 1/3)
    return [mean_cohesion] + local_cohesion, repeated_lemmas

def local_cohesion_by_entity_graph(doc):
    """ the API of the algorithm are described in 
        https://trunajod20.readthedocs.io/en/latest/api_reference/entity_grid.html
        if the 'method' argument is present, characterize the local cohesion of a doc
        by a float score computed with the corresponding method,
        else return a list of the scores computed with all 6 available methods """
    grid = EntityGrid(doc)
    coherence_values = get_local_coherence(grid)
    coherence_keys = [
      'local_coherence_PU', 'local_coherence_PW', 'local_coherence_PACC',
      'local_coherence_PU_dist', 'local_coherence_PW_dist', 'local_coherence_PACC_dist',]
    return list(zip(coherence_keys, coherence_values))

base_file_key_format = '_{principal_key}_{serial}_{language}'
# user_file_key_format = 'u' + base_file_key_format
# project_file_key_format = 'p' + base_file_key_format

def language_from_file_key(file_key):
    return file_key[-2:]

def file_key_from_principal_key(principal_key='?????', serial='???', language='??', principal_type='u'):
    file_key_format = principal_type + base_file_key_format
    return file_key_format.format(principal_key=principal_key, serial=serial, language=language)

def file_key_from_path(path):
    return Path(path).stem

def path_from_file_key(file_key):
    return os.path.join(settings.CORPORA, file_key)+'.spacy'

def get_principal_docbins(user_key=None, project_key=None):
    if user_key:
        file_key_pattern = file_key_from_principal_key(principal_key=user_key, principal_type='u')
    elif project_key:
        file_key_pattern = file_key_from_principal_key(principal_key=project_key, principal_type='p')
    else:
        file_key_pattern = file_key_from_principal_key(principal_key='?????', principal_type='u')
    path_pattern = os.path.join(settings.CORPORA, file_key_pattern)+'.spacy'
    principal_docbins = []
    for path in glob.glob(path_pattern):
        file_key = file_key_from_path(path)
        time_stamp = time.ctime(os.path.getmtime(path))
        docbin = DocBin(store_user_data=True)
        docbin.from_disk(path)
        principal_docbins.append([file_key, docbin, time_stamp])
    return principal_docbins
        
def make_serial(user_key, language='??'):
    before_serial = os.path.join(settings.CORPORA, 'u_{user_key}_'.format(user_key=user_key))
    after_serial = '_??.spacy'
    path_pattern = before_serial+'*'+after_serial
    path_names = glob.glob(path_pattern)
    if path_names:
        serials = [path_name[len(before_serial):-len(after_serial)] for path_name in path_names]
        serials.sort()
        serial = '{:03d}'.format(int(serials[-1]) + 1)
    else:
        serial = '000'
    return serial

def make_docbin(user_key, language=None):
    docbin = DocBin(store_user_data=True)
    serial = make_serial(user_key, language='??')
    file_key = file_key_from_principal_key(principal_key=user_key, serial=serial, language='ll', principal_type='u')
    path = path_from_file_key(file_key)
    docbin.to_disk(path)
    return file_key, docbin

# def addto_docbin(docbin, doc, file_key):
def addto_docbin(docbin, doc, file_key, index=None):
    n_docs = len(docbin)
    print('index:', index)
    if index is not None:
        index = int(index)
    # assert index is None or (index >= 0 and index < n_docs)
    assert index is None or (index >= 0 and index <= n_docs)
    language = language_from_file_key(file_key)
    if n_docs > 0 and doc.lang_ != language:
        print('doc has unexpected language')
        return file_key, None
    path = path_from_file_key(file_key)
    if n_docs == 0:
        docbin.add(doc)
        print('doc added')
        language = doc.lang_
        file_key = file_key[:-2] + language
        new_path = os.path.join(settings.CORPORA, file_key)+'.spacy'
        os.rename(path, new_path)
        path = new_path
        print('file_key changed to', file_key)
    else:
        model = settings.LANGUAGE_MODELS[language]
        docs = list(docbin.get_docs(model.vocab))
        obj_id = doc._.obj_id
        obj_type = doc._.obj_type
        doc_existing = None
        for i, d in enumerate(docs):
            if d._.obj_type == obj_type and d._.obj_id == obj_id:
                doc_existing = i
                print('doc exists with index', i)
                break
        print('doc_existing:', doc_existing)
        # if doc_existing == index:
        if doc_existing is not None and doc_existing == index:
            print('doc unmoved', i)
            return file_key, docbin
        else:
            if doc_existing is None: 
                docbin.add(doc)
                print('doc added')
            else:
                docs.remove(docs[doc_existing])
                print('doc removed from position', doc_existing)
                if doc_existing < index:
                    docs.insert(index-1, doc)
                    print('and reinserted at position', index-1)
                else:
                    docs.insert(index, doc)
                    print('and reinserted at position', index)
                docbin = DocBin(docs=docs, store_user_data=True)
    docbin.to_disk(path)
    return file_key, docbin

def get_docbin(file_key=None, user_key=None, language='??'):
    if file_key:
        docbin = DocBin(store_user_data=True)
        path = path_from_file_key(file_key)
        docbin.from_disk(path)
    else:
        file_key, docbin = make_docbin(user_key, language=language)
    return file_key, docbin

def removefrom_docbin(file_key, obj_type, obj_id):
    file_key, docbin = get_docbin(file_key=file_key)
    language = language_from_file_key(file_key)
    model = settings.LANGUAGE_MODELS[language]
    index = i = 0
    docs = []
    for doc in list(docbin.get_docs(model.vocab)):
        if doc._.obj_type==obj_type and doc._.obj_id==obj_id:
            index = i
        else:
            docs.append(doc)
        i += 1
    delete_docbin(file_key)
    docbin = DocBin(docs=docs, store_user_data=True)
    path = path_from_file_key(file_key)
    docbin.to_disk(path)
    return index

def replace_in_docbin(file_key, new_doc):
    file_key, docbin = get_docbin(file_key=file_key)
    obj_type = new_doc._.obj_type
    obj_id = new_doc._.obj_id
    language = language_from_file_key(file_key)
    model = settings.LANGUAGE_MODELS[language]
    docs = list(docbin.get_docs(model.vocab))
    for i, doc in enumerate(docs):
        if doc._.obj_type==obj_type and doc._.obj_id==obj_id:
            docs = docs[:i] + [new_doc] + docs[i+1:]
    docbin = DocBin(docs=docs, store_user_data=True)
    path = path_from_file_key(file_key)
    docbin.to_disk(path)
    return docbin

def get_docs(docbin, language):
    model = settings.LANGUAGE_MODELS[language]
    return docbin.get_docs(model.vocab)   

def get_docbin_summary(docbin, language):   
    summary = []
    if language in settings.SUPPORTED_LANGUAGES:
        model = settings.LANGUAGE_MODELS[language]
        docs = docbin.get_docs(model.vocab)
        for doc in docs:
            doc_summary = {'obj_type': doc._.obj_type, 'obj_id': doc._.obj_id, 'label': doc._.label, 'url': doc._.url}
            doc_summary.update(get_doc_attributes(doc))
            summary.append(doc_summary)
    return summary 

def delete_docbin(file_key):
    """ delete a .spacy file (saved docbin)
    """
    path = path_from_file_key(file_key)
    if os.path.exists(path):
        os.remove(path)

def compare_docbin(docbin, language='en'):
    model = settings.LANGUAGE_MODELS[language]
    docs = list(docbin.get_docs(model.vocab))
    i = 0
    results_1 = []
    for doc_1 in docs:
        i += 1
        results_2 = []
        for doc_2 in docs[i:]:
            similarity = doc_1.similarity(doc_2)
            results_2.append(similarity)
        results_1.append(results_2)
    return results_1
    