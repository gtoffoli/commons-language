import json
import spacy
from spacy.tokens import Doc
from tmtoolkit.preprocess._docfuncs import _init_doc, kwic
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .utils import analyze_text, visualize_text
from .utils import text_to_list, text_to_doc, get_doc_attributes, compare_docs
from .utils import get_principal_docbins, get_docbin, make_docbin, delete_docbin
from .utils import addto_docbin, removefrom_docbin, get_docbin_summary
from .utils import language_from_file_key, get_sorted_keywords, compare_docbin

if settings.ALLOW_URL_IMPORTS:
    import requests
    from bs4 import BeautifulSoup
    from readability import Document

if not Doc.has_extension("obj_type"):
    Doc.set_extension("obj_type", default='')
if not Doc.has_extension("obj_id"):
    Doc.set_extension("obj_id", default=0)
if not Doc.has_extension("label"):
    Doc.set_extension("label", default='')
if not Doc.has_extension("url"):
    Doc.set_extension("url", default='/')

@csrf_exempt
def index(request):
    'Index view'
    text = url = noframe = ''
    if request.method == 'GET':
        text = request.GET.get('text', '')
        url = request.GET.get('url', '')
        noframe = request.GET.get('noframe', '')
    elif request.method == 'POST':
        try:
            params = json.loads(request.body)
            text = params.get('text', '')
            url = params.get('url', '')
            noframe = params.GET.get('noframe', '')
        except:
            url = request.POST.get('url', '')
            text = request.POST.get('text', '')
            noframe = request.POST.get('noframe', '')
    context = {}
    if text:
        context['TEXT'] = text
    elif url:
        context['URL'] = url
    context['FRAME'] = not noframe or noframe in ['0', 'False', 'false']
    return render(request, 'nlp/index.html', context)


def about(request):
    'About view'
    context = {}
    return render(request, 'nlp/about.html', context)


def gsoc(request):
    'About gsoc'
    context = {}
    return render(request, 'nlp/gsoc.html', context)


def visualize_view(request):
    ret = {}
    text = request.POST.get('sentences')
    if (text is None):
        return render(request, 'nlp/visualize_error.html')
    markup = visualize_text(text)
    ret['json'] = markup
    return render(request, 'nlp/visualize.html', ret)

def configuration(request):
    ret = {}
    ret['app_version'] = settings.APP_VERSION
    ret['spacy_version'] = spacy.__version__
    ret['spacy_models'] = settings.LANGUAGE_MODELS
    return JsonResponse(ret)

@csrf_exempt
def analyze(request, return_doc=False):
    'API text analyze view'
    if request.method == 'POST':
        text = request.body.decode('utf-8')
        try:
            text = json.loads(text)['text']
        except ValueError:
            # catch POST form as well
            for key in request.POST.dict().keys():
                text = key

        if settings.ALLOW_URL_IMPORTS and text.startswith(('http://', 'https://', 'www')):
            page = requests.get(text)
            doc = Document(page.text)
            soup = BeautifulSoup(doc.summary())
            text = soup.get_text()
            title = doc.title().strip()
            text = '{0}.\n{1}'.format(title, text)

        if not text:
            response = JsonResponse(
                {'status': 'false', 'message': 'need some text here!'})
            response.status_code = 400
            return response

        # add some limit here
        text = text[:200000]
        if return_doc:
            # language, doc_json = text_to_doc(text, return_json=True)
            doc_json = text_to_doc(text, return_json=True)
            language = doc_json['language']
            if not language:
                response = JsonResponse(
                    {'status': 'false', 'message': 'unrecognized language'})
                response.status_code = 400
                return response
            ret = doc_json
        else:
            ret = analyze_text(text)
            ret['doc'] = None
        return JsonResponse(ret)
    else:
        ret = {'methods_allowed': 'POST'}
        return JsonResponse(ret)

@csrf_exempt
def doc(request):
    return analyze(request, return_doc=True)

@csrf_exempt
def compare(request):
    'API compare documents view'
    doc_dicts = []
    if request.method == 'POST':
        text = request.body.decode('utf-8')
        try:
            text = json.loads(text)['text']
        except ValueError:
            # catch POST form as well
            for key in request.POST.dict().keys():
                text = key

        if settings.ALLOW_URL_IMPORTS and text.startswith(('http://', 'https://', 'www')):
            lines = text_to_list(text)
            i = 0
            for line in lines[:2]:
                if not line.startswith(('http://', 'https://', 'www')):
                    response = JsonResponse({'status': 'false', 'message': 'need at least 2 urls!'})
                    response.status_code = 400
                    return response
                page = requests.get(line)
                doc = Document(page.text)
                soup = BeautifulSoup(doc.summary())
                text = soup.get_text()
                title = doc.title().strip()
                text = '{0}.\n{1}'.format(title, text)
                if not text:
                    response = JsonResponse({'status': 'false', 'message': 'need some text here!'})
                    response.status_code = 400
                    return response
    
                # add some limit here
                text = text[:200000]
                doc = text_to_doc(text)
                language = doc.lang_
                if i>0 and language!=doc_dicts[0]['language']:
                    response = JsonResponse(
                        {'status': 'false', 'message': 'texts must be in same language!'})
                    response.status_code = 400
                    return response
                    
                doc_dicts.append({'language': language, 'doc': doc})
                i += 1
            ret = compare_docs(doc_dicts)
            ret['language'] = language
            ret['text'] = text
            return JsonResponse(ret)
        else:
            response = JsonResponse({'status': 'false', 'message': 'need 2 documents!'})
            response.status_code = 400
            return response

    else:
        return JsonResponse({'methods_allowed': 'POST'})


@csrf_exempt
def new_corpus(request):
    if not request.method == 'POST':
        return JsonResponse({'status': 'false', 'message': 'invalid method!'})
    data = json.loads(request.body.decode('utf-8'))
    user_key = data['user_key']
    file_key, docbin = make_docbin(user_key)
    result = {'file_key': file_key}
    return JsonResponse(result)

@csrf_exempt
def add_doc(request):
    if not request.method == 'POST':
        return JsonResponse({'status': 'false', 'message': 'invalid method!'})
    data = json.loads(request.body.decode('utf-8'))
    file_key = data['file_key'] or None
    text = data['text']
    doc = text_to_doc(text)
    doc._.label = data['label']
    doc._.obj_type = data['obj_type']
    doc._.obj_id = data['obj_id']
    doc._.url = data['url']
    result = get_doc_attributes(doc)
    file_key, docbin = get_docbin(file_key=file_key, language=doc.lang_)
    file_key, docbin = addto_docbin(docbin, doc, file_key)
    if docbin:
        result.update({'file_key': file_key})
    else:
        result = {'file_key': ''}
    return JsonResponse(result)

@csrf_exempt
def remove_doc(request):
    if not request.method == 'POST':
        return JsonResponse({'status': 'false', 'message': 'invalid method!'})
    data = json.loads(request.body.decode('utf-8'))
    file_key = data['file_key']
    obj_type = data['obj_type']
    obj_id = data['obj_id']
    index = removefrom_docbin(file_key, obj_type, obj_id)
    result = {'index': index}
    return JsonResponse(result)

@csrf_exempt
def get_corpora(request):
    if not request.method == 'POST':
        return JsonResponse({'status': 'false', 'message': 'invalid method!'})
    data = json.loads(request.body.decode('utf-8'))
    user_key = data['user_key']
    project_key = data.get('project_key', None)
    corpora = []
    if user_key:
        for file_key, docbin, time_stamp in get_principal_docbins(user_key=user_key, project_key=project_key):
            language = language_from_file_key(file_key)
            corpora.append({'list_id': 'corpus', 'file_key': file_key, 'language': language, 'time_stamp': time_stamp, 'items': get_docbin_summary(docbin, language)})
    return JsonResponse({'corpora': corpora})

@csrf_exempt
def delete_corpus(request):
    if not request.method == 'POST':
        return JsonResponse({'status': 'false', 'message': 'invalid method!'})
    data = json.loads(request.body.decode('utf-8'))
    file_key = data['file_key']
    delete_docbin(file_key)
    return JsonResponse(data)

@csrf_exempt
def compare_docs(request):
    if not request.method == 'POST':
        return JsonResponse({'status': 'false', 'message': 'invalid method!'})
    data = json.loads(request.body.decode('utf-8'))
    file_key = data['file_key']
    file_key, docbin = get_docbin(file_key=file_key)
    language = language_from_file_key(file_key)
    n = len(docbin)
    if n < 2:
        return JsonResponse({'status': 'false', 'message': 'need 2 documents at least!'})
    result = compare_docbin(docbin, language=language)
    return JsonResponse({'status': 'true', 'result': result})
 
@csrf_exempt
def word_contexts(request):
    data = json.loads(request.body.decode('utf-8'))
    MIN_KEYWORDS = 10
    MIN_CONTEXTS = 3
    CONTEXT_SIZE = 5
    text = data.get('text', None)
    if text is not None:
        ret = analyze_text(text)
        language = ret['language']
        docs = [ret['doc']]
    else:
        file_key = data['file_key']
        obj_type = data.get('obj_type', '')
        obj_id = data.get('obj_id', '')
        file_key, docbin = get_docbin(file_key=file_key)
        language = language_from_file_key(file_key)
        model = settings.LANGUAGE_MODELS[language]
        docs = []
        i = 0
        for doc in list(docbin.get_docs(model.vocab)):
            if not obj_type or (doc._.obj_type==obj_type and str(doc._.obj_id)==obj_id):
                _init_doc(doc)
                i += 1
                doc._.label = 'doc_{}'.format(i)
                docs.append(doc)
    keywords, lemma_forms = get_sorted_keywords(language=language, docs=docs)
    keywords = [kw for kw in keywords if len(kw[0])>1]
    keywords_in_context = []
    j = 0
    for lemma, frequency in keywords:
        contexts_dict = \
            kwic(docs, list(lemma_forms[lemma]), context_size=CONTEXT_SIZE, match_type='exact', ignore_case=False,
            glob_method='match', inverse=False, with_metadata=False, as_dict=True, as_datatable=False, non_empty=False,
            glue=None, highlight_keyword=None)
        contexts = []
        for context_item in contexts_dict.items():
            for window in context_item[1]:
                left = ' '.join(window[:CONTEXT_SIZE])
                middle = window[CONTEXT_SIZE]
                right = ' '.join(window[-CONTEXT_SIZE:])
                contexts.append([left, middle, right])
        j += 1
        if j >= MIN_KEYWORDS and frequency < MIN_CONTEXTS:
            break
        if contexts:
            keyword_in_context = {'kw': lemma, 'frequency': frequency, 'contexts': contexts}
            keywords_in_context.append(keyword_in_context)
    return JsonResponse({'keywords': keywords, 'kwics': keywords_in_context})
