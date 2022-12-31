import json
import spacy
import importlib

from spacy.tokens import Doc
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .utils import analyze_doc, visualize_text # , analyze_text
from .utils import text_to_list, text_to_doc, get_doc_attributes #, compare_docs
from .utils import get_principal_docbins, get_docbin, make_docbin, delete_docbin
from .utils import addto_docbin, removefrom_docbin, get_docbin_summary
from .utils import get_docs, doc_to_json
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

@csrf_exempt
def configuration(request):
    ret = {}
    ret['app_version'] = settings.APP_VERSION
    ret['spacy_version'] = spacy.__version__
    models = []
    for code, model in settings.LANGUAGE_MODELS.items():
        models.append("{}_{}-{}".format(code, model._meta['name'], model._meta['version']))
    ret['spacy_models'] = models
    return JsonResponse(ret)

@csrf_exempt
def analyze(request):
    """ The HTTP API view analyze() returns as json objects one or more text documents
        processed by a spaCy pipeline, whose Doc structure is enriched with custom post-processing.
        Input parameters in the request body: file_key, obj_type, obj_id, concatenate;
        they are all optional but the first three cannot be all missing;
        obj_type and obj_id make sense only in pairs;
        concatenate (False/True) makes sense if file_key identify a ultiple-Soc spaCy Docbin.
        Output: a json object in the body of the POST response.
    """
    if request.method == 'POST':
        # doc = list(docs(request, return_json=False))[0]
        doc = make_docs(request, return_json=False)
        language = doc.lang_
        ret = analyze_doc(text='', doc=doc)
        analyzed_text = ret.get('analyzed_text', '')
        if analyzed_text:
            ret['text'] = analyzed_text
        return JsonResponse(ret)
    else:
        ret = {'methods_allowed': 'POST'}
        return JsonResponse(ret)

@csrf_exempt
def make_docs(request, return_json=True):
    """ The HTTP API view make_docs() returns as json objects one or more text documents
        processed only by a spaCy pipeline. 
        Input parameters in the request body: text, file_key, obj_type, obj_id;
        they cannot be all missing;  obj_type and obj_id make sense only in pairs;
        the value of the text parameter can be the URL of an online text document.
        If the argument return_json is False, the function is called internally by the analyze view().
        Output: a json object in the body of the POST response or,
        if the view is called internally, a spaCy Doc object.
        Processing:
        if the file_key parameter is present, this is intended to identify a spaCy Docbin object persisted
        in the server's filesystem; obj_type and obj_id, if present, restrict the selection to a single Doc.
        if, instead, the text parameter is present, it is converted to a spaCy Doc, before being returned
        in one of the two Output modes mentioned above.
    """
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        text = data.get('text', '')
        file_key = data.get('file_key', '')
        obj_type = data.get('obj_type', '')
        obj_id = data.get('obj_id', '')
        if file_key:
            language = language_from_file_key(file_key)
            file_key, docbin = get_docbin(file_key=file_key)
            docs = list(get_docs(docbin, language))
            if obj_type and obj_id:
                n_docs = 0
                for doc in docs:
                    if doc._.obj_type == obj_type and str(doc._.obj_id) == str(obj_id):
                        docs = [doc]
                        n_docs = 1
                        break
            else:
                n_docs = len(docbin)
            if return_json:
                data = [doc_to_json(doc) for doc in docs]
                return JsonResponse(data) # return list with 1 doc as json on API
            else:
                assert n_docs
                if n_docs > 1:
                    doc = Doc.from_docs(docs)
                else:
                    doc = docs[0]
                return doc # return a doc internally
        else:
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
            if return_json:
                data = [text_to_doc(text, return_json=True)]
                return JsonResponse(data) # return list with 1 doc as json on API
            else:
                # return [text_to_doc(text)] # return list with 1 doc internally
                return text_to_doc(text)
    else:
        ret = {'methods_allowed': 'POST'}
        return JsonResponse(ret)

@csrf_exempt
def get_corpus(request):
    """ The HTTP API view get_corpus().
        Input: file_key, a unique filename built around a user id
        Output: an extended serialized version of a spaCy doc 
    """
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        file_key = data.get('file_key', '')
        assert file_key
        language = language_from_file_key(file_key)
        file_key, docbin = get_docbin(file_key=file_key)
        doc_list = []
        for doc in get_docs(docbin, language):
            doct_dict = doc_to_json(doc, language)
            # see utils.get_docbin_summary
            doc_summary = {'obj_type': doc._.obj_type, 'obj_id': doc._.obj_id, 'label': doc._.label, 'url': doc._.url}
            doct_dict.update(doc_summary)
            doc_list.append(doct_dict)
        return JsonResponse(doc_list, safe=False)
    else:
        ret = {'methods_allowed': 'POST'}
        return JsonResponse(ret)

@csrf_exempt
def text_cohesion(request):
    """ The HTTP API view text_cohesion() .
        Output: .
    """
    if request.method == 'POST':
        doc = make_docs(request, return_json=False)
        language = doc.lang_
        ret = analyze_doc(text='', doc=doc, keys=['sentences', 'text_cohesion',])
        analyzed_text = ret.get('analyzed_text', '')
        if analyzed_text:
            ret['text'] = analyzed_text
        return JsonResponse(ret)
    else:
        ret = {'methods_allowed': 'POST'}
        return JsonResponse(ret)

@csrf_exempt
def word_contexts(request):
    MIN_KEYWORDS = 10
    MIN_CONTEXTS = 3
    CONTEXT_SIZE = 5
    from tmtoolkit.corpus import Corpus as tm_Corpus, kwic
    doc = make_docs(request, return_json=False)
    language = doc.lang_
    docs = [doc]
    
    keywords, lemma_forms = get_sorted_keywords(language=language, docs=docs)
    keywords = [kw for kw in keywords if len(kw[0])>1]
    keywords_in_context = []
    j = 0
    for lemma, frequency in keywords:
        contexts_dict = {}
        doc_dict = {}
        for doc in docs:
            doc_dict[doc._.label] = doc.text
        corpus = tm_Corpus(doc_dict, spacy_instance=settings.LANGUAGE_MODELS[language])
        contexts_dict = \
            kwic(corpus, list(lemma_forms[lemma]), context_size=CONTEXT_SIZE, match_type='exact', ignore_case=False,
            glob_method='match', inverse=False,  as_tables=False, only_non_empty=False, glue=None)
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
    return JsonResponse({'language': language, 'keywords': keywords, 'kwics': keywords_in_context})

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
    file_key = data.get('file_key', None)
    index = data.get('index', None)
    text = data['text']
    doc = text_to_doc(text)
    doc._.label = data['label']
    doc._.obj_type = data['obj_type']
    doc._.obj_id = data['obj_id']
    doc._.url = data['url']
    result = get_doc_attributes(doc)
    file_key, docbin = get_docbin(file_key=file_key, language=doc.lang_)
    file_key, docbin = addto_docbin(docbin, doc, file_key, index=index)
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
def compare(request):
    """ Caveat: this view hasn't been re-tested after major revision of this module.
        The HTTP API view compare () returns as json objects the similarity score computed
        by spaaCY among two text document.
        Input parameter in the Request body: text;
        the first two lines of the text must by two URLs identifying online text documents.
    """
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
