import json
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .utils import analyze_text, visualize_text

if settings.ALLOW_URL_IMPORTS:
    import requests
    from bs4 import BeautifulSoup
    from readability import Document


def index(request):
    'Index view'
    text = url = ''
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
    noframe = request.GET.get('noframe', False)
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
def analyze(request):
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
        ret = {}
        ret = analyze_text(text)
        return JsonResponse(ret)
    else:
        ret = {'methods_allowed': 'POST'}
        return JsonResponse(ret)
