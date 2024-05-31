"""
Django settings for nlp app
"""

import os

"""
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
CORPORA = os.path.join(MEDIA_ROOT, 'corpora')
TEMP_ROOT = os.path.join(BASE_DIR, 'temp')
"""

import spacy
from spacy.language import Language

# this is used to display the language name
LANGUAGE_MAPPING = {
        'ar': 'Arabic',
        'el': 'Greek',
        'en': 'English',
        'de': 'German',
        'es': 'Spanish',
        'hr': 'Croatian',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'fr': 'French',
        'it': 'Italian',
        'nl': 'Dutch',
        'lt': 'Lithuanian',
        'da': 'Danish',
}

# load any spaCy models that are installed
# this takes some time to load so doing it here and hopefully this improves performance

# SUPPORTED_LANGUAGES = ['da', 'de', 'el', 'en', 'es', 'fr', 'hr', 'it', 'lt', 'nl', 'pl', 'pt',]
SUPPORTED_LANGUAGES = ['ar', 'da', 'el', 'en', 'es', 'hr', 'it', 'lt',]

AVAILABLE_LANGUAGE_MODELS = {}
AVAILABLE_LANGUAGE_MODELS['ar'] = ('ar_core_news_md',)
AVAILABLE_LANGUAGE_MODELS['da'] = ('da_core_news_md','da_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['de'] = ('de_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['el'] = ('el_core_news_md', 'el_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['en'] = ('en_core_web_md', 'en_core_web_sm',)
AVAILABLE_LANGUAGE_MODELS['es'] = ('es_core_news_md', 'es_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['fr'] = ('fr_core_news_md','fr_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['it'] = ('it_core_news_md', 'it_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['lt'] = ('lt_core_news_md','lt_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['hr'] = ('hr_core_news_md','hr_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['nl'] = ('nl_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['pl'] = ('pl_core_news_sm',)
AVAILABLE_LANGUAGE_MODELS['pt'] = ('pt_core_news_sm',)

import site
sitepackages_dir = None
for sp in site.getsitepackages():
    if sp.endswith('site-packages'):
        sitepackages_dir = sp

LANGUAGE_MODELS = {}
for language in SUPPORTED_LANGUAGES:
    for model in AVAILABLE_LANGUAGE_MODELS[language]:
        try:
            LANGUAGE_MODELS[language] = spacy.load(model) # (language)
            break
        except OSError:
            try:
                LANGUAGE_MODELS[language] = spacy.load(os.path.join(sitepackages_dir, model)) # (language)
                break
            except OSError:
                print('Warning: model {} not found.'.format(model))
                continue

ar = LANGUAGE_MODELS.get('ar', None)
if ar:
    from cameltokenizer import tokenizer
    cameltokenizer = tokenizer.CamelTokenizer(ar.vocab)

    @Language.component("cameltokenizer")
    def tokenizer_extra_step(doc):
        return cameltokenizer(doc)

    ar.add_pipe("cameltokenizer", name="cameltokenizer", first=True)

# this is used for language identification. Loading here to avoid importing many times
import langid as LANG_ID
LANG_ID.set_languages(LANGUAGE_MODELS.keys())
