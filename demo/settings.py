"""
Django settings for demo project.
"""

import os
# Note: we develop on Windows and publish on Linux
if os.name == 'nt':
    IS_LINUX = False
else:
    IS_LINUX = True

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'zty*m5%vhs&odlx&!!_y63p^un(4!_31h5h@*tqt&4!&$rt0c#'

# SECURITY WARNING: don't run with debug turned on in production!
# Note: we develop on Windows and publish on Linux
# Note: DEBUG value impacts fetching of static files!!!
if IS_LINUX:
    DEBUG = False
else:
    DEBUG = True
    TEMPLATE_STRING_IF_INVALID = '%s'

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    'nlp',
    'demo',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
]

ROOT_URLCONF = 'demo.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.template.context_processors.debug',
                'django.template.context_processors.static',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'demo.wsgi.application'

# Database
# https://docs.djangoproject.com/en/2.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.0/howto/static-files/

STATIC_URL = '/static/'
MEDIA_URL='/media/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

CORS_ORIGIN_ALLOW_ALL = True

ALLOWED_HOSTS = ['nlp.wordgames.gr', 'localhost', 'nlpbuddy.io', 'www.nlpbuddy.io']

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'debug.log'),
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}

import spacy

# load any spaCy models that are installed
# this takes some time to load so doing it here and hopefully this improves performance

SUPPORTED_LANGUAGES = ['de', 'el', 'en', 'es', 'fr', 'it', 'nl', 'pt']

LANGUAGE_MODELS = {}

for language in SUPPORTED_LANGUAGES:
    try:
        LANGUAGE_MODELS[language] = spacy.load(language)
    except OSError:
        print('Warning: model {} not found. Run python3 -m spacy download {} and try again.'.format(language,language))

LANGUAGE_MODELS['de'] = spacy.load('de_core_news_sm')
LANGUAGE_MODELS['el'] = spacy.load('el_core_news_sm')
LANGUAGE_MODELS['en'] = spacy.load('en_core_web_sm')
LANGUAGE_MODELS['es'] = spacy.load('es_core_news_sm')
LANGUAGE_MODELS['fr'] = spacy.load('fr_core_news_sm')
LANGUAGE_MODELS['it'] = spacy.load('it_core_news_sm')
LANGUAGE_MODELS['nl'] = spacy.load('nl_core_news_sm')
LANGUAGE_MODELS['pt'] = spacy.load('pt_core_news_sm')

# this is used to display the language name
LANGUAGE_MAPPING = {
        'el': 'Greek',
        'en': 'English',
        'de': 'German',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'fr': 'French',
        'it': 'Italian',
        'nl': 'Dutch',
}

# this is used for language identification. Loading here to avoid importing many times
import langid as LANG_ID
LANG_ID.set_languages(LANGUAGE_MODELS.keys())
# DEBUG = False


# whether to allow to import text from URLs
# library python-readability fetches text from a URL 
# and BeautifulSoup parses/removes tags
ALLOW_URL_IMPORTS = True
