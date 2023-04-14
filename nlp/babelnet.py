from django.conf import settings
# from py_babelnet.calls import BabelnetAPI
import babelnet as bn
from babelnet import Language
from babelnet.pos import POS
from babelnet.domain import BabelDomain
from babelnet.resources import BabelSynsetID

"""
def test():
    api = BabelnetAPI(settings.BABELNET_KEY, url=settings.BABELNET_URL)
    senses = api.get_senses(lemma = "agua", searchLang = "ES")
    print('babelnet test', senses)

def getSynsetIds(lemma, searchLang='EN', pos='NOUN', targetLang=[], domains=[]):
    api = BabelnetAPI(settings.BABELNET_KEY, url=settings.BABELNET_URL)
    kargs = {
        'lemma': lemma,
        'pos': pos,
        'searchLang': searchLang,
    }
    if targetLang:
        kargs['targetLang'] = targetLang
    if domains:
        kargs['domains'] = domains
    ids = api.get_synset_ids(**kargs)
    return ids

def getSynset(id, targetLang=[]):
    api = BabelnetAPI(settings.BABELNET_KEY, url=settings.BABELNET_URL)
    kargs = {
        'id': id,
    }
    if targetLang:
        kargs['targetLang'] = targetLang
    return api.get_synset(**kargs)

def test1():
    lemma = 'fellowship'
    searchLang = 'EN'
    targetLang = ['IT']
    pos = 'NOUN'
    domain = 'EDUCATION'
    return getSynsetIds(lemma, pos, searchLang, targetLang, domain=domain)

def test2():
    id = 'bn:02844132n'
    targetLang = ['IT']
    return getSynset(id, targetLang)

def test3():
    ids = ['bn:23580461n', 'bn:00034006n',]
    targetLang = ['IT']
    for id in ids:
        print(id, '->', getSynset(id, targetLang))
"""

from babelnet.api import OnlineAPI as api

def getSynsets():
    synsets = api._get_synsets()