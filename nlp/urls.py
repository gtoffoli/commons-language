# see: https://stackoverflow.com/questions/70319606/importerror-cannot-import-name-url-from-django-conf-urls-after-upgrading-to#:~:text=The%20easiest%20fix%20is%20to,and%20replace%20url%20with%20re_path%20.&text=Alternatively%2C%20you%20could%20switch%20to,if%20you%20switch%20to%20path.
# from django.conf.urls import url
from django.urls import re_path as url
from . import views


urlpatterns = [
    url(r'^$', views.index),
    url(r'^api/configuration', views.configuration, name='nlp.views.configuration'),
    url(r'^api/analyze$', views.analyze, name='nlp.views.analyze'),
    url(r'^api/get_corpus$', views.get_corpus, name='nlp.views.get_corpus'),
    url(r'^api/compare$', views.compare, name='nlp.views.compare'),
    url(r'^api/text_cohesion', views.text_cohesion, name='nlp.views.text_cohesion'),
    url(r'^api/word_contexts/', views.word_contexts, name='nlp.views.word_contexts'),
    url(r'^api/compare_docs/', views.compare_docs, name='nlp.views.compare_docs'),

    url(r'^api/new_corpus/', views.new_corpus, name='nlp.views.new_corpus'),
    url(r'^api/add_doc/', views.add_doc, name='nlp.views.add_doc'),
    url(r'^api/remove_doc/', views.remove_doc, name='nlp.views.remove_doc'),
    # url(r'^api/update_domains/', views.update_domains, name='nlp.views.update_domains'),
    # url(r'^api/get_domains/', views.get_domains, name='nlp.views.get_domains'),
    url(r'^api/delete_corpus/', views.delete_corpus, name='nlp.views.delete_corpus'),
    url(r'^api/get_corpora/', views.get_corpora, name='nlp.views.get_corpora'),

    url(r'^about$', views.about, name='nlp.views.about'),
    url(r'^gsoc$', views.gsoc, name='nlp.views.gsoc'),
    url(r'^visualize$', views.visualize_view, name='nlp.views.visualize_view')
]
