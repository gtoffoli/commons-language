# see: https://stackoverflow.com/questions/70319606/importerror-cannot-import-name-url-from-django-conf-urls-after-upgrading-to#:~:text=The%20easiest%20fix%20is%20to,and%20replace%20url%20with%20re_path%20.&text=Alternatively%2C%20you%20could%20switch%20to,if%20you%20switch%20to%20path.
# from django.conf.urls import url
from django.conf.urls import re_path as url
from . import views


urlpatterns = [
    url(r'^$', views.index),
    url(r'^about$', views.about, name='nlp.views.about'),
    url(r'^gsoc$', views.gsoc, name='nlp.views.gsoc'),
    url(r'^api/configuration', views.configuration, name='nlp.views.configuration'),
    url(r'^api/analyze$', views.analyze, name='nlp.views.analyze'),
    # url(r'^api/doc$', views.doc, name='nlp.views.doc'),
    url(r'^api/get_docs$', views.get_docs, name='nlp.views.get_docs'),
    url(r'^api/compare$', views.compare, name='nlp.views.compare'),
    url(r'^api/new_corpus/', views.new_corpus, name='nlp.views.new_corpus'),
    url(r'^api/add_doc/', views.add_doc, name='nlp.views.add_doc'),
    url(r'^api/remove_doc/', views.remove_doc, name='nlp.views.remove_doc'),
    url(r'^api/get_corpora/', views.get_corpora, name='nlp.views.get_corpora'),
    url(r'^api/delete_corpus/', views.delete_corpus, name='nlp.views.delete_corpus'),
    url(r'^api/compare_docs/', views.compare_docs, name='nlp.views.compare_docs'),
    url(r'^api/word_contexts/', views.word_contexts, name='nlp.views.word_contexts'),
    url(r'^visualize$', views.visualize_view, name='nlp.views.visualize_view')
]
