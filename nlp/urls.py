# from django.conf.urls.static import static
# from django.conf import settings
# from django.conf.urls import url, include
# from django.contrib.sitemaps.views import sitemap
# from django.urls import path
from django.conf.urls import url
from . import views


urlpatterns = [
    url(r'^$', views.index),
    url(r'^about$', views.about, name='nlp.views.about'),
    url(r'^gsoc$', views.gsoc, name='nlp.views.gsoc'),
    url(r'^api/analyze$', views.analyze, name='nlp.views.analyze'),
    url(r'^api/doc$', views.doc, name='nlp.views.doc'),
    url(r'^api/compare$', views.compare, name='nlp.views.compare'),
    url(r'^api/new_corpus/', views.new_corpus, name='nlp.views.new_corpus'),
    # url(r'^api/make_corpus/', views.make_corpus, name='nlp.views.make_corpus'),
    url(r'^api/add_doc/', views.add_doc, name='nlp.views.add_doc'),
    url(r'^api/remove_doc/', views.remove_doc, name='nlp.views.remove_doc'),
    url(r'^api/get_corpora/', views.get_corpora, name='nlp.views.get_corpora'),
    url(r'^api/delete_corpus/', views.delete_corpus, name='nlp.views.delete_corpus'),
    url(r'^api/compare_docs/', views.compare_docs, name='nlp.views.compare_docs'),
    url(r'^api/word_contexts/', views.word_contexts, name='nlp.views.word_contexts'),
    url(r'^visualize$', views.visualize_view, name='nlp.views.visualize_view')
]
