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
    url(r'^api/delete_docs/', views.delete_docs, name='nlp.views.delete_docs'),
    url(r'^api/add_doc/', views.add_doc, name='nlp.views.add_doc'),
    url(r'^api/compare_docs/', views.compare_docs, name='nlp.views.compare_docs'),
    url(r'^visualize$', views.visualize_view, name='nlp.views.visualize_view')
]
