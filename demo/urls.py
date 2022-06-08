"""demo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

# see: https://stackoverflow.com/questions/70319606/importerror-cannot-import-name-url-from-django-conf-urls-after-upgrading-to#:~:text=The%20easiest%20fix%20is%20to,and%20replace%20url%20with%20re_path%20.&text=Alternatively%2C%20you%20could%20switch%20to,if%20you%20switch%20to%20path.
# from django.conf.urls import url, include
from django.urls import include, re_path as url
from django.contrib import admin
from django.conf import settings

urlpatterns = [
    url(r'^', include('nlp.urls')),
#    url(r'^admin/', admin.site.urls),
    ]

