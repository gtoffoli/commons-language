import os
from setuptools import setup
from Cython.Build import cythonize
import numpy

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define_macros=[['NPY_NO_DEPRECATED_API',None], ['NPY_1_7_API_VERSION',None]]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pyx_path = os.path.join(BASE_DIR, "nlp", "spacy_custom", "ar", "msatokenizer.pyx")
print(pyx_path)

setup(
    name='MsaTokenizer class',
    ext_modules=cythonize(pyx_path),
    include_dirs=[numpy.get_include()]
)