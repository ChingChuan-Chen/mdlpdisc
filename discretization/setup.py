#!/usr/bin/env python3

from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(name="discretization", 
                                        sources=["discretization.pyx"],
                                        language='c++')))
