#!/usr/bin/env python3

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import platform

extra_compile_args = []
extra_link_args = []
if platform.system() == 'Windows':
    extra_compile_args.append("/openmp")
else:
	extra_compile_args.append("-fopenmp")
	extra_link_args.append("-fopenmp")

ext = Extension(name="mdlpdisc", 
                sources=["mdlpdisc.pyx", "helper.cpp"],
                language="c++", 
                include_dirs=[numpy.get_include()],
                extra_compile_args=extra_compile_args, 
				extra_link_args=extra_link_args)

setup(
	name='mdlpdisc',
	version='0.1',
	license='GPL-3 Clause',
	url='github.com/ChingChuan-Chen/mdlpdisc',
	author='Ching-Chuan Chen',
	author_email='zw12356@gmail.com',
    install_requires=["setuptools>=18.0", "numpy>=1.11.2", "Cython"], 
	ext_modules=cythonize(ext)
)
