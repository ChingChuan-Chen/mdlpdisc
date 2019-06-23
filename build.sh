#!/usr/bin/env bash
cd mdlpdisc
python setup.py build_ext --inplace

cd ../mdlp
python setup.py build_ext --inplace
