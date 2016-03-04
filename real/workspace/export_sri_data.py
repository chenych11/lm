#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'
# noinspection PyUnresolvedReferences
from lm.real.utils import data4sri

DATA_ROOT = '../../data/'
# data4sri(src_corpus=DATA_ROOT+'/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=DATA_ROOT+'corpus/sri',
#          train_nb_words=100000000, val_nb_words=5000000, train_val_nb=100000, max_vocab=10000)

# data4sri(src_corpus=DATA_ROOT+'/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=DATA_ROOT+'corpus/sri',
#          train_nb_words=100000000, val_nb_words=5000000, train_val_nb=100000, max_vocab=50000)

for k in range(10000, 52000, 2000):
    data4sri(src_corpus=DATA_ROOT+'/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=DATA_ROOT+'corpus/sri',
             train_nb_words=100000000, val_nb_words=5000000, train_val_nb=100000, max_vocab=k)

data4sri(src_corpus=DATA_ROOT+'/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=DATA_ROOT+'corpus/sri',
         train_nb_words=100000000, val_nb_words=5000000, train_val_nb=100000, max_vocab=100000000)