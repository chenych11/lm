#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'
# noinspection PyUnresolvedReferences
from lm.utils.preprocess import grouped_sentences, smart_open
import numpy as np

DATA_PATH = '../../data/corpus/wiki-sg-norm-lc-drop-bin.bz2'
DATA_DIST = '../../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2'


def _commit_result(dist_file, sents):
    shape = np.array(sents.shape, dtype=np.int32)
    dist_file.write(shape.tobytes())
    dist_file.write(sents.tobytes())

first_chunk_size = 50000000
next_chunk_start = first_chunk_size * 2
total_size = 100000000
nb_words = 0

dist_file = smart_open(DATA_DIST, 'wb')
sents = grouped_sentences(DATA_PATH)
for chunk in sents:
    if nb_words > first_chunk_size:
        break
    nb_words += chunk.size
    _commit_result(dist_file, chunk)

nb_words_ = nb_words
for chunk in sents:
    nb_words_ += chunk.size
    if nb_words_ > next_chunk_start:
        break

for chunk in sents:
    if nb_words >= total_size:
        break
    nb_words += chunk.size
    _commit_result(dist_file, chunk)


dist_file.close()


