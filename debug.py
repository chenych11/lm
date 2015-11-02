#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

from models import TreeLangModel
import cPickle as pickle
import theano
import numpy as np

floatX = theano.config.floatX
train_file = 'data/fake/001.bz2'

with file('data/fake/tree-info.pkl', 'rb') as f:
        tree_info = pickle.load(f)
        word2cls = tree_info['idx2cls']
        word2bitstr = tree_info['idx2bitstr']

model = TreeLangModel(vocab_size=15, embed_dim=128, cntx_dim=128, word2class=word2cls, word2bitstr=word2bitstr)


X = np.loadtxt(train_file, dtype='int32')
data = X[:256]
del X



