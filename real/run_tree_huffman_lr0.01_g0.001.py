#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

from models import TreeLangModel
from keras.optimizers import AdamAnneal
import cPickle as pickle

NB_RUN_WORDS = 100000000
NB_VOCAB = 10000
NB_RUN_VAL = 100000
NB_EVALUATE = 5000000

# NB_RUN_WORDS = 1000000
# NB_VOCAB = 10000
# NB_RUN_VAL = 10000
# NB_EVALUATE = 50000
SAVE_PATH = '../data/models/lang/huffman-e128-c128-lr0.01-gamma0.001.pkl'

DATA_PATH = '../data/corpus/wiki-sg-norm-lc-drop-bin.bz2'
BATCH_SIZE = 256
VAL_INTER = 1200

with file('../data/wiki-huffman-tree-info-Vsize10000.pkl', 'rb') as f:
    tree_info = pickle.load(f)

wrd2cls = tree_info['idx2cls']
wrd2bitstr = tree_info['idx2bitstr']

opt = AdamAnneal(lr=0.01, lr_min=0.0045, gamma=0.001)
model = TreeLangModel(vocab_size=NB_VOCAB, embed_dim=128, cntx_dim=128,
                      word2class=wrd2cls, word2bitstr=wrd2bitstr, optimizer=opt)
model.compile()
model.train(data_file=DATA_PATH,
            save_path=SAVE_PATH,
            batch_size=BATCH_SIZE, train_nb_words=NB_RUN_WORDS,
            val_nb_words=NB_EVALUATE, train_val_nb=NB_RUN_VAL, validation_interval=VAL_INTER)