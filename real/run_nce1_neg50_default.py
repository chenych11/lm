#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

from utils import get_unigram_probtable
from models import NCELangModelV1

NB_RUN_WORDS = 100000000
NB_VOCAB = 10000
NB_RUN_VAL = 100000
NB_EVALUATE = 5000000
SAVE_PATH = '../data/models/lang/nce1-neg50-e128-c128.pkl'

DATA_PATH = '../data/corpus/wiki-sg-norm-lc-drop-bin.bz2'
BATCH_SIZE = 256
VAL_INTER = 1200

# NB_RUN_WORDS = 5000000
# NB_VOCAB = 10000
# NB_RUN_VAL = 100000
# NB_EVALUATE = 500000

unigram_table = get_unigram_probtable(nb_words=NB_VOCAB)

model = NCELangModelV1(vocab_size=NB_VOCAB, nb_negative=50, embed_dims=128, context_dims=128,
                       negprob_table=unigram_table, optimizer='adam')
model.compile()
# model.train(data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2',
#             save_path='../data/models/lang/nce-neg50-e128-c128.pkl',
#             batch_size=256, train_nb_words=NB_RUN_WORDS//100,
#             val_nb_words=NB_EVALUATE//10, train_val_nb=NB_RUN_VAL//5, validation_interval=40)
model.train(data_file=DATA_PATH, save_path=SAVE_PATH,
            batch_size=BATCH_SIZE, train_nb_words=NB_RUN_WORDS,
            val_nb_words=NB_EVALUATE, train_val_nb=NB_RUN_VAL, validation_interval=VAL_INTER)