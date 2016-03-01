#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'
import optparse
from keras.optimizers import adam, AdamAnneal
from models import LBLangModelV1
# noinspection PyUnresolvedReferences
from SparseEmbed.cu_gen_sparse import compose_dense_repr

DATA_PATH = '../data/corpus/wiki-sg-norm-lc-drop-bin.bz2'
EMBED_FILE = '../data/models/embeddings/rw2vec_embeddings-size200.pkl'
NB_RUN_WORDS = 100000000
NB_VOCAB = 10000
NB_RUN_VAL = 100000
NB_EVALUATE = 5000000
BATCH_SIZE = 512

parser = optparse.OptionParser(usage="%prog [OPTIONS]")
parser.add_option("-a", "--lr", type="float", dest="lr", default=0.01,
                  help="learning rate")
parser.add_option("-R", "--running-words", type="int", dest="running_words", default=NB_RUN_WORDS,
                  help="amount of training data (number of words)")
parser.add_option("-m", "--val-run", type="int", dest="val_run", default=NB_RUN_VAL,
                  help="running validation words")
parser.add_option("-n", "--nb-evaluation", type="int", dest="nb_evaluation", default=NB_EVALUATE,
                  help="running validation words")
parser.add_option("-g", "--gamma", type="float", dest="gamma", default=0.001,
                  help="decaying rate")
parser.add_option("-b", "--lr-min", type="float", dest="lr_min", default=0.005,
                  help="decaying rate")
parser.add_option("-d", "--decay", action="store_true", dest="decay", default=False,
                  help="decay lr or not")
parser.add_option("-C", "--context-size", type="int", dest="context_size", default=5,
                  help="amount of training data (number of words)")
parser.add_option("-E", "--embedding-size", type="int", dest="embed_size", default=200,
                  help="amount of training data (number of words)")
parser.add_option("-l", "--log-file", type="str", dest="log_file", default='',
                  help="amount of training data (number of words)")
parser.add_option("-r", "--report-interval", type="float", dest="interval", default=900.,
                  help="decaying rate")
parser.add_option("-s", "--save", type="str", dest="save", default='',
                  help="amount of training data (number of words)")
parser.add_option("-V", "--nb-vocab", type="int", dest="nb_vocab", default=30000,
                  help="Number of vocabulary")
parser.add_option("-D", "--corpus", type="str", dest="corpus", default=DATA_PATH,
                  help="binarized corpus file")
options, args = parser.parse_args()

nb_run_words = options.running_words
nb_run_val = options.val_run
nb_evaluate = options.nb_evaluation

if options.decay:
    opt = AdamAnneal(lr=options.lr, lr_min=options.lr_min, gamma=options.gamma)
else:
    opt = adam(lr=options.lr)

if options.log_file == '':
    log_file = None
else:
    log_file = options.log_file

if options.save == '':
    save_path = None
else:
    save_path = options.save

model = LBLangModelV1(vocab_size=options.nb_vocab,
                      context_size=options.context_size,
                      embed_dims=options.embed_size)
model.compile(opt)
model.train(data_file=options.corpus,
            save_path=save_path,
            batch_size=BATCH_SIZE,
            train_nb_words=nb_run_words,
            val_nb_words=nb_evaluate,
            train_val_nb=nb_run_val,
            validation_interval=options.interval,
            log_file=log_file)