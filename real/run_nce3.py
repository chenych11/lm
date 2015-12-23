#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

from utils import get_unigram_probtable
import optparse
from keras.optimizers import adam, AdamAnneal
from models import NCELangModelV3
import cPickle as pickle

DATA_PATH = '../data/corpus/wiki-sg-norm-lc-drop-bin.bz2'
NB_RUN_WORDS = 100000000
NB_VOCAB = 10000
NB_RUN_VAL = 100000
NB_EVALUATE = 5000000
BATCH_SIZE = 256

parser = optparse.OptionParser(usage="%prog [OPTIONS]")
parser.add_option("-a", "--lr", type="float", dest="lr", default=0.01,
                  help="learning rate")
parser.add_option("-R", "--running-words", type="int", dest="running_words", default=NB_RUN_WORDS,
                  help="amount of training data (number of words)")
parser.add_option("-S", "--coding-file", type="str", dest="coding_file",
                  help="sparse coding file (pickle)")
parser.add_option("-e", "--embedding-file", type="str", dest="embedding_file",
                  help="initial embedding file (pickle)")
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
parser.add_option("-N", "--nb-negative", type="int", dest="negative", default=50,
                  help="amount of training data (number of words)")
parser.add_option("-C", "--context-size", type="int", dest="context_size", default=128,
                  help="amount of training data (number of words)")
parser.add_option("-E", "--embedding-size", type="int", dest="embed_size", default=128,
                  help="amount of training data (number of words)")
parser.add_option("-l", "--log-file", type="str", dest="log_file", default='',
                  help="amount of training data (number of words)")
parser.add_option("-r", "--report-interval", type="float", dest="interval", default=1200.,
                  help="decaying rate")
parser.add_option("-s", "--save", type="str", dest="save", default='',
                  help="amount of training data (number of words)")
options, args = parser.parse_args()

nb_run_words = options.running_words
nb_run_val = options.val_run
nb_evaluate = options.nb_evaluation


with file(options.coding_file, 'rb') as f:
    sparse_coding = pickle.load(f)
    # print sparse_coding.dtype

nb_vocab = sparse_coding.shape[0]
unigram_table = get_unigram_probtable(nb_words=nb_vocab, save_path='../data/wiki-unigram-prob-size%d.pkl' % nb_vocab)

if options.embedding_file != '':
    with file(options.embedding_file, 'rb') as f:
        ini_embeds = pickle.load(f)
    # print ini_embeds.dtype
    # print ini_embeds.shape
    # import sys
    # sys.exit(0)
else:
    ini_embeds = None

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

model = NCELangModelV3(sparse_coding=sparse_coding, nb_negative=options.negative,
                       embed_dims=options.embed_size, context_dims=options.context_size,
                       init_embeddings=[ini_embeds], negprob_table=unigram_table, optimizer=opt)
model.compile()
model.train(data_file=DATA_PATH,
            save_path=save_path,
            batch_size=BATCH_SIZE, train_nb_words=nb_run_words,
            val_nb_words=nb_evaluate, train_val_nb=nb_run_val,
            validation_interval=options.interval, log_file=log_file)