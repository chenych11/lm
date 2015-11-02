#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

from models import SimpleLangModel, NCELangModel, NCELangModelV1, TreeLangModel
from keras.layers.core import Dropout, Dense
import os
import logging
import optparse
import cPickle as pickle
import theano

floatX = theano.config.floatX

parser = optparse.OptionParser(usage="%prog [OPTIONS]")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print progress bar to stdout")
parser.add_option("-s", "--simple",
                  action="store_true", dest="train_simple", default=False,
                  help="Train Simple language model")
parser.add_option("-n", "--nce",
                  action="store_true", dest="train_nce", default=False,
                  help="Train NCE based language model")
parser.add_option("-c", "--nce1",
                  action="store_true", dest="train_nce1", default=False,
                  help="Train NCE based language model V1")
parser.add_option("-b", "--batch-size", type='int', dest="batch_size", default=256,
                  help="Batch size")
parser.add_option("-t", "--test",
                  action="store_true", dest="test", default=False,
                  help="train on small data set")
parser.add_option("-r", "--tree",
                  action="store_true", dest="tree", default=False,
                  help="Train hierarchical softmax language model")
parser.add_option("-m", "--tree-type", type='str',
                  dest="tree_type", default='huffman',
                  help="Specify the type of the tree")
parser.add_option("-d", "--debug",
                  action="store_true", dest="debug", default=False,
                  help="show debug information")
parser.add_option("-g", "--unigram",
                  action="store_true", dest="unigram", default=False,
                  help="Whether use unigram distribution for noise samples")
parser.add_option("-z", "--optimizer", type='str',
                  dest="optimizer", default='sgd',
                  help="Specify optimizer")
options, args = parser.parse_args()
# ====================================================================================
# if TESTLM environment variable is defined, run the program on a small data set.
if os.environ.get('TESTLM') is not None or options.test:
    data_path = os.path.abspath('data/fake/test')
else:
    data_path = os.path.abspath('data/fake')
#data_path = os.path.abspath('data/fake/test')

if options.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if options.unigram:
    import cPickle as pickle
    with file(os.path.join(data_path, 'meta.pkl'), 'rb') as mf:
        meta = pickle.load(mf)
        negprob_table = meta['rel_freq']
else:
    negprob_table = None

if options.train_simple:
    logging.info('Train simple language model')
    model = SimpleLangModel(vocab_size=15, optimizer=options.optimizer)
    model.add(model.WordEmbedding(embed_dim=128))
    model.add(model.LangLSTM(out_dim=128))
    # model.add(Dropout(0.5))
    model.add(Dense(input_dim=128, output_dim=15, activation='softmax'))
    model.compile()
    model.train_from_dir(data_path, validation_split=0.05, batch_size=options.batch_size, verbose=options.verbose)

if options.train_nce:
    logging.info('Train NCE based language model')
    model = NCELangModel(vocab_size=15, nb_negative=2, embed_dims=128, negprob_table=negprob_table,
                         optimizer=options.optimizer)
    model.compile()
    logging.debug('compile success')
    model.train_from_dir(data_path, validation_split=0.05, batch_size=options.batch_size, verbose=options.verbose)

if options.train_nce1:
    logging.info('Train NCE based language model (1)')
    model = NCELangModelV1(vocab_size=15, nb_negative=2, embed_dims=128, negprob_table=negprob_table,
                           optimizer=options.optimizer)
    model.compile()
    logging.debug('compile success')
    model.train_from_dir(data_path, validation_split=0.05, batch_size=options.batch_size, verbose=options.verbose)

if options.tree:
    logging.info('Train hierarchical softmax language model')

    if options.tree_type == 'huffman':
        logging.info('train with Huffman Tree')
        tree_file = 'data/fake/tree-info-huffman.pkl'
    else:
        tree_file = 'data/fake/tree-info.pkl'
        logging.info('Train with Brown tree')

    with file('data/fake/tree-info.pkl', 'rb') as f:
        tree_info = pickle.load(f)
        word2cls = tree_info['idx2cls']
        word2bitstr = tree_info['idx2bitstr'].astype(floatX)

    model = TreeLangModel(vocab_size=15, embed_dim=128, cntx_dim=128, word2class=word2cls, word2bitstr=word2bitstr)
    model.compile(optimizer=options.optimizer)
    logging.debug('compile success')
    model.train_from_dir(data_path, validation_split=0.05, batch_size=options.batch_size, verbose=options.verbose)
