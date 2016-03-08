#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'
import theano
import re
import os
import math
from time import time, sleep
from theano import tensor as T
from keras.models import Sequential, Graph, make_batches
from keras import optimizers
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers import containers
# from keras.layers.embeddings import LookupTable
import numpy as np
import logging
from keras.layers.core import Reshape
# from keras.regularizers import l2
from layers import LangLSTMLayer, PartialSoftmax, Split, LookupProb, PartialSoftmaxV1, \
    TreeLogSoftmax, SparseEmbedding, Identity, PartialSoftmaxV4, SharedWeightsDense, \
    LangLSTMLayerV5, LangLSTMLayerV6, SparseEmbeddingV6, EmbeddingParam, LBLScoreV1, \
    PartialSoftmaxLBL, PartialSoftmaxLBLV4, SharedWeightsDenseLBLV4, PartialSoftmaxFFNN, \
    PartialSoftmaxV7, SharedWeightsDenseV7, PartialSoftmaxV8, SharedWeightsDenseV8
from utils import LangHistory, LangModelLogger, categorical_crossentropy, objective_fnc, \
    TableSampler, slice_X, chunk_sentences, epsilon
# noinspection PyUnresolvedReferences
from lm.utils.preprocess import import_wordmap, grouped_sentences
import theano.sparse as tsp
import cPickle as pickle
from scipy.sparse import hstack as sp_hstack, vstack as sp_vstack, csr_matrix
# from profilehooks import profile
import scipy.sparse as sparse
from multiprocessing import Queue, Process, Array, Event as MEvent
from threading import Thread, Event
from scipy import stats
import ctypes
import numba
import os

floatX = theano.config.floatX
logger = logging.getLogger('lm.real.models')
MAX_SETN_LEN = 65  # actually 64


class LogInfo(object):
    def __init__(self, file_name=None):
        super(LogInfo, self).__init__()
        if file_name is not None:
            self.logger = file(file_name, 'w')
        else:
            self.logger = None

    def info(self, message):
        if self.logger is not None:
            self.logger.writelines(['INFO:', message, '\n'])

    def debug(self, message):
        if self.logger is not None:
            self.logger.writelines(['DEBUG:', message, '\n'])

    def close(self):
        if self.logger is not None:
            self.logger.close()


class LangModel(object):
    def __init__(self):
        super(LangModel, self).__init__()

    @staticmethod
    def encode_length(y_label, y_pred, mask=None):
        """
        :param y_label: true index labels with shape (ns, nt)
        :param y_pred: predicted probabilities with shape (ns, nt, V)
        :param mask: mask
        :return: PPL
        """
        nb_rows = y_label.shape[0]
        nb_cols = y_label.shape[1]
        row_idx = T.reshape(T.arange(nb_rows), (nb_rows, 1))
        col_idx = T.reshape(T.arange(nb_cols), (1, nb_cols))
        probs_ = y_pred[row_idx, col_idx, y_label]

        if mask is None:
            nb_words = nb_rows * nb_cols
            probs = probs_.ravel() + 1.0e-37
        else:
            nb_words = mask.sum()
            probs = T.reshape(probs_, mask.shape)[mask.nonzero()] + 1.0e-37

        return -T.sum(T.log(probs)), nb_words

    @staticmethod
    def _get_shared_param(param):
        if isinstance(param, T.TensorVariable):
            # noinspection PyUnresolvedReferences
            return param.subtensor_info.base
        else:
            return param

    def save_params(self, save_path):
        # noinspection PyUnresolvedReferences
        params = [self._get_shared_param(param).get_value(borrow=True) for param in self.params]
        with file(save_path, 'wb') as f:
            pickle.dump(params, f, protocol=-1)

    def load_params(self, file_name):
        with file(file_name, 'rb') as f:
            params = pickle.load(f)
        # noinspection PyUnresolvedReferences
        for idx, param_ in enumerate(self.params):
            param = self._get_shared_param(param_)
            param.set_value(params[idx])

    def get_val_data(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', val_nb_words=100000):
        # noinspection PyUnresolvedReferences
        max_vocab = self.vocab_size - 1
        if isinstance(data_file, basestring):
            sent_gen = grouped_sentences(data_file)
        else:
            sent_gen = data_file

        val_sents = [None for _ in range(MAX_SETN_LEN)]
        val_nb = 0
        for sents in sent_gen:
            val_nb += sents.size
            chunk_sentences(val_sents, sents, 1000000, no_return=True)
            if val_nb >= val_nb_words:
                break
        val_sents_ = [None for _ in range(MAX_SETN_LEN)]
        for idx in range(MAX_SETN_LEN):
            if val_sents[idx]:
                val_sents_[idx] = np.vstack(val_sents[idx]['sents'])

        val_sents = [sents for sents in val_sents_ if sents is not None]
        for sents in val_sents:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab

        return val_sents


class SimpleLangModel(Sequential, LangModel):
    def __init__(self, vocab_size, embed_dims=128, context_dims=128, optimizer='adam'):
        super(SimpleLangModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims

        self.optimizer = optimizers.get(optimizer)
        self.loss = categorical_crossentropy
        self.loss_fnc = objective_fnc(self.loss)

        self.add(Embedding(input_dim=vocab_size, output_dim=embed_dims))
        self.add(LangLSTMLayer(input_dim=embed_dims, output_dim=context_dims))
        # self.add(Dropout(0.5))
        self.add(Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax'))

    def train_fake(self, X, y, callbacks, show_metrics, batch_size=128, extra_callbacks=(LangModelLogger(), ),
                   validation_split=0., validation_data=None, shuffle=False, verbose=1):
        self.fit(X, y, callbacks, show_metrics, batch_size=batch_size, nb_epoch=1, verbose=verbose,
                 extra_callbacks=extra_callbacks, validation_split=validation_split,
                 validation_data=validation_data, shuffle=shuffle, show_accuracy=False)

    def train_from_dir_fake(self, dir_, data_regex=re.compile(r'\d{3}.bz2'), callbacks=LangHistory(),
                            show_metrics=('loss', 'ppl'), *args, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if data_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            y = X
            logger.info('Training on %s' % f)
            self.train_fake(X, y, callbacks, show_metrics, *args, **kwargs)

    def train_from_dir(self, data_fn='wiki-sg-norm-lc-drop-bin.bz2',
                       callbacks=LangHistory(), show_metrics=('loss', 'ppl'), **kwargs):
        max_vocab = self.vocab_size - 1
        # wordmap = import_wordmap()
        # idx2word = wordmap['idx2word'][:max_vocab]

        for n, sents in enumerate(grouped_sentences(data_fn)):
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            y = sents
            if (n+1) % 1000 != 0:
                self.fit(sents, y, callbacks, show_metrics, nb_epoch=1,
                         extra_callbacks=(LangModelLogger(), ), **kwargs)
            else:
                self.evaluate(sents, y)

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(file_name=log_file)
        log_file.info('training with file: %s\n' % data_file)
        log_file.info('training with batch size %d\n' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training\n' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds\n' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            loss, ppl = self._loop_train(chunk, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('SimpleModel:Train - ETA: %02d:%02d - loss: %5.1f - ppl: %7.2f - speed: %.1f sent/s %.1f words/s' %
                        (eta_h, eta_m, loss, ppl, speed1, speed2))
            log_file.info('SimpleModel:Train - time: %f - loss: %.6f - ppl: %.6f' % (end_, loss, ppl))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        nb_words = 0.
        code_len = 0.
        nb_sents = 0.
        loss = 0.0

        for sents in val_sents:
            loss_, code_len_, nb_words_ = self._test_loop(self._test, [sents, sents], batch_size, verbose=0)
            nb_words += nb_words_
            code_len += code_len_
            nb_sents += sents.shape[0]
            loss += loss_ * sents.shape[0]

        loss /= nb_sents
        ppl = math.exp(code_len/nb_words)
        logger.info('SimpleModel:Val val_loss: %.3f - val_ppl: %.2f' % (loss, ppl))
        log_file.info('SimpleModel:Val val_loss: %.6f - val_ppl: %.6f' % (loss, ppl))

        return loss, ppl

    # noinspection PyMethodOverriding
    def compile(self, optimizer=None):
        if optimizer is not None:
            logger.info('compiling with %s' % optimizer)
            self.optimizer = optimizers.get(optimizer)
        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)

        # target of model
        self.y = T.zeros_like(self.X_train)
        self.weights = None

        if hasattr(self.layers[-1], "get_output_mask"):
            mask = self.layers[-1].get_output_mask()
        else:
            mask = None

        train_loss = self.loss_fnc(self.y, self.y_train, mask)
        test_loss = self.loss_fnc(self.y, self.y_test, mask)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        self.y.name = 'y'

        train_ce, nb_trn_wrd = self.encode_length(self.y, self.y_train, mask)
        test_ce, nb_tst_wrd = self.encode_length(self.y, self.y_test, mask)

        self.class_mode = 'categorical'
        self.theano_mode = None

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        train_ins = [self.X_train, self.y]
        test_ins = [self.X_test, self.y]

        self._train = theano.function(train_ins, [train_loss, train_ce, nb_trn_wrd], updates=updates,
                                      allow_input_downcast=True)
        self._train.out_labels = ['loss', 'encode_len', 'nb_words']
        self._test = theano.function(test_ins, [test_loss, test_ce, nb_tst_wrd], allow_input_downcast=True)
        self._test.out_labels = ['loss', 'encode_len', 'nb_words']

        self.all_metrics = ['loss', 'ppl', 'val_loss', 'val_ppl']

        def __summary_outputs(outs, batch_sizes):
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._train.summarize_outputs = __summary_outputs
        self._test.summarize_outputs = __summary_outputs

        # noinspection PyUnresolvedReferences
        self.fit = self._Sequential__fit_unweighted

    def _loop_train(self, data, batch_size):
        nb = data.shape[0]
        nb_words = 0.
        code_len = 0.
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins = data[start:end]
            loss_, code_len_, nb_words_ = self._train(ins, ins)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * ins.shape[0]

        loss /= nb
        ppl = math.exp(code_len/nb_words)
        return loss, ppl

    def evaluate(self, X, y, batch_size=128, show_accuracy=False, verbose=1, sample_weight=None):
        outs = self._test_loop(self._test, [X, y], batch_size, verbose)
        return outs


class NCELangModel(Graph, LangModel):
    def __init__(self, vocab_size, nb_negative, embed_dims=128, context_dims=128,
                 negprob_table=None, optimizer='adam'):
        super(NCELangModel, self).__init__(weighted_inputs=False)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative
        self.loss = categorical_crossentropy
        self.loss_fnc = objective_fnc(self.loss)

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            negprob_table = theano.shared(negprob_table_)
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
            negprob_table = theano.shared(negprob_table.astype(theano.config.floatX))

        self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='idxes', ndim=3, dtype='int32')
        self.add_node(Split(split_at=1, split_axis=0), name=('pos_sents', ''), inputs='idxes')

        seq = containers.Sequential()
        seq.add(self.nodes['pos_sents'])
        seq.add(Embedding(vocab_size, embed_dims))
        seq.add(LangLSTMLayer(embed_dims, output_dim=context_dims))
        # seq.add(Dropout(0.5))

        self.add_node(seq, name='seq')
        self.add_node(PartialSoftmax(input_dim=context_dims, output_dim=vocab_size),
                      name='part_prob', inputs=('idxes', 'seq'))
        self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='idxes')

        test_node = Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax')
        test_node.params = []
        test_node.W = self.nodes['part_prob'].W
        test_node.b = self.nodes['part_prob'].b
        self.add_node(test_node, name='true_prob', inputs='seq')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        y_test = T.sum(T.log(eps + pos_prob_tst[0] / sum_pos_neg_tst[0])) / nb_words
        y_test += T.sum(T.log(eps + neg_prob_tst[1:] / sum_pos_neg_tst[1:])) / nb_words

        true_labels = self.inputs['idxes'].get_output()[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        self._train = theano.function([self.inputs['idxes'].get_output(True)], outputs=train_loss,
                                      updates=updates)
        self._test = theano.function([self.inputs['idxes'].get_output(False)],
                                     outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def _loop_train(self, data, batch_size):
        nb = data.shape[1]
        nb_words = data[0].size
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins = data[:, start:end]
            loss_ = self._train(ins)
            loss += loss_ * ins[0].size

        loss /= nb_words
        return loss

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            x = self.negative_sample(chunk)
            loss = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break
        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = [self.negative_sample(sents)]
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    @staticmethod
    def _test_loop(f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[1]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=1)
            batch_outs = f(*ins_batch)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class NCELangModelV1(Graph, LangModel):
    def __init__(self, vocab_size, nb_negative, embed_dims=128, context_dims=128,
                 negprob_table=None, optimizer='adam'):
        super(NCELangModelV1, self).__init__(weighted_inputs=False)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative

        if negprob_table is None:
            self.neg_prob_table = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
        self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='pos_sents', ndim=2, dtype='int32')
        self.add_input(name='lookup_prob', ndim=3)
        self.add_input(name='unique_idxes', ndim=1, dtype='int32')
        self.add_input(name='poses', ndim=3, dtype='int32')

        seq = containers.Sequential()
        seq.add(self.inputs['pos_sents'])
        seq.add(Embedding(vocab_size, embed_dims))
        seq.add(LangLSTMLayer(embed_dims, output_dim=context_dims))
        # seq.add(Dropout(0.5))

        self.add_node(seq, name='seq')

        self.add_node(PartialSoftmaxV1(input_dim=context_dims, output_dim=vocab_size),
                      name='part_prob', inputs=('unique_idxes', 'poses', 'seq'))

        test_node = Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax')
        test_node.params = []
        test_node.W = self.nodes['part_prob'].W
        test_node.b = self.nodes['part_prob'].b
        self.add_node(test_node, name='true_prob', inputs='seq')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        y_test = T.sum(T.log(eps + pos_prob_tst[0] / sum_pos_neg_tst[0])) / nb_words
        y_test += T.sum(T.log(eps + neg_prob_tst[1:] / sum_pos_neg_tst[1:])) / nb_words

        true_labels = self.inputs['pos_sents'].get_output()
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        train_inputs = [self.inputs['pos_sents'].get_output(False),
                        self.inputs['lookup_prob'].get_output(False),
                        self.inputs['unique_idxes'].get_output(False),
                        self.inputs['poses'].get_output(False)]
        test_inputs = [self.inputs['pos_sents'].get_output(True),
                       self.inputs['lookup_prob'].get_output(True),
                       self.inputs['unique_idxes'].get_output(True),
                       self.inputs['poses'].get_output(True)]

        self._train = theano.function(train_inputs, outputs=train_loss, updates=updates)
        self._test = theano.function(test_inputs, outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def prepare_input(self, X):
        ins = X
        neg_idxes = self.negative_sample(ins)
        neg_probs = self.neg_prob_table[neg_idxes]
        unique_idxes, indeces = np.unique(neg_idxes, return_inverse=True)
        indeces = np.reshape(indeces, neg_probs.shape)
        unique_idxes = unique_idxes.astype(X.dtype)
        indeces = indeces.astype(X.dtype)
        return [ins, neg_probs, unique_idxes, indeces]

    def _loop_train(self, data, batch_size):
        nb = data.shape[0]
        nb_words = data.size
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            x = data[start:end]
            ins = self.prepare_input(x)
            loss_ = self._train(*ins)
            loss += loss_ * x.size

        loss /= nb_words
        return loss

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800):

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %.0f seconds' % float(validation_interval))
        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            loss = self._loop_train(chunk, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size)
                if save_path is not None:
                    self.save_params(save_path)
                break

    def validation(self, val_sents, batch_size):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            loss_, code_len_, nb_words_ = self._test_loop(self._test, [sents], batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    def _test_loop(self, f, ins, batch_size=128):
        nb_sample = ins[0].shape[0]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            x_slice = slice_X(ins, start_=batch_start, end_=batch_end, axis=0)[0]
            ins_batch = self.prepare_input(x_slice)
            batch_outs = f(*ins_batch)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class NCELangModelV2(Graph, LangModel):
    def __init__(self, vocab_size, nb_negative, embed_dims=128, context_dims=128,
                 negprob_table=None, optimizer='adam'):
        super(NCELangModelV2, self).__init__(weighted_inputs=False)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative
        self.loss = categorical_crossentropy
        self.loss_fnc = objective_fnc(self.loss)

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            negprob_table = theano.shared(negprob_table_)
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
            negprob_table = theano.shared(negprob_table.astype(theano.config.floatX))

        self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='idxes', ndim=3, dtype='int32')
        self.add_node(Split(split_at=1, split_axis=0), name=('pos_sents', ''), inputs='idxes')

        seq = containers.Sequential()
        seq.add(self.nodes['pos_sents'])
        seq.add(Embedding(vocab_size, embed_dims))
        seq.add(LangLSTMLayer(embed_dims, output_dim=context_dims))
        # seq.add(Dropout(0.5))

        self.add_node(seq, name='seq')
        self.add_node(PartialSoftmax(input_dim=context_dims, output_dim=vocab_size),
                      name='part_prob', inputs=('idxes', 'seq'))
        self.add_node(Dense(input_dim=context_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='seq')
        self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='idxes')

        test_node = Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax')
        test_node.params = []
        test_node.W = self.nodes['part_prob'].W
        test_node.b = self.nodes['part_prob'].b
        self.add_node(test_node, name='true_prob', inputs='seq')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0], nrm_const.shape[1]))
        nrm_const = nrm_const.dimshuffle('x', 0, 1)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], nrm_const_tst.shape[1]))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0, 1)
        pos_prob_tst *= nrm_const_tst

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        y_test = T.sum(T.log(eps + pos_prob_tst[0] / sum_pos_neg_tst[0])) / nb_words
        y_test += T.sum(T.log(eps + neg_prob_tst[1:] / sum_pos_neg_tst[1:])) / nb_words

        true_labels = self.inputs['idxes'].get_output()[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        self._train = theano.function([self.inputs['idxes'].get_output(True)], outputs=train_loss,
                                      updates=updates)
        self._test = theano.function([self.inputs['idxes'].get_output(False)],
                                     outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def _loop_train(self, data, batch_size):
        nb = data.shape[1]
        nb_words = data[0].size
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins = data[:, start:end]
            loss_ = self._train(ins)
            loss += loss_ * ins[0].size

        loss /= nb_words
        return loss

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            x = self.negative_sample(chunk)
            loss = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break
        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = [self.negative_sample(sents)]
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    @staticmethod
    def _test_loop(f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[1]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=1)
            batch_outs = f(*ins_batch)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class NCELangModelV3(Graph, LangModel):
    def __init__(self, sparse_coding, nb_negative, embed_dims=128, context_dims=128,
                 init_embeddings=None, negprob_table=None, optimizer='adam'):
        super(NCELangModelV3, self).__init__(weighted_inputs=False)
        vocab_size = sparse_coding.shape[0]  # the extra word is for OOV
        self.nb_base = sparse_coding.shape[1] - 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative
        self.loss = categorical_crossentropy
        self.loss_fnc = objective_fnc(self.loss)
        self.sparse_coding = sparse_coding

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            negprob_table = theano.shared(negprob_table_)
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
            negprob_table = theano.shared(negprob_table.astype(theano.config.floatX))

        self.sampler = TableSampler(self.neg_prob_table)

        codes = tsp.csr_matrix('sparse-codes', dtype=floatX)
        shape = T.ivector('sents-shape')

        self.add_node(Identity(inputs={True: codes, False: codes}), name='codes_flat')
        self.add_node(Identity(inputs={True: shape, False: shape}), name='sents_shape')
        self.add_input(name='idxes', ndim=3, dtype='int32')

        self.add_node(SparseEmbedding(self.nb_base+1, embed_dims, weights=init_embeddings),
                      name='embedding', inputs=('codes_flat', 'sents_shape'))
        self.add_node(LangLSTMLayer(embed_dims, output_dim=context_dims), name='encoder', inputs='embedding')
        # seq.add(Dropout(0.5))
        self.add_node(PartialSoftmax(input_dim=context_dims, output_dim=vocab_size),
                      name='part_prob', inputs=('idxes', 'encoder'))
        self.add_node(Dense(input_dim=context_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='encoder')
        self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='idxes')

        test_node = Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax')
        test_node.params = []
        test_node.W = self.nodes['part_prob'].W
        test_node.b = self.nodes['part_prob'].b
        self.add_node(test_node, name='true_prob', inputs='encoder')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0], nrm_const.shape[1]))
        nrm_const = nrm_const.dimshuffle('x', 0, 1)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], nrm_const_tst.shape[1]))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0, 1)
        pos_prob_tst *= nrm_const_tst

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        y_test = T.sum(T.log(eps + pos_prob_tst[0] / sum_pos_neg_tst[0])) / nb_words
        y_test += T.sum(T.log(eps + neg_prob_tst[1:] / sum_pos_neg_tst[1:])) / nb_words

        input0 = self.inputs['idxes'].get_output(True)
        input1 = self.nodes['sents_shape'].get_output(True)
        input2 = self.nodes['codes_flat'].get_output(True)

        true_labels = input0[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        self._train = theano.function([input0, input1, input2], outputs=train_loss,
                                      updates=updates)
        self._test = theano.function([input0, input1, input2],
                                     outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def _loop_train(self, data, batch_size):
        nb = data.shape[1]
        nb_words = data[0].size
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins0 = data[:, start:end]
            ins = self.prepare_input(ins0)
            loss_ = self._train(*ins)
            loss += loss_ * ins0[0].size

        loss /= nb_words
        return loss

    def prepare_input(self, data):
        """
        :param data:
        :type data: numpy.ndarray
        :return:
        """
        x = [None] * 3
        x[0] = data
        x[1] = data[0].shape
        idx = data[0].ravel()
        x[2] = self.sparse_coding[idx]
        return x

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            x = self.negative_sample(chunk)
            loss = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break
        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = [self.negative_sample(sents)]
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[1]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=1)
            x = self.prepare_input(ins_batch[0])
            batch_outs = f(*x)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class NCELangModelV4(Graph, LangModel):
    def __init__(self, sparse_coding, nb_negative, embed_dims=128, context_dims=128,
                 init_embeddings=None, negprob_table=None, optimizer='adam'):
        super(NCELangModelV4, self).__init__(weighted_inputs=False)
        vocab_size = sparse_coding.shape[0]  # the extra word is for OOV
        self.nb_base = sparse_coding.shape[1] - 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative
        self.loss = categorical_crossentropy
        self.loss_fnc = objective_fnc(self.loss)
        self.sparse_coding = sparse_coding

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            negprob_table = theano.shared(negprob_table_, borrow=True)
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
            negprob_table = theano.shared(self.neg_prob_table, borrow=True)

        self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='idxes', ndim=3, dtype='int32')
        idxes = self.inputs['idxes'].get_output(True)
        shape = idxes.shape[1:]
        codes = tsp.csr_matrix('sp-codes', dtype=floatX)
        nb_pos_words = shape[0] * shape[1]
        pos_codes = codes[:nb_pos_words]

        self.add_node(Identity(inputs={True: pos_codes, False: pos_codes}), name='codes_flat')
        self.add_node(Identity(inputs={True: shape, False: shape}), name='sents_shape')
        self.add_node(Identity(inputs={True: codes, False: codes}), name='sparse_codes')

        self.add_node(SparseEmbedding(self.nb_base+1, embed_dims, weights=init_embeddings),
                      name='embedding', inputs=('codes_flat', 'sents_shape'))
        self.add_node(LangLSTMLayer(embed_dims, output_dim=context_dims), name='encoder', inputs='embedding')
        # seq.add(Dropout(0.5))
        self.add_node(PartialSoftmaxV4(input_dim=context_dims, base_size=self.nb_base+1),
                      name='part_prob', inputs=('idxes', 'sparse_codes', 'encoder'))
        self.add_node(Dense(input_dim=context_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='encoder')
        self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='idxes')

        # test_node = Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax')
        # test_node.params = []
        # test_node.W = self.nodes['part_prob'].W
        # test_node.b = self.nodes['part_prob'].b
        # self.add_node(test_node, name='true_prob', inputs='encoder')
        self.add_node(SharedWeightsDense(self.nodes['part_prob'].W, self.nodes['part_prob'].b, self.sparse_coding,
                                         activation='softmax'),
                      name='true_prob', inputs='encoder')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0], nrm_const.shape[1]))
        nrm_const = nrm_const.dimshuffle('x', 0, 1)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], nrm_const_tst.shape[1]))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0, 1)
        pos_prob_tst *= nrm_const_tst

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        y_test = T.sum(T.log(eps + pos_prob_tst[0] / sum_pos_neg_tst[0])) / nb_words
        y_test += T.sum(T.log(eps + neg_prob_tst[1:] / sum_pos_neg_tst[1:])) / nb_words

        input0 = self.inputs['idxes'].get_output(True)
        input1 = self.nodes['sparse_codes'].get_output(True)

        true_labels = input0[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        self._train = theano.function([input0, input1], outputs=train_loss,
                                      updates=updates)
        self._test = theano.function([input0, input1],
                                     outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def _loop_train(self, data, batch_size):
        nb = data.shape[1]
        nb_words = data[0].size
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins0 = data[:, start:end]
            ins = self.prepare_input(ins0)
            loss_ = self._train(*ins)
            loss += loss_ * ins0[0].size

        loss /= nb_words
        return loss

    def prepare_input(self, data):
        """
        :param data:
        :type data: numpy.ndarray
        :return:
        """
        x = [None] * 2
        x[0] = data
        idx = x[0].ravel()
        x[1] = self.sparse_coding[idx]
        return x

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            x = self.negative_sample(chunk)
            loss = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break

        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = [self.negative_sample(sents)]
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[1]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=1)
            x = self.prepare_input(ins_batch[0])
            batch_outs = f(*x)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class NCELangModelV5(Graph, LangModel):
    def __init__(self, sparse_coding, nb_negative, embed_dims=128,
                 init_embeddings=None, negprob_table=None, optimizer='adam'):
        super(NCELangModelV5, self).__init__(weighted_inputs=False)
        vocab_size = sparse_coding.shape[0]  # the extra word is for OOV
        self.nb_base = sparse_coding.shape[1] - 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative
        self.loss = categorical_crossentropy
        self.loss_fnc = objective_fnc(self.loss)
        self.sparse_coding = sparse_coding

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            negprob_table = theano.shared(negprob_table_)
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
            negprob_table = theano.shared(negprob_table.astype(theano.config.floatX))

        self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='idxes', ndim=3, dtype='int32')
        idxes = self.inputs['idxes'].get_output(True)
        shape = idxes.shape[1:]
        codes = tsp.csr_matrix('sp-codes', dtype=floatX)
        nb_pos_words = shape[0] * shape[1]
        pos_codes = codes[:nb_pos_words]

        self.add_node(Identity(inputs={True: pos_codes, False: pos_codes}), name='codes_flat')
        self.add_node(Identity(inputs={True: shape, False: shape}), name='sents_shape')
        self.add_node(Identity(inputs={True: codes, False: codes}), name='sparse_codes')

        self.add_node(SparseEmbedding(self.nb_base+1, embed_dims, weights=init_embeddings),
                      name='embedding', inputs=('codes_flat', 'sents_shape'))
        self.add_node(LangLSTMLayerV5(embed_dims), name='encoder', inputs='embedding')
        # seq.add(Dropout(0.5))
        self.add_node(PartialSoftmaxV4(input_dim=embed_dims, base_size=self.nb_base+1),
                      name='part_prob', inputs=('idxes', 'sparse_codes', 'encoder'))
        self.add_node(Dense(input_dim=embed_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='encoder')
        self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='idxes')

        # test_node = Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax')
        # test_node.params = []
        # test_node.W = self.nodes['part_prob'].W
        # test_node.b = self.nodes['part_prob'].b
        # self.add_node(test_node, name='true_prob', inputs='encoder')
        self.add_node(SharedWeightsDense(self.nodes['part_prob'].W, self.nodes['part_prob'].b, self.sparse_coding,
                                         activation='softmax'),
                      name='true_prob', inputs='encoder')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0], nrm_const.shape[1]))
        nrm_const = nrm_const.dimshuffle('x', 0, 1)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], nrm_const_tst.shape[1]))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0, 1)
        pos_prob_tst *= nrm_const_tst

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        y_test = T.sum(T.log(eps + pos_prob_tst[0] / sum_pos_neg_tst[0])) / nb_words
        y_test += T.sum(T.log(eps + neg_prob_tst[1:] / sum_pos_neg_tst[1:])) / nb_words

        input0 = self.inputs['idxes'].get_output(True)
        input1 = self.nodes['sparse_codes'].get_output(True)

        true_labels = input0[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        self._train = theano.function([input0, input1], outputs=train_loss,
                                      updates=updates)
        self._test = theano.function([input0, input1],
                                     outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def _loop_train(self, data, batch_size):
        nb = data.shape[1]
        nb_words = data[0].size
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins0 = data[:, start:end]
            ins = self.prepare_input(ins0)
            loss_ = self._train(*ins)
            loss += loss_ * ins0[0].size

        loss /= nb_words
        return loss

    def prepare_input(self, data):
        """
        :param data:
        :type data: numpy.ndarray
        :return:
        """
        x = [None] * 2
        x[0] = data
        idx = x[0].ravel()
        x[1] = self.sparse_coding[idx]
        return x

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            x = self.negative_sample(chunk)
            loss = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break

        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = [self.negative_sample(sents)]
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[1]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=1)
            x = self.prepare_input(ins_batch[0])
            batch_outs = f(*x)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class NCELangModelV6(Graph, LangModel):
    def __init__(self, sparse_coding, nb_negative, embed_dims=128,
                 init_embeddings=None, negprob_table=None, optimizer='adam'):
        super(NCELangModelV6, self).__init__(weighted_inputs=False)
        vocab_size = sparse_coding.shape[0]  # the extra word is for OOV
        self.nb_base = sparse_coding.shape[1] - 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative
        self.loss = categorical_crossentropy
        self.loss_fnc = objective_fnc(self.loss)
        self.sparse_coding = sparse_coding

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            negprob_table = theano.shared(negprob_table_)
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
            negprob_table = theano.shared(negprob_table.astype(theano.config.floatX))

        self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='idxes', ndim=3, dtype='int32')
        idxes = self.inputs['idxes'].get_output(True)
        shape = idxes.shape[1:]
        codes = tsp.csr_matrix('sp-codes', dtype=floatX)
        nb_pos_words = shape[0] * shape[1]
        pos_codes = codes[:nb_pos_words]

        self.add_node(Identity(inputs={True: pos_codes, False: pos_codes}), name='codes_flat')
        self.add_node(Identity(inputs={True: shape, False: shape}), name='sents_shape')
        self.add_node(Identity(inputs={True: codes, False: codes}), name='sparse_codes')

        self.add_node(SparseEmbeddingV6(self.nb_base+1, embed_dims, weights=init_embeddings),
                      name='embedding', inputs=('codes_flat', 'sents_shape'))
        self.add_node(LangLSTMLayerV6(embed_dims), name='encoder', inputs='embedding')
        # seq.add(Dropout(0.5))
        self.add_node(PartialSoftmaxV4(input_dim=embed_dims, base_size=self.nb_base+1),
                      name='part_prob', inputs=('idxes', 'sparse_codes', 'encoder'))
        self.add_node(Dense(input_dim=embed_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='encoder')
        self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='idxes')

        # test_node = Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax')
        # test_node.params = []
        # test_node.W = self.nodes['part_prob'].W
        # test_node.b = self.nodes['part_prob'].b
        # self.add_node(test_node, name='true_prob', inputs='encoder')
        self.add_node(SharedWeightsDense(self.nodes['part_prob'].W, self.nodes['part_prob'].b, self.sparse_coding,
                                         activation='softmax'),
                      name='true_prob', inputs='encoder')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0], nrm_const.shape[1]))
        nrm_const = nrm_const.dimshuffle('x', 0, 1)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], nrm_const_tst.shape[1]))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0, 1)
        pos_prob_tst *= nrm_const_tst

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        y_test = T.sum(T.log(eps + pos_prob_tst[0] / sum_pos_neg_tst[0])) / nb_words
        y_test += T.sum(T.log(eps + neg_prob_tst[1:] / sum_pos_neg_tst[1:])) / nb_words

        input0 = self.inputs['idxes'].get_output(True)
        input1 = self.nodes['sparse_codes'].get_output(True)

        true_labels = input0[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        self._train = theano.function([input0, input1], outputs=train_loss,
                                      updates=updates)
        self._test = theano.function([input0, input1],
                                     outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def _loop_train(self, data, batch_size):
        nb = data.shape[1]
        nb_words = data[0].size
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins0 = data[:, start:end]
            ins = self.prepare_input(ins0)
            loss_ = self._train(*ins)
            loss += loss_ * ins0[0].size

        loss /= nb_words
        return loss

    def prepare_input(self, data):
        """
        :param data:
        :type data: numpy.ndarray
        :return:
        """
        x = [None] * 2
        x[0] = data
        idx = x[0].ravel()
        x[1] = self.sparse_coding[idx]
        return x

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            x = self.negative_sample(chunk)
            loss = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break
        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = [self.negative_sample(sents)]
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[1]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=1)
            x = self.prepare_input(ins_batch[0])
            batch_outs = f(*x)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class NCELangModelV7(Graph, LangModel):
    """ extend V4, bias for softmax not compressed.
    """
    def __init__(self, sparse_coding, nb_negative, embed_dims=128, context_dims=128,
                 init_embeddings=None, negprob_table=None, optimizer='adam'):
        super(NCELangModelV7, self).__init__(weighted_inputs=False)
        vocab_size = sparse_coding.shape[0]  # the extra word is for OOV
        self.nb_base = sparse_coding.shape[1] - 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative
        self.loss = categorical_crossentropy
        self.loss_fnc = objective_fnc(self.loss)
        self.sparse_coding = sparse_coding

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            negprob_table = theano.shared(negprob_table_, borrow=True)
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
            negprob_table = theano.shared(self.neg_prob_table, borrow=True)

        self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='idxes', ndim=3, dtype='int32')
        idxes = self.inputs['idxes'].get_output(True)
        shape = idxes.shape[1:]
        codes = tsp.csr_matrix('sp-codes', dtype=floatX)
        nb_pos_words = shape[0] * shape[1]
        pos_codes = codes[:nb_pos_words]

        self.add_node(Identity(inputs={True: pos_codes, False: pos_codes}), name='codes_flat')
        self.add_node(Identity(inputs={True: shape, False: shape}), name='sents_shape')
        self.add_node(Identity(inputs={True: codes, False: codes}), name='sparse_codes')

        self.add_node(SparseEmbedding(self.nb_base+1, embed_dims, weights=init_embeddings),
                      name='embedding', inputs=('codes_flat', 'sents_shape'))
        self.add_node(LangLSTMLayer(embed_dims, output_dim=context_dims), name='encoder', inputs='embedding')
        # seq.add(Dropout(0.5))
        self.add_node(PartialSoftmaxV7(input_dim=context_dims, base_size=self.nb_base+1, vocab_size=self.vocab_size),
                      name='part_prob', inputs=('idxes', 'sparse_codes', 'encoder'))
        self.add_node(Dense(input_dim=context_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='encoder')
        self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='idxes')

        # test_node = Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax')
        # test_node.params = []
        # test_node.W = self.nodes['part_prob'].W
        # test_node.b = self.nodes['part_prob'].b
        # self.add_node(test_node, name='true_prob', inputs='encoder')
        self.add_node(SharedWeightsDenseV7(self.nodes['part_prob'].W, self.nodes['part_prob'].b, self.sparse_coding,
                                           activation='softmax'),
                      name='true_prob', inputs='encoder')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0], nrm_const.shape[1]))
        nrm_const = nrm_const.dimshuffle('x', 0, 1)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], nrm_const_tst.shape[1]))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0, 1)
        pos_prob_tst *= nrm_const_tst

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        y_test = T.sum(T.log(eps + pos_prob_tst[0] / sum_pos_neg_tst[0])) / nb_words
        y_test += T.sum(T.log(eps + neg_prob_tst[1:] / sum_pos_neg_tst[1:])) / nb_words

        input0 = self.inputs['idxes'].get_output(True)
        input1 = self.nodes['sparse_codes'].get_output(True)

        true_labels = input0[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        self._train = theano.function([input0, input1], outputs=train_loss,
                                      updates=updates)
        self._test = theano.function([input0, input1],
                                     outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def _loop_train(self, data, batch_size):
        nb = data.shape[1]
        nb_words = data[0].size
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins0 = data[:, start:end]
            ins = self.prepare_input(ins0)
            loss_ = self._train(*ins)
            loss += loss_ * ins0[0].size

        loss /= nb_words
        return loss

    def prepare_input(self, data):
        """
        :param data:
        :type data: numpy.ndarray
        :return:
        """
        x = [None] * 2
        x[0] = data
        idx = x[0].ravel()
        x[1] = self.sparse_coding[idx]
        return x

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            x = self.negative_sample(chunk)
            loss = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break

        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = [self.negative_sample(sents)]
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[1]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=1)
            x = self.prepare_input(ins_batch[0])
            batch_outs = f(*x)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class NCELangModelV8(Graph, LangModel):
    """ extend V4, bias for softmax not compressed.
    """
    def __init__(self, sparse_coding, nb_negative, embed_dims=128, context_dims=128,
                 init_embeddings=None, negprob_table=None, optimizer='adam'):
        super(NCELangModelV8, self).__init__(weighted_inputs=False)
        vocab_size = sparse_coding.shape[0]  # the extra word is for OOV
        self.nb_base = sparse_coding.shape[1] - 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative
        self.loss = categorical_crossentropy
        self.loss_fnc = objective_fnc(self.loss)
        self.sparse_coding = sparse_coding

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            negprob_table = theano.shared(negprob_table_, borrow=True)
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
            negprob_table = theano.shared(self.neg_prob_table, borrow=True)

        self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='idxes', ndim=3, dtype='int32')
        idxes = self.inputs['idxes'].get_output(True)
        shape = idxes.shape[1:]
        codes = tsp.csr_matrix('sp-codes', dtype=floatX)
        nb_pos_words = shape[0] * shape[1]
        pos_codes = codes[:nb_pos_words]

        self.add_node(Identity(inputs={True: pos_codes, False: pos_codes}), name='codes_flat')
        self.add_node(Identity(inputs={True: shape, False: shape}), name='sents_shape')
        self.add_node(Identity(inputs={True: codes, False: codes}), name='sparse_codes')

        self.add_node(SparseEmbedding(self.nb_base+1, embed_dims, weights=init_embeddings),
                      name='embedding', inputs=('codes_flat', 'sents_shape'))
        self.add_node(LangLSTMLayer(embed_dims, output_dim=context_dims), name='encoder', inputs='embedding')
        # seq.add(Dropout(0.5))
        self.add_node(PartialSoftmaxV8(input_dim=context_dims, base_size=self.nb_base+1),
                      name='part_prob', inputs=('idxes', 'sparse_codes', 'encoder'))
        self.add_node(Dense(input_dim=context_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='encoder')
        self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='idxes')

        # test_node = Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax')
        # test_node.params = []
        # test_node.W = self.nodes['part_prob'].W
        # test_node.b = self.nodes['part_prob'].b
        # self.add_node(test_node, name='true_prob', inputs='encoder')
        self.add_node(SharedWeightsDenseV8(self.nodes['part_prob'].W, self.nodes['part_prob'].b, self.sparse_coding,
                                           activation='softmax'),
                      name='true_prob', inputs='encoder')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0], nrm_const.shape[1]))
        nrm_const = nrm_const.dimshuffle('x', 0, 1)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], nrm_const_tst.shape[1]))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0, 1)
        pos_prob_tst *= nrm_const_tst

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        y_test = T.sum(T.log(eps + pos_prob_tst[0] / sum_pos_neg_tst[0])) / nb_words
        y_test += T.sum(T.log(eps + neg_prob_tst[1:] / sum_pos_neg_tst[1:])) / nb_words

        input0 = self.inputs['idxes'].get_output(True)
        input1 = self.nodes['sparse_codes'].get_output(True)

        true_labels = input0[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        self._train = theano.function([input0, input1], outputs=train_loss,
                                      updates=updates)
        self._test = theano.function([input0, input1],
                                     outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def _loop_train(self, data, batch_size):
        nb = data.shape[1]
        nb_words = data[0].size
        loss = 0.0
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins0 = data[:, start:end]
            ins = self.prepare_input(ins0)
            loss_ = self._train(*ins)
            loss += loss_ * ins0[0].size

        loss /= nb_words
        return loss

    def prepare_input(self, data):
        """
        :param data:
        :type data: numpy.ndarray
        :return:
        """
        x = [None] * 2
        x[0] = data
        idx = x[0].ravel()
        x[1] = self.sparse_coding[idx]
        return x

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            x = self.negative_sample(chunk)
            loss = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break

        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = [self.negative_sample(sents)]
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[1]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=1)
            x = self.prepare_input(ins_batch[0])
            batch_outs = f(*x)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class TreeLangModel(Graph, LangModel):
    def __init__(self, vocab_size, embed_dim, cntx_dim, word2class, word2bitstr, optimizer='adam'):
        super(TreeLangModel, self).__init__()

        self.add_input('cls_idx', ndim=3, dtype='int32')
        self.add_input('word_bitstr_mask', ndim=3, dtype=floatX)
        self.add_input('word_idx', ndim=2, dtype='int32')

        seq = containers.Sequential()
        seq.add(Embedding(vocab_size, output_dim=embed_dim))
        seq.add(LangLSTMLayer(embed_dim, output_dim=cntx_dim))
        # seq.add(Dense(input_dim=cntx_dim, output_dim=cntx_dim, activation='sigmoid'))
        # seq.add(Dropout(0.5))

        self.add_node(seq, name='seq', inputs='word_idx')
        self.add_node(TreeLogSoftmax(vocab_size, embed_dim=cntx_dim), name='tree_softmax',
                      inputs=('seq', 'cls_idx', 'word_bitstr_mask'))

        self.add_output('tree_softmax', 'tree_softmax')
        self.add_output('true_labels', 'word_idx')

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = cntx_dim
        self.optimizer = optimizers.get(optimizer)
        self.word2class = word2class
        self.word2bitstr = word2bitstr

    @staticmethod
    def encode_length(y_label, y_pred_log, mask=None):
        if mask is None:
            nb_words = y_label.shape[0] * y_label.shape[1]
            log_probs = y_pred_log.ravel() + 1.0e-37
        else:
            nb_words = mask.sum()
            log_probs = y_pred_log[mask.nonzero()] + 1.0e-37

        return -T.sum(log_probs), nb_words

    # noinspection PyMethodOverriding
    def compile(self):
        logprob_layer = self.outputs['tree_softmax']
        logprob_trn = logprob_layer.get_output(train=True)
        logprob_tst = logprob_layer.get_output(train=False)

        # eps = 1.0e-37
        #TODO: mask not supported here
        # nb_words = logprob_trn.size.astype(theano.config.floatX)
        # train_loss = -T.sum(logprob_trn) / nb_words
        # test_loss = -T.sum(logprob_tst) / nb_words
        nb_sents = logprob_trn.shape[0].astype(theano.config.floatX)
        train_loss = -T.sum(logprob_trn) / nb_sents
        test_loss = -T.sum(logprob_tst) / nb_sents

        true_labels = self.outputs['true_labels'].get_output(train=True)
        #TODO: mask not supported here
        encode_len, nb_words = self.encode_length(true_labels, logprob_tst)

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        train_inputs = [self.inputs['word_idx'].get_output(True),
                        self.inputs['cls_idx'].get_output(True),
                        self.inputs['word_bitstr_mask'].get_output(True)]
        test_inputs = [self.inputs['word_idx'].get_output(False),
                       self.inputs['cls_idx'].get_output(False),
                       self.inputs['word_bitstr_mask'].get_output(False)]

        self._train = theano.function(train_inputs, outputs=train_loss, updates=updates)
        self._test = theano.function(test_inputs, outputs=[test_loss, encode_len, nb_words])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words')
        self.all_metrics = ['loss', 'val_loss', 'val_ppl']

        def __summarize_outputs(outs, batch_sizes):
            """
                :param outs: outputs of the _test* function. It is a list, and each element a list of
                values of the outputs of the _test* function on corresponding batch.
                :type outs: list
                :param batch_sizes: batch sizes. A list with the same length with outs. Each element
                is a size of corresponding batch.
                :type batch_sizes: list
                Aggregate outputs of batches as if the test function evaluates
                the metric values on the union of the batches.
                Note this function must be redefined for each specific problem
            """
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def prepare_input(self, X):
        ins = [X, None, None]
        ins[1] = self.word2class[ins[0]]
        ins[2] = self.word2bitstr[ins[0]]
        return ins

    def _loop_train(self, data, batch_size):
        nb = data.shape[0]
        loss = 0.0
        X = self.prepare_input(data)
        for start in xrange(0, nb, batch_size):
            end = start + batch_size
            ins = [x[start:end] for x in X]
            loss_ = self._train(*ins)
            loss += loss_ * ins[0].shape[0]

        loss /= nb
        return loss

    @staticmethod
    def _test_loop(f, ins, batch_size=128, verbose=0):
        """
            Abstract method to loop over some data in batches.
        """
        nb_sample = len(ins[0])
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=0)
            batch_outs = f(*ins_batch)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(ins_batch[0].shape[0])

        outs = f.summarize_outputs(outs, batch_info)
        return outs

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0
        nb_sents = 0.

        for sents in val_sents:
            nb_sents += sents.shape[0]
            x = self.prepare_input(sents)
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * sents.shape[0]

        loss /= nb_sents
        ppl = math.exp(code_len/nb_words)
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            # loss, ce, nb_wrd = self._train(chunk, chunk)
            loss = self._loop_train(chunk, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.1f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break

        log_file.close()


class LBLangModelV1(Graph, LangModel):
    # the standard LBL language model
    def __init__(self, vocab_size, context_size, embed_dims=128, optimizer='adam'):
        super(LBLangModelV1, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        # self.loss = categorical_crossentropy
        # self.loss_fnc = objective_fnc(self.loss)
        self.optimizer = optimizers.get(optimizer)
        self.context_size = context_size
        self.weights = None
        # self.max_sent_len = max_sent_len

        self.add_input(name='ngrams', ndim=2, dtype='int32')

        # self.add_node(Embedding(vocab_size+context_size, embed_dims, W_regularizer=l2(0.0005/vocab_size)),
        #               name='embedding', inputs='ngrams')
        self.add_node(Embedding(vocab_size+context_size, embed_dims), name='embedding', inputs='ngrams')
        self.add_node(EmbeddingParam(), name='embedding_param', inputs='embedding')
        self.add_node(Reshape(-1), name='reshape', inputs='embedding')
        composer_node = Dense(context_size*embed_dims, embed_dims)
        composer_node.params = [composer_node.W]   # drop the bias parameters
        # composer_node.W_regularizer = l2(0.0001)
        # composer_node.W_regularizer.set_param(composer_node.W)
        # composer_node.regularizers = [composer_node.W_regularizer]
        # del composer_node.b
        # replace the default behavior of Dense
        composer_node.get_output = lambda train: node_get_output(composer_node, train)
        self.add_node(composer_node, name='context_vec', inputs='reshape')
        self.add_node(LBLScoreV1(vocab_size), name='score', inputs=('context_vec', 'embedding_param'))
        # self.add_node(LBLScoreV1(vocab_size, b_regularizer=l2(0.0001)),
        #               name='score', inputs=('context_vec', 'embedding_param'))

        self.add_output('prob', 'score')

        def node_get_output(layer, train=False):
            X = layer.get_input(train)
            output = T.dot(X, layer.W)  # there is no activation.
            return output

        self.fit = None

    @staticmethod
    def encode_length(y_label, y_pred, mask=None):
        """
        :param y_label: true index labels with shape (ns, )
        :param y_pred: predicted probabilities with shape (ns, V)
        :param mask: mask
        :return: PPL
        """
        ## there is no need to clip here, for the prob. have already clipped by LBLayer
        # epsilon = 1e-7
        # y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
        # # scale preds so that the class probas of each sample sum to 1
        # y_pred /= y_pred.sum(axis=-1, keepdims=True)

        nb_samples = y_label.shape[0]
        idx = T.arange(nb_samples)
        probs_ = y_pred[idx, y_label]

        return -T.sum(T.log(probs_)), nb_samples

    # noinspection PyMethodOverriding
    def compile(self, optimizer=None):
        # from theano.compile.nanguardmode import NanGuardMode
        if optimizer is not None:
            logger.info('compiling with %s' % optimizer)
            self.optimizer = optimizers.get(optimizer)
        # input of model
        self.X_train = self.get_input(train=True)   # (n, m)   # assuming (m+1)-gram
        self.X_test = self.get_input(train=False)   # (n, m)

        self.y_train = self.get_output(train=True)  # (n, V)
        self.y_test = self.get_output(train=False)  # (n, V)

        # self.y_train, cntx_nrm = self.get_output(train=True)  # (n, V)
        # self.y_test, _ = self.get_output(train=False)  # (n, V)
        # nrm = self.nodes['embedding_param'].get_max_norm()

        # todo: mask support
        mask = None
        self.y = T.vector('y', dtype='int32')

        train_ce, nb_trn_wrd = self.encode_length(self.y, self.y_train, mask)
        test_ce, nb_tst_wrd = self.encode_length(self.y, self.y_test, mask)

        train_loss = train_ce / nb_trn_wrd.astype(floatX)
        test_loss = test_ce / nb_tst_wrd.astype(floatX)
        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        train_ins = [self.X_train, self.y]
        test_ins = [self.X_test, self.y]
        # predict_ins = [self.X_test]

        # self._train = theano.function(train_ins, [train_loss, train_ce, nb_trn_wrd, nrm, cntx_nrm], updates=updates,
        #                               allow_input_downcast=True,
        #                               mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        # self._train.out_labels = ['loss', 'encode_len', 'nb_words', 'max_norm', 'max_cntxt_nrm']

        self._train = theano.function(train_ins, [train_loss, train_ce, nb_trn_wrd], updates=updates,
                                      allow_input_downcast=True)
        self._train.out_labels = ['loss', 'encode_len', 'nb_words']
        # self._predict = theano.function(predict_ins, self.y_test, allow_input_downcast=True)
        # self._predict.out_labels = ['predicted']
        self._test = theano.function(test_ins, [test_loss, test_ce, nb_tst_wrd], allow_input_downcast=True)
        self._test.out_labels = ['loss', 'encode_len', 'nb_words']

        # self._train_with_acc = theano.function(train_ins, [train_loss, train_accuracy, train_ce, nb_trn_wrd],
        #                                        updates=updates,
        #                                        allow_input_downcast=True, mode=theano_mode)
        # self._test_with_acc = theano.function(test_ins, [test_loss, test_accuracy],
        #                                       allow_input_downcast=True, mode=theano_mode)

        # self.__compile_fncs(train_ins, train_loss, test_ins, test_loss, predict_ins, updates)

        self.all_metrics = ['loss', 'ppl', 'val_loss', 'val_ppl']

        # self._train.label2idx = dict((l, idx) for idx, l in enumerate(['loss', 'encode_len', 'nb_words']))
        # self._test.label2idx = dict((l, idx) for idx, l in enumerate(['loss', 'encode_len', 'nb_words']))
        #
        # def __get_metrics_values(f, outs, metrics, prefix=''):
        #     ret = []
        #     label2idx = f.label2idx
        #     for mtrx in metrics:
        #         if mtrx == 'loss':
        #             idx = label2idx[mtrx]
        #             ret.append((prefix+mtrx, outs[idx]))
        #         elif mtrx == 'ppl':
        #             nb_words = outs[label2idx['nb_words']]
        #             encode_len = outs[label2idx['encode_len']]
        #             ret.append((prefix+'ppl', math.exp(float(encode_len)/float(nb_words))))
        #         else:
        #             logger.warn('Specify UNKNOWN metrics ignored')
        #     return ret

        def __summary_outputs(outs, batch_sizes):
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        # # self._train_with_acc.get_metrics_values = lambda outs, metrics, prefix='': \
        # #     __get_metrics_values(self._train_with_acc, outs, metrics, prefix)
        # self._train.get_metrics_values = lambda outs, metrics, prefix='': \
        #     __get_metrics_values(self._train, outs, metrics, prefix)
        # self._test.get_metrics_values = lambda outs, metrics, prefix='': \
        #     __get_metrics_values(self._test, outs, metrics, prefix)
        # # self._test_with_acc.get_metrics_values = lambda outs, metrics, prefix='': \
        # #     __get_metrics_values(self._test_with_acc, outs, metrics, prefix)

        # self._train_with_acc.summary_outputs = __summarize_outputs
        self._train.summarize_outputs = __summary_outputs
        self._test.summarize_outputs = __summary_outputs
        # self._test_with_acc.summary_outputs = __summary_outputs

        self.fit = self._fit_unweighted

    def _loop_train(self, data, batch_size):
        nb_words = data[0].shape[0]
        # loss, nrm, cnrm = 0.0, 0.0, 0.0
        loss = 0.0
        enc_len = 0.0
        batches = make_batches(nb_words, batch_size)
        for start, end in batches:
            ins_batch = slice_X(data, start_=start, end_=end, axis=0)
            loss_, enc_len_, _ = self._train(*ins_batch)
            loss += loss_ * ins_batch[1].size
            enc_len += enc_len_

        loss /= nb_words
        ppl = math.exp(enc_len/nb_words)
        # return loss, nrm, cnrm
        return loss, ppl

    def prepare_input(self, sents):
        ns = sents.shape[0]
        nt = sents.shape[1]
        nb_ele = sents.size  # NO. of words in the sentences.

        pad_idx = np.arange(self.vocab_size, self.vocab_size+self.context_size).reshape((1, -1))
        pad_idx = pad_idx.repeat(ns, axis=0)  # (ns, c), where c is context size
        idxes = np.hstack((pad_idx, sents))   # (ns, c+s), where s is sentence length

        X = np.empty(shape=(nb_ele, self.context_size), dtype='int32')
        y_label = np.empty(shape=(nb_ele, ), dtype='int32')
        start_end = np.array([0, 0], dtype='int32')
        k = 0
        for i in range(ns):  # loop on sentences
            start_end[0], start_end[1] = 0, self.context_size
            for _ in range(nt):  # loop on time (each time step corresponds to a word)
                X[k] = idxes[i, start_end[0]:start_end[1]]
                y_label[k] = idxes[i, start_end[1]]
                k += 1
                start_end += 1
        return X, y_label

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            x = self.prepare_input(chunk)
            # loss, nrm, cnrm = self._loop_train(x, batch_size)
            loss, ppl = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.3f - ppl: %6.2f speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, ppl, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                logger.info('Training finished. Evaluating ...')
                log_file.info('Training finished. Evaluating ...')
                self.validation(val_sents, batch_size, log_file)
                if save_path is not None:
                    self.save_params(save_path)
                break
        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = self.prepare_input(sents)
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        # try:
        ppl = math.exp(code_len/nb_words)
        # except OverflowError:
        #     logger.error("code_len: %.3f - nb_words: %d" % (code_len, nb_words))
        #     ppl = self.vocab_size * 1000
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    @staticmethod
    def _test_loop(f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[0]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            ins_batch = slice_X(ins, start_=batch_start, end_=batch_end, axis=0)
            batch_outs = f(*ins_batch)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(batch_end - batch_start)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class LBLangModelV2(Graph, LangModel):
    # the standard LBL language model with sparse coding extension
    def __init__(self, sparse_coding, context_size, nb_negative, embed_dims=200, init_embeddings=None,
                 negprob_table=None, optimizer='adam'):
        super(LBLangModelV2, self).__init__()
        self.nb_negative = nb_negative
        self.sparse_coding = sparse_coding
        vocab_size = sparse_coding.shape[0]  # the extra word is for OOV
        self.nb_base = sparse_coding.shape[1] - 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        # self.loss = categorical_crossentropy
        # self.loss_fnc = objective_fnc(self.loss)
        self.optimizer = optimizers.get(optimizer)
        self.context_size = context_size
        self.weights = None
        # self.max_sent_len = max_sent_len
        tmp1 = sparse.csr_matrix((self.vocab_size, context_size), dtype=floatX)
        tmp2 = sparse.csr_matrix((context_size, self.nb_base+1), dtype=floatX)
        tmp3 = sparse.vstack([tmp1, sparse.csr_matrix(np.eye(context_size, dtype=floatX))])
        tmp4 = sparse.vstack([self.sparse_coding, tmp2])
        self.sparse_coding_pad = sparse.hstack([tmp4, tmp3], format='csr')

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            # negprob_table = theano.shared(negprob_table_)
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)
            # negprob_table = theano.shared(negprob_table.astype(theano.config.floatX))

        # self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='ngrams', ndim=2, dtype='int32')          # (ns, c), where c is the context size
        self.add_input(name='label_with_neg', ndim=2, dtype='int32')  # (k+1, ns)
        self.add_input(name='lookup_prob', ndim=2, dtype=floatX)      # (k+1, ns)

        cntx_codes = tsp.csr_matrix('cntx-codes', dtype=floatX)
        label_codes = tsp.csr_matrix('label_codes', dtype=floatX)
        cntx_idxes = self.inputs['ngrams'].get_output()
        # label_idxes = self.inputs['label_with_neg'].get_output()
        batch_shape = cntx_idxes.shape

        self.add_node(Identity(inputs={True: cntx_codes, False: cntx_codes}), name='cntx_codes_flat')
        self.add_node(Identity(inputs={True: label_codes, False: label_codes}), name='label_codes_flat')
        self.add_node(Identity(inputs={True: batch_shape, False: batch_shape}), name='cntx_shape')
        # self.add_node(Identity(inputs={True: codes, False: codes}), name='sparse_codes')

        self.add_node(SparseEmbedding(self.nb_base+self.context_size+1, embed_dims, weights=init_embeddings),
                      name='embedding', inputs=('cntx_codes_flat', 'cntx_shape'))
        self.add_node(EmbeddingParam(), name='embedding_param', inputs='embedding')
        self.add_node(Reshape(-1), name='reshape', inputs='embedding')
        composer_node = Dense(context_size*embed_dims, embed_dims)
        composer_node.params = [composer_node.W]   # drop the bias parameters
        composer_node.get_output = lambda train: node_get_output(composer_node, train)
        self.add_node(composer_node, name='context_vec', inputs='reshape')
        self.add_node(PartialSoftmaxLBL(base_size=self.nb_base+1,
                                        word_vecs=self.nodes['embedding'].W),
                      name='part_prob', inputs=('label_with_neg', 'label_codes_flat', 'context_vec'))

        # self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='label_with_neg')
        self.add_node(SharedWeightsDense(self.nodes['part_prob'].W,
                                         self.nodes['part_prob'].b,
                                         self.sparse_coding, activation='softmax'),
                      name='true_prob', inputs='context_vec')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')

        def node_get_output(layer, train=False):
            X = layer.get_input(train)
            output = T.dot(X, layer.W)  # there is no activation.
            return output

        self.fit = None

        self.jobs_pools = None
        self.jobs_pools_post = None
        self.in_training_phase = Event()
        self.trn_finished = Event()
        self.all_finished = MEvent()

    def __del__(self):
        self.trn_finished.set()
        self.all_finished.set()


    @staticmethod
    def encode_length(y_label, y_pred, mask=None):
        """
        :param y_label: true index labels with shape (ns, )
        :param y_pred: predicted probabilities with shape (ns, V)
        :param mask: mask
        :return: PPL
        """
        ## there is no need to clip here, for the prob. have already clipped by LBLayer
        # epsilon = 1e-7
        # y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
        # # scale preds so that the class probas of each sample sum to 1
        # y_pred /= y_pred.sum(axis=-1, keepdims=True)

        nb_samples = y_label.shape[0]
        idx = T.arange(nb_samples)
        probs_ = y_pred[idx, y_label]

        return -T.sum(T.log(probs_)), nb_samples

    # noinspection PyMethodOverriding
    def compile(self, optimizer=None):
        # from theano.compile.nanguardmode import NanGuardMode
        if optimizer is not None:
            logger.info('compiling with %s' % optimizer)
            self.optimizer = optimizers.get(optimizer)
        # input of model
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        # normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        pre_prob_tst = T.clip(pre_prob_tst, epsilon, 1.-epsilon)
        pre_prob_tst = pre_prob_tst / T.sum(pre_prob_tst, axis=-1, keepdims=True)

        # nrm_const = normlzer_layer.get_output(train=True)
        # nrm_const = T.reshape(nrm_const, (nrm_const.shape[0], nrm_const.shape[1]))
        # nrm_const = nrm_const.dimshuffle('x', 0, 1)
        # pos_prob_trn *= nrm_const

        # nrm_const_tst = normlzer_layer.get_output(train=False)
        # nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], nrm_const_tst.shape[1]))
        # nrm_const_tst = nrm_const_tst.dimshuffle('x', 0, 1)
        # pos_prob_tst *= nrm_const_tst

        #TODO: mask not supported here
        # eps = 1.0e-10
        nb_words = pos_prob_trn[0].size.astype(floatX)
        nb_words_ = pos_prob_tst[0].size.astype(floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(T.clip(pos_prob_trn[0]/sum_pos_neg_trn[0], epsilon, 1.-epsilon))) / nb_words
        y_train += T.sum(T.log(T.clip(neg_prob_trn[1:]/sum_pos_neg_trn[1:], epsilon, 1.-epsilon))) / nb_words
        y_test = T.sum(T.log(T.clip(pos_prob_tst[0]/sum_pos_neg_tst[0], epsilon, 1.-epsilon))) / nb_words_
        y_test += T.sum(T.log(T.clip(neg_prob_tst[1:] / sum_pos_neg_tst[1:], epsilon, 1.-epsilon))) / nb_words_

        input0 = self.inputs['ngrams'].get_output(True)
        input1 = self.inputs['label_with_neg'].get_output(True)
        input2 = self.nodes['cntx_codes_flat'].get_output(True)
        input3 = self.nodes['label_codes_flat'].get_output(True)
        input4 = self.inputs['lookup_prob'].get_output(True)

        true_labels = input1[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        train_ins = [input0, input1, input2, input3, input4]
        test_ins = [input0, input1, input2, input3, input4]

        self._train = theano.function(train_ins, train_loss, updates=updates)
        self._train.out_labels = ['loss', 'encode_len', 'nb_words']
        self._test = theano.function(test_ins, [test_loss, encode_len, nb_words])
        self._test.out_labels = ['loss', 'encode_len', 'nb_words']

        self.all_metrics = ['loss', 'ppl', 'val_loss', 'val_ppl']

        def __summary_outputs(outs, batch_sizes):
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._train.summarize_outputs = __summary_outputs
        self._test.summarize_outputs = __summary_outputs

        self.fit = self._fit_unweighted

    # @profile
    # def get_sp_codes(self, idxes, pad=True):
    #     if pad:
    #         sp_reprs = self.sparse_coding_pad[idxes.ravel()]
    #     else:
    #         sp_reprs = self.sparse_coding[idxes.ravel()]
    #     return sp_reprs

    # def _loop_train(self, data, batch_size):
    #     nb_words = data[0].shape[0]
    #     loss = 0.0
    #     batches = make_batches(nb_words, batch_size)
    #     for start, end in batches:
    #         X = data[0][start:end]
    #         y = data[1][:, start:end]
    #         sp_x = self.get_sp_codes(X, pad=True)
    #         sp_y = self.get_sp_codes(y, pad=False)
    #         loss_ = self._train(X, y, sp_x, sp_y)
    #         loss += loss_ * y[0].size
    #
    #     loss /= nb_words
    #     return loss

    # def _loop_train(self):
    #     X, y, sp_x, sp_y = self.jobs_pools_post.get()
    #     loss = self._train(X, y, sp_x, sp_y)
    #     return loss

    # def negative_sample(self, y):
    #     ret = np.empty(shape=(self.nb_negative+1,) + y.shape, dtype=y.dtype)
    #     ret[0] = y
    #     ret[1:] = self.sampler.sample(shape=ret[1:].shape)
    #
    #     return ret

    # @profile
    # def pad_sparse_repr(self, idx):
    #     pad = np.zeros((1, self.context_size), dtype=floatX)
    #     if idx >= self.vocab_size:
    #         pad[0, idx-self.vocab_size] = 1.0
    #         ret = sp_hstack([csr_matrix((1, self.sparse_coding.shape[1]), dtype=floatX), pad], format='csr')
    #     else:
    #         ret = sp_hstack([self.sparse_coding[idx], csr_matrix(pad)], format='csr')
    #
    #     return ret

    # #@profile
    # def prepare_input(self, sents_queue, jobs_pools_post, batch_size):
    #     sents = sents_queue.get()
    #     ns = sents.shape[0]
    #     nt = sents.shape[1]
    #     nb_ele = sents.size  # NO. of words in the sentences.
    #
    #     pad_idx = np.arange(self.vocab_size, self.vocab_size+self.context_size).reshape((1, -1))
    #     pad_idx = pad_idx.repeat(ns, axis=0)  # (ns, c), where c is context size
    #     idxes = np.hstack((pad_idx, sents))   # (ns, c+s), where s is sentence length
    #
    #     X = np.empty(shape=(nb_ele, self.context_size), dtype='int32')
    #     y_label = np.empty(shape=(nb_ele, ), dtype='int32')
    #     start_end = np.array([0, 0], dtype='int32')
    #     k = 0
    #     for i in range(ns):  # loop on sentences
    #         start_end[0], start_end[1] = 0, self.context_size
    #         for _ in range(nt):  # loop on time (each time step corresponds to a word)
    #             X[k] = idxes[i, start_end[0]:start_end[1]]
    #             y_label[k] = idxes[i, start_end[1]]
    #             k += 1
    #             start_end += 1
    #
    #     y_label = self.negative_sample(y_label)
    #
    #     nb_sample = X.shape[0]
    #     batches = make_batches(nb_sample, batch_size)
    #     X_, y_, sp_x_, sp_y_ = (None, ) * 4
    #     for batch_index, (batch_start, batch_end) in enumerate(batches):
    #         X_ = X[batch_start:batch_end]
    #         y_ = y_label[:, batch_start:batch_end]
    #         sp_x_ = self.get_sp_codes(X_.ravel(), pad=True)
    #         sp_y_ = self.get_sp_codes(y_label.ravel(), pad=False)
    #
    #     if X_ and X_.shape[0] > 0:
    #         self.jobs_pools_post.put((X_, y_, sp_x_, sp_y_))

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None, nb_data_workers=6, data_pool_size=10):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        data_workers = []
        pre_data = Queue(data_pool_size)
        post_data = Queue(data_pool_size*30)
        self.jobs_pools = pre_data
        self.jobs_pools_post = post_data

        xk = Array(ctypes.c_int32, np.arange(self.vocab_size, dtype='int32'), lock=False)
        # a_type = ctypes.c_double if str(self.neg_prob_table.dtype) == 'float64' else ctypes.c_float
        assert str(self.neg_prob_table.dtype) == 'float32'
        pk = Array(ctypes.c_float, self.neg_prob_table, lock=False)

        # a_type = ctypes.c_double if str(self.sparse_coding.dtype) == 'float64' else ctypes.c_float
        assert str(self.sparse_coding.dtype) == 'float32'
        sp_data = Array(ctypes.c_float, self.sparse_coding.data, lock=False)
        assert str(self.sparse_coding.indices.dtype) == 'int32'
        assert str(self.sparse_coding.indptr.dtype) == 'int32'
        sp_indices = Array(ctypes.c_int32, self.sparse_coding.indices, lock=False)
        sp_indptr = Array(ctypes.c_int32, self.sparse_coding.indptr, lock=False)

        # a_type = ctypes.c_double if str(self.sparse_coding_pad.dtype) == 'float64' else ctypes.c_float
        assert str(self.sparse_coding_pad.dtype) == 'float32'
        sp_pad_data = Array(ctypes.c_float, self.sparse_coding_pad.data, lock=False)
        assert str(self.sparse_coding_pad.indices.dtype) == 'int32'
        assert str(self.sparse_coding_pad.indptr.dtype) == 'int32'
        sp_pad_indices = Array(ctypes.c_int32, self.sparse_coding_pad.indices, lock=False)
        sp_pad_indptr = Array(ctypes.c_int32, self.sparse_coding_pad.indptr, lock=False)

        for _ in range(nb_data_workers):
            # prepare_input(sents_queue, jobs_pool, all_finished,
            #       vocab_size, context_size, batch_size, nb_negative, xk, pk,
            #       sp_data, sp_indices, sp_indptr, sp_shape,
            #       sp_pad_data, sp_pad_indices, sp_pad_inptr, sp_pad_shape):
            p = Process(target=prepare_input, args=(pre_data, post_data, self.all_finished,
                                                    self.vocab_size, self.context_size, batch_size, self.nb_negative,
                                                    xk, pk, sp_data, sp_indices, sp_indptr, self.sparse_coding.shape,
                                                    sp_pad_data, sp_pad_indices, sp_pad_indptr, self.sparse_coding_pad.shape))
            p.daemon = True
            data_workers.append(p)
            p.start()

        self.validation(train_val_sents, log_file)

        def chunk_trn_generator():
            for sents in sent_gen:
                if self.trn_finished.is_set():
                    break

                mask = (sents > max_vocab)
                sents[mask] = max_vocab
                chunk = chunk_sentences(sentences, sents, batch_size)
                if chunk is None:
                    continue
                self.in_training_phase.wait()
                pre_data.put(chunk)

            self.trn_finished.set()
            logger.debug('trn data finished')

        gen_chunk_thread = Thread(target=chunk_trn_generator)
        gen_chunk_thread.setDaemon(True)
        gen_chunk_thread.start()

        loss = 0.0
        nb_chunk = 0.0
        nb_cyc = 0
        start_ = time()
        next_val_time = start_ + validation_interval
        self.in_training_phase.set()

        while not self.trn_finished.is_set() or not post_data.empty():
            ins = post_data.get()
            # if ins is None:
            #     if nb_none == nb_data_workers:
            #         break
            #     else:
            #         nb_none += 1
            #         continue
            loss_ = self._train(*ins)
            nb_cyc += 1
            nb_cyc %= 20
            nb_words_trained += ins[0].shape[0]
            nb_chunk += ins[0].shape[0]
            loss += loss_ * ins[0].shape[0]
            if nb_cyc == 0:
                end_ = time()
                elapsed = float(end_ - start_)
                speed2 = nb_words_trained/elapsed
                eta = (train_nb_words - nb_words_trained) / speed2
                eta_h = int(math.floor(eta/3600))
                eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
                loss /= nb_chunk
                logger.info('%s:Train - ETA: %02d:%02d - loss: %5.3f - speed: %.1f words/s' %
                            (self.__class__.__name__, eta_h, eta_m, loss, speed2))
                log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))
                nb_chunk = 0.0
                loss = 0.0

                if end_ > next_val_time:
                    logger.debug('pausing training data generation and consuming all generated data')
                    self.in_training_phase.clear()
                    while not self.jobs_pools_post.empty() or not self.jobs_pools.empty():
                        ins = self.jobs_pools_post.get()
                        self._train(*ins)
                    logger.debug('Before validation')
                    # noinspection PyUnresolvedReferences
                    self.validation(train_val_sents, log_file)
                    logger.debug('END validation. resume training data generation')
                    self.in_training_phase.set()
                    next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                self.trn_finished.set()
                break

        # consume all the produced tasks. The data generation thread will automatically shutdown, for the trn_finished
        # event is set.
        self.in_training_phase.set()  # make sure it is not blocking
        while not self.jobs_pools_post.empty() or not self.jobs_pools.empty():
            ins = self.jobs_pools_post.get()
            self._train(*ins)

        # Now the training data is consumed out. Let's evaluate...
        logger.info('Training finished. Evaluating ...')
        log_file.info('Training finished. Evaluating ...')
        self.validation(val_sents, log_file)
        self.all_finished.set()  # signal the all_finished event to shutdown all worker processes.
        if save_path is not None:
            self.save_params(save_path)
        log_file.close()

        for _ in range(10):
            flags = map(Process.is_alive, data_workers)
            if not any(flags):
                break
            for flag, p in zip(flags, data_workers):
                if flag is True:
                    logger.info("%s is alive" % p.name)
            sleep(5)

    def validation(self, val_sents, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        def chunk_val_generator():
            for sents in val_sents:
                self.jobs_pools.put(sents)

        gen_chunk_thread = Thread(target=chunk_val_generator)
        gen_chunk_thread.setDaemon(True)
        gen_chunk_thread.start()

        logger.debug('begin val loop')
        while True:
            ins = self.jobs_pools_post.get()
            loss_, code_len_, nb_words_ = self._test(*ins)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_
            if self.jobs_pools_post.empty() and self.jobs_pools.empty():
                break
        logger.debug('end val loop')

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)

        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    # def _test_loop(self, f, ins, batch_size=128, verbose=0):
    #     nb_sample = ins[0].shape[0]
    #     outs = [[] for _ in range(f.n_returned_outputs)]
    #     batch_info = []
    #     batches = make_batches(nb_sample, batch_size)
    #     for batch_index, (batch_start, batch_end) in enumerate(batches):
    #         X = ins[0][batch_start:batch_end]
    #         y = ins[1][:, batch_start:batch_end]
    #         sp_x = ins[2][batch_start*self.context_size:batch_end*self.context_size]
    #         sp_y = ins[3][batch_start*self.context_size:batch_end*self.context_size]
    #         # sp_x = self.get_sp_codes(X, pad=True)
    #         # sp_y = self.get_sp_codes(y, pad=False)
    #         batch_outs = f(X, y, sp_x, sp_y)
    #         for idx, v in enumerate(batch_outs):
    #             outs[idx].append(v)
    #         batch_info.append(batch_end - batch_start)
    #
    #     outs = f.summarize_outputs(outs, batch_info)
    #     return outs

    # #@profile
    # def _loop_train1(self, data, batch_size):
    #     nb_words = data[0].shape[0]
    #     loss = 0.0
    #     batches = make_batches(nb_words, batch_size)
    #     st = time()
    #     for k, (start, end) in enumerate(batches):
    #         X = data[0][start:end]
    #         y = data[1][:, start:end]
    #         sp_x = self.get_sp_codes(X, pad=True)
    #         sp_y = self.get_sp_codes(y, pad=False)
    #         print "BEF %d: %.5f" % (k, time()-st)
    #         loss_ = self.__train(X, y, sp_x, sp_y)
    #         print "AFT %d: %.5f" % (k, time()-st)
    #         loss += loss_ * y[0].size
    #
    #     loss /= nb_words
    #     return loss
    #
    # #@profile
    # def __train(self, X, y, sp_x, sp_y):
    #     return self._train(X, y, sp_x, sp_y)
    #
    # #@profile
    # def profile_model(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2',
    #                   batch_size=256, max_group=5):
    #     sent_gen = grouped_sentences(data_file)
    #     max_vocab = self.vocab_size - 1
    #     st = time()
    #     for k, sents in enumerate(sent_gen):
    #         mask = (sents > max_vocab)
    #         sents[mask] = max_vocab
    #         x = self.prepare_input(sents)
    #         self._loop_train1(x, batch_size)
    #
    #         if k == max_group:
    #             break


class LBLangModelV3(Graph, LangModel):
    # the standard LBL language model with sparse coding extension, ZRegression
    def __init__(self, sparse_coding, context_size, nb_negative, embed_dims=200, max_part_sum=0.7, alpha=1.0, beta=1.,
                 init_embeddings=None, negprob_table=None, optimizer='adam'):
        super(LBLangModelV3, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.max_part_sum = max_part_sum
        self.nb_negative = nb_negative
        self.sparse_coding = sparse_coding
        vocab_size = sparse_coding.shape[0]  # the extra word is for OOV
        self.nb_base = sparse_coding.shape[1] - 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        # self.loss = categorical_crossentropy
        # self.loss_fnc = objective_fnc(self.loss)
        self.optimizer = optimizers.get(optimizer)
        self.context_size = context_size
        self.weights = None
        # self.max_sent_len = max_sent_len
        tmp1 = sparse.csr_matrix((self.vocab_size, context_size), dtype=floatX)
        tmp2 = sparse.csr_matrix((context_size, self.nb_base+1), dtype=floatX)
        tmp3 = sparse.vstack([tmp1, sparse.csr_matrix(np.eye(context_size, dtype=floatX))])
        tmp4 = sparse.vstack([self.sparse_coding, tmp2])
        self.sparse_coding_pad = sparse.hstack([tmp4, tmp3], format='csr')

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            self.neg_prob_table = negprob_table_
        else:
            nrm = np.sum(negprob_table)
            if nrm != 1.0:
                logger.warn('Sampling table not normalized! sum to: %.6f' % nrm)
                negprob_table /= nrm
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)

        # self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='ngrams', ndim=2, dtype='int32')          # (ns, c), where c is the context size
        self.add_input(name='label_with_neg', ndim=2, dtype='int32')  # (k+1, ns)
        self.add_input(name='lookup_prob', ndim=2, dtype=floatX)      # (k+1, ns)

        cntx_codes = tsp.csr_matrix('cntx-codes', dtype=floatX)
        label_codes = tsp.csr_matrix('label_codes', dtype=floatX)
        cntx_idxes = self.inputs['ngrams'].get_output()
        batch_shape = cntx_idxes.shape

        self.add_node(Identity(inputs={True: cntx_codes, False: cntx_codes}), name='cntx_codes_flat')
        self.add_node(Identity(inputs={True: label_codes, False: label_codes}), name='label_codes_flat')
        self.add_node(Identity(inputs={True: batch_shape, False: batch_shape}), name='cntx_shape')
        # self.add_node(Identity(inputs={True: codes, False: codes}), name='sparse_codes')

        self.add_node(SparseEmbedding(self.nb_base+self.context_size+1, embed_dims, weights=init_embeddings),
                      name='embedding', inputs=('cntx_codes_flat', 'cntx_shape'))
        self.add_node(EmbeddingParam(), name='embedding_param', inputs='embedding')
        self.add_node(Reshape(-1), name='reshape', inputs='embedding')
        composer_node = Dense(context_size*embed_dims, embed_dims)
        composer_node.params = [composer_node.W]   # drop the bias parameters
        composer_node.get_output = lambda train: node_get_output(composer_node, train)
        self.add_node(composer_node, name='context_vec', inputs='reshape')
        self.add_node(PartialSoftmaxLBL(base_size=self.nb_base+1,
                                        word_vecs=self.nodes['embedding'].W),
                      name='part_prob', inputs=('label_with_neg', 'label_codes_flat', 'context_vec'))
        self.add_node(Dense(input_dim=embed_dims, output_dim=embed_dims, activation='sigmoid'),
                      name='normalizer0', inputs='context_vec')
        self.add_node(Dense(input_dim=embed_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='normalizer0')
        self.add_node(SharedWeightsDense(self.nodes['part_prob'].W,
                                         self.nodes['part_prob'].b,
                                         self.sparse_coding, activation='softmax'),
                      name='true_prob', inputs='context_vec')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

        def node_get_output(layer, train=False):
            X = layer.get_input(train)
            output = T.dot(X, layer.W)  # there is no activation.
            return output

        self.fit = None

        self.jobs_pools = None
        self.jobs_pools_post = None
        self.in_training_phase = Event()
        self.trn_finished = Event()
        self.all_finished = MEvent()

    def __del__(self):
        self.trn_finished.set()
        self.all_finished.set()


    @staticmethod
    def encode_length(y_label, y_pred, mask=None):
        """
        :param y_label: true index labels with shape (ns, )
        :param y_pred: predicted probabilities with shape (ns, V)
        :param mask: mask
        :return: PPL
        """
        ## there is no need to clip here, for the prob. have already clipped by LBLayer
        # epsilon = 1e-7
        # y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
        # # scale preds so that the class probas of each sample sum to 1
        # y_pred /= y_pred.sum(axis=-1, keepdims=True)

        nb_samples = y_label.shape[0]
        idx = T.arange(nb_samples)
        probs_ = y_pred[idx, y_label]

        return -T.sum(T.log(probs_)), nb_samples

    # noinspection PyMethodOverriding
    def compile(self, optimizer=None):
        # from theano.compile.nanguardmode import NanGuardMode
        if optimizer is not None:
            logger.info('compiling with %s' % optimizer)
            self.optimizer = optimizers.get(optimizer)
        # output of model
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)     # (k+1, ns)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)        # (ns, 1)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0],))  # (ns, )
        nrm_const = nrm_const.dimshuffle('x', 0)                 # (1, ns)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], ))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0)
        pos_prob_tst *= nrm_const_tst

        pre_prob_tst = T.clip(pre_prob_tst, epsilon, 1.-epsilon)
        pre_prob_tst = pre_prob_tst / T.sum(pre_prob_tst, axis=-1, keepdims=True)

        #TODO: mask not supported here

        nb_words = pos_prob_trn[0].size.astype(floatX)
        nb_words_ = pos_prob_tst[0].size.astype(floatX)

        max_part_sum = self.max_part_sum
        part_sum = T.sum(pos_prob_trn, axis=0)
        tmp = T.switch(part_sum > max_part_sum, part_sum-max_part_sum, 0.0)
        not_prob_loss = T.sum(tmp)/(T.nonzero(tmp)[0].size+1.0)

        part_sum_tst = T.sum(pos_prob_tst, axis=0)
        tmp = T.switch(part_sum_tst > max_part_sum, part_sum_tst-max_part_sum, 0.0)
        not_prob_loss_tst = T.sum(tmp)/(T.nonzero(tmp)[0].size+1.0)

        delta = 0.001
        dif_pos_neg = pos_prob_trn[0] - T.sum(pos_prob_trn[1:], axis=0)/float(self.nb_negative)
        dif_gain = dif_pos_neg - delta
        dif_loss = T.sum(T.switch(dif_gain > 0.0, 0.0, -dif_gain))/nb_words

        # sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        # sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        # y_train = T.sum(T.log(T.clip(pos_prob_trn[0]/sum_pos_neg_trn[0], epsilon, 1.-epsilon))) / nb_words
        # y_train += T.sum(T.log(T.clip(neg_prob_trn[1:]/sum_pos_neg_trn[1:], epsilon, 1.-epsilon))) / nb_words
        # y_test = T.sum(T.log(T.clip(pos_prob_tst[0]/sum_pos_neg_tst[0], epsilon, 1.-epsilon))) / nb_words_
        # y_test += T.sum(T.log(T.clip(neg_prob_tst[1:] / sum_pos_neg_tst[1:], epsilon, 1.-epsilon))) / nb_words_

        eps = 1.0e-10
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn + 2*eps
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst + 2*eps
        y_train = T.sum(T.log((pos_prob_trn[0]+eps)/sum_pos_neg_trn[0]))
        y_train += T.sum(T.log((neg_prob_trn[1:]+eps)/sum_pos_neg_trn[1:]))
        y_train /= nb_words
        y_test = T.sum(T.log((pos_prob_tst[0]+eps)/sum_pos_neg_tst[0]))
        y_test += T.sum(T.log((neg_prob_tst[1:]+eps) / sum_pos_neg_tst[1:]))
        y_test /= nb_words_

        input0 = self.inputs['ngrams'].get_output(True)
        input1 = self.inputs['label_with_neg'].get_output(True)
        input2 = self.nodes['cntx_codes_flat'].get_output(True)
        input3 = self.nodes['label_codes_flat'].get_output(True)
        input4 = self.inputs['lookup_prob'].get_output(True)

        true_labels = input1[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train + self.alpha*not_prob_loss + self.beta * dif_loss
        test_loss = -y_test + self.alpha*not_prob_loss_tst
        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        train_ins = [input0, input1, input2, input3, input4]
        test_ins = [input0, input1, input2, input3, input4]

        self._train = theano.function(train_ins, [train_loss, T.max(part_sum)], updates=updates)
        self._train.out_labels = ['loss']
        self._test = theano.function(test_ins, [test_loss, encode_len, nb_words])
        self._test.out_labels = ['loss', 'encode_len', 'nb_words']

        self.all_metrics = ['loss', 'ppl', 'val_loss', 'val_ppl']

        def __summary_outputs(outs, batch_sizes):
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._train.summarize_outputs = __summary_outputs
        self._test.summarize_outputs = __summary_outputs

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None, nb_data_workers=6, data_pool_size=10):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        data_workers = []
        pre_data = Queue(data_pool_size)
        post_data = Queue(data_pool_size*30)
        self.jobs_pools = pre_data
        self.jobs_pools_post = post_data

        xk = Array(ctypes.c_int32, np.arange(self.vocab_size, dtype='int32'), lock=False)
        # a_type = ctypes.c_double if str(self.neg_prob_table.dtype) == 'float64' else ctypes.c_float
        assert str(self.neg_prob_table.dtype) == 'float32'
        pk = Array(ctypes.c_float, self.neg_prob_table, lock=False)

        # a_type = ctypes.c_double if str(self.sparse_coding.dtype) == 'float64' else ctypes.c_float
        assert str(self.sparse_coding.dtype) == 'float32'
        sp_data = Array(ctypes.c_float, self.sparse_coding.data, lock=False)
        assert str(self.sparse_coding.indices.dtype) == 'int32'
        assert str(self.sparse_coding.indptr.dtype) == 'int32'
        sp_indices = Array(ctypes.c_int32, self.sparse_coding.indices, lock=False)
        sp_indptr = Array(ctypes.c_int32, self.sparse_coding.indptr, lock=False)

        # a_type = ctypes.c_double if str(self.sparse_coding_pad.dtype) == 'float64' else ctypes.c_float
        assert str(self.sparse_coding_pad.dtype) == 'float32'
        sp_pad_data = Array(ctypes.c_float, self.sparse_coding_pad.data, lock=False)
        assert str(self.sparse_coding_pad.indices.dtype) == 'int32'
        assert str(self.sparse_coding_pad.indptr.dtype) == 'int32'
        sp_pad_indices = Array(ctypes.c_int32, self.sparse_coding_pad.indices, lock=False)
        sp_pad_indptr = Array(ctypes.c_int32, self.sparse_coding_pad.indptr, lock=False)

        for _ in range(nb_data_workers):
            # prepare_input(sents_queue, jobs_pool, all_finished,
            #       vocab_size, context_size, batch_size, nb_negative, xk, pk,
            #       sp_data, sp_indices, sp_indptr, sp_shape,
            #       sp_pad_data, sp_pad_indices, sp_pad_inptr, sp_pad_shape):
            p = Process(target=prepare_input, args=(pre_data, post_data, self.all_finished,
                                                    self.vocab_size, self.context_size, batch_size, self.nb_negative,
                                                    xk, pk, sp_data, sp_indices, sp_indptr, self.sparse_coding.shape,
                                                    sp_pad_data, sp_pad_indices, sp_pad_indptr, self.sparse_coding_pad.shape))
            p.daemon = True
            data_workers.append(p)
            p.start()

        self.in_training_phase.clear()
        self.validation(train_val_sents, log_file)

        def chunk_trn_generator():
            for sents in sent_gen:
                if self.trn_finished.is_set():
                    break

                mask = (sents > max_vocab)
                sents[mask] = max_vocab
                chunk = chunk_sentences(sentences, sents, batch_size)
                if chunk is None:
                    continue
                self.in_training_phase.wait()
                pre_data.put(chunk)

            self.trn_finished.set()
            logger.debug('trn data finished')

        gen_chunk_thread = Thread(target=chunk_trn_generator)
        gen_chunk_thread.setDaemon(True)
        gen_chunk_thread.start()

        loss = 0.0
        nb_chunk = 0.0
        nb_cyc = 0
        part_sum = 0.0
        start_ = time()
        next_val_time = start_ + validation_interval
        next_report = start_ + 1.0
        self.in_training_phase.set()

        while not self.trn_finished.is_set() or not post_data.empty():
            ins = post_data.get()
            # if ins is None:
            #     if nb_none == nb_data_workers:
            #         break
            #     else:
            #         nb_none += 1
            #         continue
            loss_, part_sum_ = self._train(*ins)
            nb_cyc += 1
            nb_cyc %= 20
            nb_words_trained += ins[0].shape[0]
            nb_chunk += ins[0].shape[0]
            loss += loss_ * ins[0].shape[0]
            part_sum = max(part_sum_, part_sum)
            if nb_cyc == 0:
                end_ = time()
                if end_ >= next_report:
                    elapsed = float(end_ - start_)
                    speed2 = nb_words_trained/elapsed
                    eta = (train_nb_words - nb_words_trained) / speed2
                    eta_h = int(math.floor(eta/3600))
                    eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
                    loss /= nb_chunk
                    logger.info('%s:Train - ETA: %02d:%02d - loss: %5.3f - part_sum: %.2f - speed: %.1f words/s' %
                                (self.__class__.__name__, eta_h, eta_m, loss, part_sum, speed2))
                    log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))
                    nb_chunk = 0.0
                    loss = 0.0
                    part_sum = 0.0
                    next_report = end_ + 1.0

                if end_ > next_val_time:
                    logger.debug('pausing training data generation and consuming all generated data')
                    self.in_training_phase.clear()
                    while not self.jobs_pools_post.empty() or not self.jobs_pools.empty():
                        ins = self.jobs_pools_post.get()
                        nb_words_trained += ins[0].shape[0]
                        self._train(*ins)
                    logger.debug('Before validation')
                    # noinspection PyUnresolvedReferences
                    self.validation(train_val_sents, log_file)
                    logger.debug('END validation. resume training data generation')
                    self.in_training_phase.set()
                    next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                self.trn_finished.set()
                break

        # consume all the produced tasks. The data generation thread will automatically shutdown, for the trn_finished
        # event is set.
        self.in_training_phase.set()  # make sure it is not blocking
        while not self.jobs_pools_post.empty() or not self.jobs_pools.empty():
            ins = self.jobs_pools_post.get()
            self._train(*ins)

        # Now the training data is consumed out. Let's evaluate...
        logger.info('Training finished. Evaluating ...')
        log_file.info('Training finished. Evaluating ...')
        self.validation(val_sents, log_file)
        self.all_finished.set()  # signal the all_finished event to shutdown all worker processes.
        if save_path is not None:
            self.save_params(save_path)
        log_file.close()

        for _ in range(10):
            flags = map(Process.is_alive, data_workers)
            if not any(flags):
                break
            for flag, p in zip(flags, data_workers):
                if flag is True:
                    logger.info("%s is alive" % p.name)
            sleep(5)

    def validation(self, val_sents, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        def chunk_val_generator():
            for sents in val_sents:
                self.jobs_pools.put(sents)

        gen_chunk_thread = Thread(target=chunk_val_generator)
        gen_chunk_thread.setDaemon(True)
        gen_chunk_thread.start()

        logger.debug('begin val loop')
        while True:
            ins = self.jobs_pools_post.get()
            loss_, code_len_, nb_words_ = self._test(*ins)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_
            if self.jobs_pools_post.empty() and self.jobs_pools.empty():
                break
        logger.debug('end val loop')

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)

        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl


class LBLangModelV4(Graph, LangModel):
    # the standard LBL language model with NCE
    def __init__(self, vocab_size, context_size, embed_dims=128, nb_negative=50, negprob_table=None, optimizer='adam'):
        super(LBLangModelV4, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.nb_negative = nb_negative
        # self.loss = categorical_crossentropy
        # self.loss_fnc = objective_fnc(self.loss)
        self.optimizer = optimizers.get(optimizer)
        self.context_size = context_size
        self.weights = None

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)

        self.sampler = TableSampler(self.neg_prob_table)
        # self.max_sent_len = max_sent_len

        self.add_input(name='ngrams', ndim=2, dtype='int32')          # (ns, c)
        self.add_input(name='label_with_neg', ndim=2, dtype='int32')  # (k+1, ns)
        self.add_input(name='lookup_prob', ndim=2, dtype=floatX)      # (k+1, ns)

        self.add_node(Embedding(vocab_size+context_size, embed_dims), name='embedding', inputs='ngrams')
        self.add_node(EmbeddingParam(), name='embedding_param', inputs='embedding')
        self.add_node(Reshape(-1), name='reshape', inputs='embedding')
        composer_node = Dense(context_size*embed_dims, embed_dims)
        composer_node.params = [composer_node.W]   # drop the bias parameters
        # replace the default behavior of Dense
        composer_node.get_output = lambda train: node_get_output(composer_node, train)
        self.add_node(composer_node, name='context_vec', inputs='reshape')
        self.add_node(PartialSoftmaxLBLV4(embed_dims, self.vocab_size,
                                          word_vecs=self.nodes['embedding'].W[:vocab_size]),
                      name='part_prob', inputs=('label_with_neg', 'context_vec'))
        self.add_node(Dense(input_dim=embed_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='context_vec')
        self.add_node(SharedWeightsDenseLBLV4(self.nodes['part_prob'].W,
                                              self.nodes['part_prob'].b,
                                              activation='softmax'),
                      name='true_prob', inputs='context_vec')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

        def node_get_output(layer, train=False):
            X = layer.get_input(train)
            output = T.dot(X, layer.W)  # there is no activation.
            return output

        self.fit = None

    @staticmethod
    def encode_length(y_label, y_pred, mask=None):
        """
        :param y_label: true index labels with shape (ns, )
        :param y_pred: predicted probabilities with shape (ns, V)
        :param mask: mask
        :return: PPL
        """
        ## there is no need to clip here, for the prob. have already clipped by LBLayer
        # epsilon = 1e-7
        # y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
        # # scale preds so that the class probas of each sample sum to 1
        # y_pred /= y_pred.sum(axis=-1, keepdims=True)

        nb_samples = y_label.shape[0]
        idx = T.arange(nb_samples)
        probs_ = y_pred[idx, y_label]

        return -T.sum(T.log(probs_)), nb_samples

    # noinspection PyMethodOverriding
    def compile(self, optimizer=None):
        # from theano.compile.nanguardmode import NanGuardMode
        if optimizer is not None:
            logger.info('compiling with %s' % optimizer)
            self.optimizer = optimizers.get(optimizer)

        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)     # (k+1, ns)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)        # (ns, 1)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0],))  # (ns, )
        nrm_const = nrm_const.dimshuffle('x', 0)                 # (1, ns)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], ))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0)
        pos_prob_tst *= nrm_const_tst

        pre_prob_tst = T.clip(pre_prob_tst, epsilon, 1.-epsilon)
        pre_prob_tst = pre_prob_tst / T.sum(pre_prob_tst, axis=-1, keepdims=True)

        #TODO: mask not supported here
        # eps = 1.0e-10
        nb_words = pos_prob_trn[0].size.astype(floatX)
        nb_words_ = pos_prob_tst[0].size.astype(floatX)
        # sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        # sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        #
        # y_train = T.sum(T.log(T.clip(pos_prob_trn[0]/sum_pos_neg_trn[0], epsilon, 1.-epsilon))) / nb_words
        # y_train += T.sum(T.log(T.clip(neg_prob_trn[1:]/sum_pos_neg_trn[1:], epsilon, 1.-epsilon))) / nb_words
        # y_test = T.sum(T.log(T.clip(pos_prob_tst[0]/sum_pos_neg_tst[0], epsilon, 1.-epsilon))) / nb_words_
        # y_test += T.sum(T.log(T.clip(neg_prob_tst[1:] / sum_pos_neg_tst[1:], epsilon, 1.-epsilon))) / nb_words_

        eps = 1.0e-10
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn + 2*eps
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst + 2*eps
        y_train = T.sum(T.log((pos_prob_trn[0]+eps)/sum_pos_neg_trn[0]))
        y_train += T.sum(T.log((neg_prob_trn[1:]+eps)/sum_pos_neg_trn[1:]))
        y_train /= nb_words
        y_test = T.sum(T.log((pos_prob_tst[0]+eps)/sum_pos_neg_tst[0]))
        y_test += T.sum(T.log((neg_prob_tst[1:]+eps) / sum_pos_neg_tst[1:]))
        y_test /= nb_words_

        train_loss = -y_train
        test_loss = -y_test

        input0 = self.inputs['ngrams'].get_output(True)
        input1 = self.inputs['label_with_neg'].get_output(True)
        input2 = self.inputs['lookup_prob'].get_output(True)

        true_labels = input1[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        train_ins = test_ins = [input0, input1, input2]

        self._train = theano.function(train_ins, train_loss, updates=updates)
        self._train.out_labels = ['loss']
        self._test = theano.function(test_ins, [test_loss, encode_len, nb_words])
        self._test.out_labels = ['loss', 'encode_len', 'nb_words']

        def __summary_outputs(outs, batch_sizes):
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summary_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    @numba.jit
    def prepare_input(self, sents):
        ns = sents.shape[0]
        nt = sents.shape[1]
        nb_ele = sents.size  # NO. of words in the sentences.

        pad_idx = np.arange(self.vocab_size, self.vocab_size+self.context_size).reshape((1, -1))
        pad_idx = pad_idx.repeat(ns, axis=0)  # (ns, c), where c is context size
        idxes = np.hstack((pad_idx, sents))   # (ns, c+s), where s is sentence length

        X = np.empty(shape=(nb_ele, self.context_size), dtype='int32')
        y_label = np.empty(shape=(nb_ele, ), dtype='int32')
        start_end = np.array([0, 0], dtype='int32')
        k = 0
        for i in range(ns):  # loop on sentences
            start_end[0], start_end[1] = 0, self.context_size
            for _ in range(nt):  # loop on time (each time step corresponds to a word)
                X[k] = idxes[i, start_end[0]:start_end[1]]
                y_label[k] = idxes[i, start_end[1]]
                k += 1
                start_end += 1

        y_label_neg = self.negative_sample(y_label)
        neg_prob = self.neg_prob_table[y_label_neg]
        return X, y_label_neg, neg_prob

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_trained = 0.
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        self.validation(train_val_sents, batch_size, log_file)
        start_ = time()
        next_val_time = start_ + validation_interval
        for sents in sent_gen:
            mask = (sents > max_vocab)
            sents[mask] = max_vocab
            chunk = chunk_sentences(sentences, sents, batch_size)
            if chunk is None:
                continue

            x = self.prepare_input(chunk)
            loss = self._loop_train(x, batch_size)
            nb_trained += chunk.shape[0]
            nb_words_trained += chunk.size
            end_ = time()
            elapsed = float(end_ - start_)
            speed1 = nb_trained/elapsed
            speed2 = nb_words_trained/elapsed
            eta = (train_nb_words - nb_words_trained) / speed2
            eta_h = int(math.floor(eta/3600))
            eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
            logger.info('%s:Train - ETA: %02d:%02d - loss: %5.3f - speed: %.1f sent/s %.1f words/s' %
                        (self.__class__.__name__, eta_h, eta_m, loss, speed1, speed2))
            log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))

            if end_ > next_val_time:
                # noinspection PyUnresolvedReferences
                self.validation(train_val_sents, batch_size, log_file)
                next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                break

        logger.info('Training finished. Evaluating ...')
        log_file.info('Training finished. Evaluating ...')
        self.validation(val_sents, batch_size, log_file)
        if save_path is not None:
            self.save_params(save_path)
        log_file.close()

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        for sents in val_sents:
            x = self.prepare_input(sents)
            loss_, code_len_, nb_words_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_

        loss /= nb_words
        # try:
        ppl = math.exp(code_len/nb_words)
        # except OverflowError:
        #     logger.error("code_len: %.3f - nb_words: %d" % (code_len, nb_words))
        #     ppl = self.vocab_size * 1000
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl

    def _loop_train(self, data, batch_size):
        nb_words = data[0].shape[0]
        loss = 0.0
        batches = make_batches(nb_words, batch_size)
        for start, end in batches:
            X = data[0][start:end]
            y = data[1][:, start:end]
            p = data[2][:, start:end]
            loss_ = self._train(X, y, p)
            loss += loss_ * X.shape[0]

        loss /= nb_words
        return loss

    @staticmethod
    def _test_loop(f, ins, batch_size=128, verbose=0):
        nb_sample = ins[0].shape[0]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            X = ins[0][batch_start:batch_end]
            y = ins[1][:, batch_start:batch_end]
            p = ins[2][:, batch_start:batch_end]
            batch_outs = f(X, y, p)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(X.shape[0])

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class FFNNLangModel(Graph, LangModel):
    # the standard LBL language model with sparse coding extension, ZRegression
    def __init__(self, sparse_coding, context_size, nb_negative, embed_dims=200, context_dims=200,
                 max_part_sum=0.7, alpha=1.0,
                 init_embeddings=None, negprob_table=None, optimizer='adam'):
        super(FFNNLangModel, self).__init__()
        self.nb_negative = nb_negative
        self.alpha = alpha
        self.max_part_sum = max_part_sum
        self.sparse_coding = sparse_coding
        vocab_size = sparse_coding.shape[0]  # the extra word is for OOV
        self.nb_base = sparse_coding.shape[1] - 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        # self.loss = categorical_crossentropy
        # self.loss_fnc = objective_fnc(self.loss)
        self.optimizer = optimizers.get(optimizer)
        self.context_size = context_size
        self.weights = None
        # self.max_sent_len = max_sent_len
        tmp1 = sparse.csr_matrix((self.vocab_size, context_size), dtype=floatX)
        tmp2 = sparse.csr_matrix((context_size, self.nb_base+1), dtype=floatX)
        tmp3 = sparse.vstack([tmp1, sparse.csr_matrix(np.eye(context_size, dtype=floatX))])
        tmp4 = sparse.vstack([self.sparse_coding, tmp2])
        self.sparse_coding_pad = sparse.hstack([tmp4, tmp3], format='csr')

        if negprob_table is None:
            negprob_table_ = np.ones(shape=(vocab_size,), dtype=theano.config.floatX)/vocab_size
            self.neg_prob_table = negprob_table_
        else:
            self.neg_prob_table = negprob_table.astype(theano.config.floatX)

        # self.sampler = TableSampler(self.neg_prob_table)

        self.add_input(name='ngrams', ndim=2, dtype='int32')          # (ns, c), where c is the context size
        self.add_input(name='label_with_neg', ndim=2, dtype='int32')  # (k+1, ns)
        self.add_input(name='lookup_prob', ndim=2, dtype=floatX)      # (k+1, ns)

        cntx_codes = tsp.csr_matrix('cntx-codes', dtype=floatX)
        label_codes = tsp.csr_matrix('label_codes', dtype=floatX)
        cntx_idxes = self.inputs['ngrams'].get_output()
        # label_idxes = self.inputs['label_with_neg'].get_output()
        batch_shape = cntx_idxes.shape

        self.add_node(Identity(inputs={True: cntx_codes, False: cntx_codes}), name='cntx_codes_flat')
        self.add_node(Identity(inputs={True: label_codes, False: label_codes}), name='label_codes_flat')
        self.add_node(Identity(inputs={True: batch_shape, False: batch_shape}), name='cntx_shape')
        # self.add_node(Identity(inputs={True: codes, False: codes}), name='sparse_codes')

        self.add_node(SparseEmbedding(self.nb_base+self.context_size+1, embed_dims, weights=init_embeddings),
                      name='embedding', inputs=('cntx_codes_flat', 'cntx_shape'))
        self.add_node(EmbeddingParam(), name='embedding_param', inputs='embedding')
        self.add_node(Reshape(-1), name='reshape', inputs='embedding')
        self.add_node(Dense(context_size*embed_dims, context_dims), name='context_vec', inputs='reshape')
        self.add_node(PartialSoftmaxFFNN(context_dims, base_size=self.nb_base+1),
                      name='part_prob', inputs=('label_with_neg', 'label_codes_flat', 'context_vec'))
        self.add_node(Dense(input_dim=context_dims, output_dim=context_dims, activation='sigmoid'),
                      name='normalizer1', inputs='context_vec')
        self.add_node(Dense(input_dim=context_dims, output_dim=1, activation='exponential'),
                      name='normalizer', inputs='normalizer1')
        self.add_node(SharedWeightsDense(self.nodes['part_prob'].W,
                                         self.nodes['part_prob'].b,
                                         self.sparse_coding, activation='softmax'),
                      name='true_prob', inputs='context_vec')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')

        self.fit = None
        self.jobs_pools = None
        self.jobs_pools_post = None
        self.in_training_phase = Event()
        self.trn_finished = Event()
        self.all_finished = MEvent()

    def __del__(self):
        self.trn_finished.set()
        self.all_finished.set()


    @staticmethod
    def encode_length(y_label, y_pred, mask=None):
        """
        :param y_label: true index labels with shape (ns, )
        :param y_pred: predicted probabilities with shape (ns, V)
        :param mask: mask
        :return: PPL
        """
        ## there is no need to clip here, for the prob. have already clipped by LBLayer
        # epsilon = 1e-7
        # y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
        # # scale preds so that the class probas of each sample sum to 1
        # y_pred /= y_pred.sum(axis=-1, keepdims=True)

        nb_samples = y_label.shape[0]
        idx = T.arange(nb_samples)
        probs_ = y_pred[idx, y_label]

        return -T.sum(T.log(probs_)), nb_samples

    # noinspection PyMethodOverriding
    def compile(self, optimizer=None):
        # from theano.compile.nanguardmode import NanGuardMode
        if optimizer is not None:
            logger.info('compiling with %s' % optimizer)
            self.optimizer = optimizers.get(optimizer)
        # output of model
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']

        pos_prob_trn = pos_prob_layer.get_output(train=True)     # (k+1, ns)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)        # (ns, 1)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0],))  # (ns, )
        nrm_const = nrm_const.dimshuffle('x', 0)                 # (1, ns)
        pos_prob_trn *= nrm_const

        nrm_const_tst = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst, (nrm_const_tst.shape[0], ))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0)
        pos_prob_tst *= nrm_const_tst

        pre_prob_tst = T.clip(pre_prob_tst, epsilon, 1.-epsilon)
        pre_prob_tst = pre_prob_tst / T.sum(pre_prob_tst, axis=-1, keepdims=True)

        #TODO: mask not supported here
        # eps = 1.0e-10
        nb_words = pos_prob_trn[0].size.astype(floatX)
        nb_words_ = pos_prob_tst[0].size.astype(floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        sum_pos_neg_tst = pos_prob_tst + neg_prob_tst
        y_train = T.sum(T.log(T.clip(pos_prob_trn[0]/sum_pos_neg_trn[0], epsilon, 1.-epsilon))) / nb_words
        y_train += T.sum(T.log(T.clip(neg_prob_trn[1:]/sum_pos_neg_trn[1:], epsilon, 1.-epsilon))) / nb_words
        y_test = T.sum(T.log(T.clip(pos_prob_tst[0]/sum_pos_neg_tst[0], epsilon, 1.-epsilon))) / nb_words_
        y_test += T.sum(T.log(T.clip(neg_prob_tst[1:] / sum_pos_neg_tst[1:], epsilon, 1.-epsilon))) / nb_words_

        # max_part_sum = self.max_part_sum
        # part_sum = T.sum(pos_prob_trn, axis=0)
        # tmp = T.switch(part_sum > max_part_sum, part_sum-max_part_sum, 0.0)
        # not_prob_loss = T.sum(tmp)/(T.nonzero(tmp)[0].size+1.0)

        not_prob_loss = T.sum(pos_prob_trn) / nb_words
        # not_prob_loss = T.as_tensor_variable(0.0)

        # part_sum_tst = T.sum(pos_prob_tst, axis=0)
        # tmp = T.switch(part_sum_tst > max_part_sum, part_sum_tst-max_part_sum, 0.0)
        # not_prob_loss_tst = T.sum(tmp)/(T.nonzero(tmp)[0].size+1.0)

        not_prob_loss_tst = T.sum(pos_prob_tst) / nb_words_
        # not_prob_loss_tst = T.as_tensor_variable(0.0)

        input0 = self.inputs['ngrams'].get_output(True)
        input1 = self.inputs['label_with_neg'].get_output(True)
        input2 = self.nodes['cntx_codes_flat'].get_output(True)
        input3 = self.nodes['label_codes_flat'].get_output(True)
        input4 = self.inputs['lookup_prob'].get_output(True)

        true_labels = input1[0]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train + self.alpha * not_prob_loss
        test_loss = -y_test + self.alpha * not_prob_loss_tst
        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        train_ins = [input0, input1, input2, input3, input4]
        test_ins = [input0, input1, input2, input3, input4]

        self._train = theano.function(train_ins, [train_loss, not_prob_loss], updates=updates)
        self._train.out_labels = ['loss']
        self._test = theano.function(test_ins, [test_loss, encode_len, nb_words])
        self._test.out_labels = ['loss', 'encode_len', 'nb_words']

        self.all_metrics = ['loss', 'ppl', 'val_loss', 'val_ppl']

        def __summary_outputs(outs, batch_sizes):
            out = np.array(outs, dtype=theano.config.floatX)
            loss, encode_len, nb_words = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_loss, smry_encode_len, smry_nb_words]

        self._train.summarize_outputs = __summary_outputs
        self._test.summarize_outputs = __summary_outputs

    def train(self, data_file='../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2', save_path=None,
              batch_size=256, train_nb_words=100000000, val_nb_words=100000, train_val_nb=100000,
              validation_interval=1800, log_file=None, nb_data_workers=6, data_pool_size=10):
        opt_info = self.optimizer.get_config()
        opt_info = ', '.join(["{}: {}".format(n, v) for n, v in opt_info.items()])

        logger.info('training with file: %s' % data_file)
        logger.info('training with batch size %d' % batch_size)
        logger.info('training with %d words; validate with %d words during training; '
                    'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        logger.info('validate every %f seconds' % float(validation_interval))
        logger.info('optimizer: %s' % opt_info)

        log_file = LogInfo(log_file)
        log_file.info('training with file: %s' % data_file)
        log_file.info('training with batch size %d' % batch_size)
        log_file.info('training with %d words; validate with %d words during training; '
                      'evaluate with %d words after training' % (train_nb_words, train_val_nb, val_nb_words))
        log_file.info('validate every %f seconds' % float(validation_interval))
        log_file.info('optimizer: %s' % opt_info)

        sentences = [None for _ in range(MAX_SETN_LEN)]  # TODO: sentences longer than 64 are ignored.

        max_vocab = self.vocab_size - 1
        nb_words_trained = 0.0
        sent_gen = grouped_sentences(data_file)
        val_sents = self.get_val_data(sent_gen, val_nb_words)
        train_val_sents = self.get_val_data(sent_gen, train_val_nb)

        data_workers = []
        pre_data = Queue(data_pool_size)
        post_data = Queue(data_pool_size*30)
        self.jobs_pools = pre_data
        self.jobs_pools_post = post_data

        xk = Array(ctypes.c_int32, np.arange(self.vocab_size, dtype='int32'), lock=False)
        # a_type = ctypes.c_double if str(self.neg_prob_table.dtype) == 'float64' else ctypes.c_float
        assert str(self.neg_prob_table.dtype) == 'float32'
        pk = Array(ctypes.c_float, self.neg_prob_table, lock=False)

        # a_type = ctypes.c_double if str(self.sparse_coding.dtype) == 'float64' else ctypes.c_float
        assert str(self.sparse_coding.dtype) == 'float32'
        sp_data = Array(ctypes.c_float, self.sparse_coding.data, lock=False)
        assert str(self.sparse_coding.indices.dtype) == 'int32'
        assert str(self.sparse_coding.indptr.dtype) == 'int32'
        sp_indices = Array(ctypes.c_int32, self.sparse_coding.indices, lock=False)
        sp_indptr = Array(ctypes.c_int32, self.sparse_coding.indptr, lock=False)

        # a_type = ctypes.c_double if str(self.sparse_coding_pad.dtype) == 'float64' else ctypes.c_float
        assert str(self.sparse_coding_pad.dtype) == 'float32'
        sp_pad_data = Array(ctypes.c_float, self.sparse_coding_pad.data, lock=False)
        assert str(self.sparse_coding_pad.indices.dtype) == 'int32'
        assert str(self.sparse_coding_pad.indptr.dtype) == 'int32'
        sp_pad_indices = Array(ctypes.c_int32, self.sparse_coding_pad.indices, lock=False)
        sp_pad_indptr = Array(ctypes.c_int32, self.sparse_coding_pad.indptr, lock=False)

        for _ in range(nb_data_workers):
            # prepare_input(sents_queue, jobs_pool, all_finished,
            #       vocab_size, context_size, batch_size, nb_negative, xk, pk,
            #       sp_data, sp_indices, sp_indptr, sp_shape,
            #       sp_pad_data, sp_pad_indices, sp_pad_inptr, sp_pad_shape):
            p = Process(target=prepare_input, args=(pre_data, post_data, self.all_finished,
                                                    self.vocab_size, self.context_size, batch_size, self.nb_negative,
                                                    xk, pk, sp_data, sp_indices, sp_indptr, self.sparse_coding.shape,
                                                    sp_pad_data, sp_pad_indices, sp_pad_indptr, self.sparse_coding_pad.shape))
            p.daemon = True
            data_workers.append(p)
            p.start()

        self.in_training_phase.clear()
        self.validation(train_val_sents, log_file)

        def chunk_trn_generator():
            for sents in sent_gen:
                if self.trn_finished.is_set():
                    break

                mask = (sents > max_vocab)
                sents[mask] = max_vocab
                chunk = chunk_sentences(sentences, sents, int(batch_size//10))
                if chunk is None:
                    continue
                self.in_training_phase.wait()
                pre_data.put(chunk)

            self.trn_finished.set()
            logger.debug('trn data finished')

        gen_chunk_thread = Thread(target=chunk_trn_generator)
        gen_chunk_thread.setDaemon(True)
        gen_chunk_thread.start()

        loss = 0.0
        nb_chunk = 0.0
        nb_cyc = 0
        part_sum = 0.0
        start_ = time()
        next_val_time = start_ + validation_interval
        next_report = start_ + 1.0
        self.in_training_phase.set()

        while not self.trn_finished.is_set() or not post_data.empty():
            ins = post_data.get()
            # if ins is None:
            #     if nb_none == nb_data_workers:
            #         break
            #     else:
            #         nb_none += 1
            #         continue
            loss_, part_sum_ = self._train(*ins)
            nb_cyc += 1
            nb_cyc %= 20
            nb_words_trained += ins[0].shape[0]
            nb_chunk += ins[0].shape[0]
            loss += loss_ * ins[0].shape[0]
            part_sum = max(part_sum, part_sum_)
            if nb_cyc == 0:
                end_ = time()
                if end_ > next_report:
                    elapsed = float(end_ - start_)
                    speed2 = nb_words_trained/elapsed
                    eta = (train_nb_words - nb_words_trained) / speed2
                    eta_h = int(math.floor(eta/3600))
                    eta_m = int(math.ceil((eta - eta_h * 3600)/60.))
                    loss /= nb_chunk
                    logger.info('%s:Train - ETA: %02d:%02d - loss: %5.3f - mean_prob: %.2e - speed: %.1f words/s' %
                                (self.__class__.__name__, eta_h, eta_m, loss, part_sum, speed2))
                    log_file.info('%s:Train - time: %f - loss: %.6f' % (self.__class__.__name__, end_, loss))
                    nb_chunk = 0.0
                    loss = 0.0
                    next_report = end_ + 1.0
                    part_sum = 0.0

                if end_ > next_val_time:
                    logger.debug('pausing training data generation and consuming all generated data')
                    self.in_training_phase.clear()
                    while not self.jobs_pools_post.empty() or not self.jobs_pools.empty():
                        ins = self.jobs_pools_post.get()
                        self._train(*ins)
                        nb_words_trained += ins[0].shape[0]
                    logger.debug('Before validation')
                    # noinspection PyUnresolvedReferences
                    self.validation(train_val_sents, log_file)
                    logger.debug('END validation. resume training data generation')
                    self.in_training_phase.set()
                    next_val_time = time() + validation_interval

            if nb_words_trained >= train_nb_words:
                self.trn_finished.set()
                break

        # consume all the produced tasks. The data generation thread will automatically shutdown, for the trn_finished
        # event is set.
        self.in_training_phase.set()  # make sure it is not blocking
        while not self.jobs_pools_post.empty() or not self.jobs_pools.empty():
            ins = self.jobs_pools_post.get()
            self._train(*ins)

        # Now the training data is consumed out. Let's evaluate...
        logger.info('Training finished. Evaluating ...')
        log_file.info('Training finished. Evaluating ...')
        self.validation(val_sents, log_file)
        self.all_finished.set()  # signal the all_finished event to shutdown all worker processes.
        if save_path is not None:
            self.save_params(save_path)
        log_file.close()

        for _ in range(10):
            flags = map(Process.is_alive, data_workers)
            if not any(flags):
                break
            for flag, p in zip(flags, data_workers):
                if flag is True:
                    logger.info("%s is alive" % p.name)
            sleep(5)

    def validation(self, val_sents, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0

        def chunk_val_generator():
            for sents in val_sents:
                self.jobs_pools.put(sents)

        gen_chunk_thread = Thread(target=chunk_val_generator)
        gen_chunk_thread.setDaemon(True)
        gen_chunk_thread.start()

        logger.debug('begin val loop')
        while True:
            ins = self.jobs_pools_post.get()
            loss_, code_len_, nb_words_ = self._test(*ins)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_
            if self.jobs_pools_post.empty() and self.jobs_pools.empty():
                break
        logger.debug('end val loop')

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)

        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f' % (self.__class__.__name__, loss, ppl))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f' % (self.__class__.__name__, loss, ppl))

        return loss, ppl


def negative_sampleLBLV2(y, sampler, nb_negative):
        ret = np.empty(shape=(nb_negative+1,) + y.shape, dtype=y.dtype)
        ret[0] = y
        ret[1:] = sampler.rvs(size=ret[1:].shape).astype(y.dtype)
        return ret


# @numba.jit([(numba.int32[:, :], numba.int32, numba.int32),
#             (numba.int32[:, :], numba.int64, numba.int32),
#             (numba.int32[:, :], numba.int32, numba.int64),
#             (numba.int32[:, :], numba.int64, numba.int64)], nogil=True)
@numba.jit
def get_cntx_label(sents, vocab_size, context_size):
    ns = sents.shape[0]
    nt = sents.shape[1]
    nb_ele = sents.size  # NO. of words in the sentences.

    pad_idx = np.arange(vocab_size, vocab_size+context_size).reshape((1, -1))
    pad_idx = pad_idx.repeat(ns, axis=0)  # (ns, c), where c is context size
    idxes = np.hstack((pad_idx, sents))   # (ns, c+s), where s is sentence length

    X = np.empty(shape=(nb_ele, context_size), dtype='int32')
    y_label = np.empty(shape=(nb_ele, ), dtype='int32')
    start_end = np.array([0, 0], dtype='int32')
    k = 0
    for i in range(ns):  # loop on sentences
        start_end[0], start_end[1] = 0, context_size
        for _ in range(nt):  # loop on time (each time step corresponds to a word)
            X[k] = idxes[i, start_end[0]:start_end[1]]
            y_label[k] = idxes[i, start_end[1]]
            k += 1
            start_end += 1
    return X, y_label


def prepare_input(sents_queue, jobs_pool, all_finished,
                  vocab_size, context_size, batch_size, nb_negative, xk, pk,
                  sp_data, sp_indices, sp_indptr, sp_shape,
                  sp_pad_data, sp_pad_indices, sp_pad_inptr, sp_pad_shape):
    xk = np.frombuffer(xk, dtype='int32')
    pk = np.frombuffer(pk, dtype='float32')
    sp_data = np.frombuffer(sp_data, dtype='float32')
    sp_indices = np.frombuffer(sp_indices, dtype='int32')
    sp_indptr = np.frombuffer(sp_indptr, dtype='int32')
    sp_pad_data = np.frombuffer(sp_pad_data, dtype='float32')
    sp_pad_indices = np.frombuffer(sp_pad_indices, dtype='int32')
    sp_pad_inptr = np.frombuffer(sp_pad_inptr, dtype='int32')
    sparse_coding = sparse.csr_matrix((sp_data, sp_indices, sp_indptr), shape=sp_shape)
    sparse_coding_pad = sparse.csr_matrix((sp_pad_data, sp_pad_indices, sp_pad_inptr), shape=sp_pad_shape)
    logger.debug('sp:shape: %s, %s, len(pk): %d - sum(pk): %.2f - min(pk): %.2f' %
                 (str(sparse_coding.shape), str(sparse_coding_pad.shape), len(pk), pk.sum(), pk.min()))
    assert xk[0] == 0
    sd = abs(int(np.frombuffer(os.urandom(4), dtype='int32')))
    np.random.seed(sd)
    logger.debug('seed: %d' % sd)
    custm = stats.rv_discrete(name='custm', values=(xk, pk))

    while not all_finished.is_set() or not sents_queue.empty():
        sents = sents_queue.get()
        X, y_label = get_cntx_label(sents, vocab_size, context_size)
        y_label = negative_sampleLBLV2(y_label, custm, nb_negative)
        probs = pk[y_label]

        nb_sample = X.shape[0]
        batches = make_batches(nb_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            if batch_end <= batch_start:
                break

            X_ = X[batch_start:batch_end].copy()
            y_ = y_label[:, batch_start:batch_end].copy()
            sp_x_ = sparse_coding_pad[X_.ravel()]
            sp_y_ = sparse_coding[y_.ravel()]
            probs_ = probs[:, batch_start:batch_end].copy()

            jobs_pool.put((X_, y_, sp_x_, sp_y_, probs_))

if __name__ == '__main__':
    from keras.optimizers import rmsprop, AdamAnneal, adam, adadelta, sgd
    from utils import get_unigram_probtable
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(level=logging.DEBUG)
    # data_path = '../data/corpus/wiki-sg-norm-lc-drop-bin.bz2'
    # model = SimpleLangModel(vocab_size=10000, embed_dims=128, context_dims=128, optimizer='adam')
    # model.compile()
    # # model.train_from_dir(data_fn=data_path, validation_split=0., batch_size=256, verbose=1)
    # model.train(data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2',
    #             save_path='../data/models/lang/simple-e128-c128.pkl',
    #             batch_size=256, train_nb_words=100000000,
    #             val_nb_words=5000000, train_val_nb=100000)
    #             #batch_size=256, train_nb_words=1000000, val_nb_words=10000, validation_interval=20)

    # data_path = '../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2'
    # # vocab_size, context_size, embed_dims=128, optimizer='adam'
    # opt = AdamAnneal(lr=0.001, lr_min=0.0001, gamma=0.03)
    # nb_vocab = 20000
    # unigram_table = get_unigram_probtable(nb_words=nb_vocab, save_path='../data/wiki-unigram-prob-size%d.pkl' % nb_vocab)
    # model = LBLangModelV4(vocab_size=nb_vocab, context_size=5, embed_dims=200, negprob_table=unigram_table)
    # model.compile(opt)
    # # model.train_from_dir(data_fn=data_path, validation_split=0., batch_size=256, verbose=1)
    # model.train(data_file=data_path,
    #             batch_size=512, train_nb_words=100000000,
    #             val_nb_words=5000000, train_val_nb=100000)

    context_size = 5
    embed_dim = 200
    data_path = '../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2'
    # opt = AdamAnneal(lr=0.005, lr_min=0.001, gamma=0.0005)
    opt = adam(lr=0.001)
    # opt.clipnorm = 100.0
    # opt = adadelta()

    with file('../data/sparse/total-app-a0.1-b0.1-w1-0.1-15000.pkl', 'rb') as f:
        sparse_coding = pickle.load(f)
        # print sparse_coding.dtype

    nb_vocab = 50000
    sparse_coding = sparse_coding[nb_vocab//1000]
    nb_vocab, nb_base = sparse_coding.shape
    nb_base -= 1
    unigram_table = get_unigram_probtable(nb_words=nb_vocab, save_path='../data/wiki-unigram-prob-size%d.pkl' % nb_vocab)

    # sparse_coding, context_size, nb_negative, embed_dims=200, init_embeddings=None,
    #             negprob_table=None, optimizer='adam'
    model = FFNNLangModel(sparse_coding=sparse_coding,
                          context_size=context_size,
                          max_part_sum=0.01, alpha=10,
                          embed_dims=embed_dim,
                          nb_negative=20,
                          init_embeddings=None)
    # model = LBLangModelV4(vocab_size=10001, context_size=5, embed_dims=200, nb_negative=50, negprob_table=unigram_table)
    logger.debug('model constructed')
    model.compile(opt)
    logger.debug('model compiled')
    #model.train_from_dir(data_fn=data_path, validation_split=0., batch_size=256, verbose=1)
    model.train(data_file=data_path,
                # save_path='../data/models/lang/ffnn-lbl-e200-c200-lr0.01-lr_min0.001-gamma0.03-d-test.pkl',
                batch_size=3096, train_nb_words=100000000,
                val_nb_words=500000, train_val_nb=30000,
                validation_interval=180,
                log_file='../logs/nce-ffnn-alpha5-e200-c200-lr0.01-lr_min0.001-gamma0.03-d-test.log',
                nb_data_workers=2)
    #model.profile_model()