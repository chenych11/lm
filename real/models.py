#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'
import theano
import re
import os
import math
from time import time
from theano import tensor as T
from keras.models import Sequential, Graph, make_batches
from keras import optimizers
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers import containers
# from keras.layers.embeddings import LookupTable
import numpy as np
import logging
from layers import LangLSTMLayer, PartialSoftmax, Split, LookupProb, PartialSoftmaxV1, \
    TreeLogSoftmax, SparseEmbedding, Identity, PartialSoftmaxV4, SharedWeightsDense, \
    LangLSTMLayerV5, LangLSTMLayerV6, SparseEmbeddingV6
from utils import LangHistory, LangModelLogger, categorical_crossentropy, objective_fnc, \
    TableSampler, slice_X, chunk_sentences
# noinspection PyUnresolvedReferences
from lm.utils.preprocess import import_wordmap, grouped_sentences
import theano.sparse as tsp
import cPickle as pickle

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

        return T.sum(T.log(1.0/probs)), nb_words

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


if __name__ == '__main__':
    data_path = '../data/corpus/wiki-sg-norm-lc-drop-bin.bz2'
    model = SimpleLangModel(vocab_size=10000, embed_dims=128, context_dims=128, optimizer='adam')
    model.compile()
    # model.train_from_dir(data_fn=data_path, validation_split=0., batch_size=256, verbose=1)
    model.train(data_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2',
                save_path='../data/models/lang/simple-e128-c128.pkl',
                batch_size=256, train_nb_words=100000000,
                val_nb_words=5000000, train_val_nb=100000)
                #batch_size=256, train_nb_words=1000000, val_nb_words=10000, validation_interval=20)