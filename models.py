#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

import theano
from theano import tensor as T
from keras.utils.theano_utils import shared_zeros
from keras import activations, initializations, regularizers, constraints
from keras.layers.recurrent import Recurrent
from keras.models import Sequential, Graph, make_batches, batch_shuffle
from keras.layers.embeddings import Embedding
from keras.layers.core import Layer, Dense, Dropout, MultiInputLayer, LayerList, Reshape
from keras.callbacks import BaseLogger, History
from keras import callbacks as cbks
from keras import optimizers
from keras import objectives
from keras.models import objective_fnc
from keras.layers import containers
from keras.utils.generic_utils import Progbar
import numpy as np
from scipy.stats import rv_discrete
import logging
import os
import re
import math

logger = logging.getLogger('lm.models')
floatX = theano.config.floatX

class LangHistory(History):

    # def on_train_begin(self, logs=None):
    #     # logs = {} if logs is None else logs
    #     self.epoch = []
    #     self.history = {}
    #
    # def on_epoch_begin(self, epoch, logs=None):
    #     self.seen = 0
    #     self.totals = {}

    def on_batch_end(self, batch, logs=None):
        logs = {} if logs is None else logs
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k == 'encode_len' or 'nb_words':
                try:
                    self.totals[k] += v
                except KeyError:
                    self.totals[k] = v
                continue

            try:
                self.totals[k] += v * batch_size
            except KeyError:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.totals, 'encode_len') and hasattr(self, 'nb_words'):
            ppl = math.exp(self.totals['encode_len']/float(self.totals['nb_words']))
            k = 'ppl'
            try:
                self.history[k].append(ppl)
            except KeyError:
                self.history[k] = [ppl]

        if hasattr(self.totals, 'val_encode_len') and hasattr(self, 'val_nb_words'):
            val_ppl = math.exp(self.totals['val_encode_len']/float(self.totals['val_nb_words']))
            k = 'val_ppl'
            try:
                self.history[k].append(val_ppl)
            except KeyError:
                self.history[k] = [val_ppl]

        k = 'loss'
        v = self.totals[k]
        try:
            self.history[k].append(v/float(self.seen))
        except KeyError:
            self.history[k] = [v/float(self.seen)]


class LangModelLogger(BaseLogger):
    def __init__(self):
        super(LangModelLogger, self).__init__()
        self.verbose = None
        self.nb_epoch = None
        self.seen = 0
        self.totals = {}
        self.progbar = None
        self.log_values = None

    # def on_train_begin(self, logs=None):
    #     logger.debug('Begin training...')
    #     self.verbose = self.params['verbose']
    #     self.nb_epoch = self.params['nb_epoch']
    #
    # def on_epoch_begin(self, epoch, logs=None):
    #     # print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
    #     self.progbar = Progbar(target=self.params['nb_sample'], verbose=1)
    #     self.seen = 0
    #     self.totals = {}
    #
    # def on_batch_begin(self, batch, logs=None):
    #     if self.seen < self.params['nb_sample']:
    #         self.log_values = []
    #         self.params['metrics'] = ['loss', 'ppl', 'val_loss', 'val_ppl']

    def on_batch_end(self, batch, logs=None):
        logs = {} if logs is None else logs
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k == 'encode_len' or 'nb_words':
                try:
                    self.totals[k] += v
                except KeyError:
                    self.totals[k] = v
                continue

            try:
                self.totals[k] += v * batch_size
            except KeyError:
                self.totals[k] = v * batch_size

        if 'encode_len' in self.totals and 'nb_words' in self.totals and 'ppl' in self.params['metrics']:
            self.totals['ppl'] = math.exp(self.totals['encode_len']/float(self.totals['nb_words']))
            self.log_values.append(('ppl', self.totals['ppl']))
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # skip progbar update for the last batch; will be handled by on_epoch_end
            if self.seen < self.params['nb_sample']:
                self.progbar.update(self.seen, self.log_values)

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            self.progbar = Progbar(target=self.params['nb_sample'],
                                   verbose=self.verbose)
        self.seen = 0
        self.totals = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        # logger.debug('log keys: %s' % str(logs.keys()))
        for k in self.params['metrics']:
            if k in self.totals:
                if k != 'ppl':
                    self.log_values.append((k, self.totals[k] / self.seen))
                else:
                    self.totals['ppl'] = math.exp(self.totals['encode_len']/float(self.totals['nb_words']))
                    self.log_values.append((k, self.totals['ppl']))
            if k in logs:
                self.log_values.append((k, logs[k]))
        if 'val_encode_len' in logs and 'val_nb_words' in logs:
            val_ppl = math.exp(logs['val_encode_len']/float(logs['val_nb_words']))
            self.log_values.append(('val_ppl', val_ppl))

        self.progbar.update(self.seen, self.log_values)


class LangLSTMLayer(Recurrent):
    """ Modified from LSTMLayer: adaptation for Language modelling
        optimized version: Not using mask in _step function and tensorized computation.
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    """

    def __init__(self, input_dim, output_dim=128, train_init_cell=True, train_init_h=True,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 input_activation='tanh', gate_activation='hard_sigmoid', output_activation='tanh',
                 weights=None, truncate_gradient=-1):

        super(LangLSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)
        self.input = T.tensor3()
        self.time_range = None

        W_z = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_z = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_z = shared_zeros(self.output_dim)

        W_i = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_i = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_i = shared_zeros(self.output_dim)

        W_f = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_f = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_f = self.forget_bias_init(self.output_dim)

        W_o = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_o = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_o = shared_zeros(self.output_dim)

        self.h_m1 = shared_zeros(shape=(1, self.output_dim), name='h0')
        self.c_m1 = shared_zeros(shape=(1, self.output_dim), name='c0')

        W = np.vstack((W_z[np.newaxis, :, :],
                       W_i[np.newaxis, :, :],
                       W_f[np.newaxis, :, :],
                       W_o[np.newaxis, :, :]))  # shape = (4, input_dim, output_dim)
        R = np.vstack((R_z[np.newaxis, :, :],
                       R_i[np.newaxis, :, :],
                       R_f[np.newaxis, :, :],
                       R_o[np.newaxis, :, :]))  # shape = (4, output_dim, output_dim)
        self.W = theano.shared(W, name='Input to hidden weights (zifo)', borrow=True)
        self.R = theano.shared(R, name='Recurrent weights (zifo)', borrow=True)
        self.b = theano.shared(np.zeros(shape=(4, self.output_dim), dtype=theano.config.floatX),
                               name='bias', borrow=True)

        self.params = [self.W, self.R, self.b]
        if train_init_cell:
            self.params.append(self.c_m1)
        if train_init_h:
            self.params.append(self.h_m1)

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              Y_t,  # sequence
              h_tm1, c_tm1,  # output_info
              R):  # non_sequence
        # h_mask_tm1 = mask_tm1 * h_tm1
        # c_mask_tm1 = mask_tm1 * c_tm1
        G_tm1 = T.dot(h_tm1, R)
        M_t = Y_t + G_tm1
        z_t = self.input_activation(M_t[:, 0, :])
        ifo_t = self.gate_activation(M_t[:, 1:, :])
        i_t = ifo_t[:, 0, :]
        f_t = ifo_t[:, 1, :]
        o_t = ifo_t[:, 2, :]
        # c_t_cndt = f_t * c_tm1 + i_t * z_t
        # h_t_cndt = o_t * self.output_activation(c_t_cndt)
        c_t = f_t * c_tm1 + i_t * z_t
        h_t = o_t * self.output_activation(c_t)
        # h_t = mask * h_t_cndt + (1-mask) * h_tm1
        # c_t = mask * c_t_cndt + (1-mask) * c_tm1
        return h_t, c_t

    def get_output_mask(self, train=None):
        return None

    def _get_output_with_mask(self, train=False):
        X = self.get_input(train)
        # mask = self.get_padded_shuffled_mask(train, X, pad=0)
        mask = self.get_input_mask(train=train)
        ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32').ravel()
        max_time = T.max(ind) - 1   # drop the last frame
        X = X.dimshuffle((1, 0, 2))
        Y = T.dot(X, self.W) + self.b
        # h0 = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        h0 = T.repeat(self.h_m1, X.shape[1], axis=0)
        c0 = T.repeat(self.c_m1, X.shape[1], axis=0)

        [outputs, _], updates = theano.scan(
            self._step,
            sequences=Y,
            outputs_info=[h0, c0],
            non_sequences=[self.R], n_steps=max_time,
            truncate_gradient=self.truncate_gradient, strict=True,
            allow_gc=theano.config.scan.allow_gc)

        res = T.concatenate([h0.dimshuffle('x', 0, 1), outputs], axis=0).dimshuffle((1, 0, 2))
        return res

    def _get_output_without_mask(self, train=False):
        X = self.get_input(train)
        # mask = self.get_padded_shuffled_mask(train, X, pad=0)
        # mask = self.get_input_mask(train=train)
        # ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32')
        # max_time = T.max(ind)
        max_time = X.shape[1] - 1  # drop the last frame
        X = X.dimshuffle((1, 0, 2))
        Y = T.dot(X, self.W) + self.b
        # h0 = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        h0 = T.repeat(self.h_m1, X.shape[1], axis=0)
        c0 = T.repeat(self.c_m1, X.shape[1], axis=0)

        [outputs, _], updates = theano.scan(
            self._step,
            sequences=Y,
            outputs_info=[h0, c0],
            non_sequences=[self.R], n_steps=max_time,
            truncate_gradient=self.truncate_gradient, strict=True,
            allow_gc=theano.config.scan.allow_gc)

        res = T.concatenate([h0.dimshuffle('x', 0, 1), outputs], axis=0).dimshuffle((1, 0, 2))
        return res

    def get_output(self, train=False):
        mask = self.get_input_mask(train=train)
        if mask is None:
            return self._get_output_without_mask(train=train)
        else:
            return self._get_output_with_mask(train=train)

    def set_init_cell_parameter(self, is_param=True):
        if is_param:
            if self.c_m1 not in self.params:
                self.params.append(self.c_m1)
        else:
            self.params.remove(self.c_m1)

    def set_init_h_parameter(self, is_param=True):
        if is_param:
            if self.h_m1 not in self.params:
                self.params.append(self.h_m1)
        else:
            self.params.remove(self.h_m1)

    def get_time_range(self, train):
        mask = self.get_input_mask(train=train)
        ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32')
        self.time_range = ind
        return ind

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "forget_bias_init": self.forget_bias_init.__name__,
                "input_activation": self.input_activation.__name__,
                "gate_activation": self.gate_activation.__name__,
                "truncate_gradient": self.truncate_gradient}


class LangModel(object):
    def __init__(self):
        super(LangModel, self).__init__()

    @staticmethod
    def encode_length(y_label, y_pred, mask=None):
        # probs_ = T.sum(y_true * y_pred, axis=-1)
        # TODO: it may be very slow when the vocabulary is very large.
        # probs_ = y_pred[y_true.nonzero()]
        # y_label = T.flatten(y_label)
        # y_pred = T.reshape(y_pred, (-1, y_pred.shape[-1]))
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


class SimpleLangModel(Sequential):
    def __init__(self, vocab_size, embed_dims=128, context_dims=128, loss='categorical_crossentropy', optimizer='adam'):
        super(SimpleLangModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims

        self.optimizer = optimizers.get(optimizer)
        self.loss = objectives.get(loss)
        self.loss_fnc = objective_fnc(self.loss)

        self.add(Embedding(input_dim=vocab_size, output_dim=embed_dims))
        self.add(LangLSTMLayer(input_dim=embed_dims, output_dim=context_dims))
        # self.add(Dropout(0.5))
        self.add(Dense(input_dim=context_dims, output_dim=vocab_size, activation='softmax'))

    @staticmethod
    def encode_length(y_true, y_pred, mask):
        probs_ = T.sum(y_true * y_pred, axis=-1)

        if mask is None:
            nb_words = y_true.shape[0] * y_true.shape[1]
            probs = probs_.ravel() + 1.0e-30
        else:
            nb_words = mask.sum()
            probs = probs_[mask.nonzero()] + 1.0e-30

        return T.sum(T.log(1.0/probs)), nb_words

    def WordEmbedding(self, embed_dim=None):
        if embed_dim is not None and self.embed_dim is not None:
            logger.warn('The dimension of embedding is specified, but is not equal to the model\'s embed_dim.'
                        'The newly specified dimension is used.')
        dims = embed_dim if embed_dim is not None else self.embed_dim
        if dims is not None:
            self.embed_dim = dims
            return Embedding(self.vocab_size, self.embed_dim)
        else:
            raise ValueError('Embedding dimension not specified')

    def LangLSTM(self, out_dim):
        if self.embed_dim is None:
            raise ValueError('Embedding dimension not specified')
        return LangLSTMLayer(self.embed_dim, output_dim=out_dim)

    def train(self, X, y, callbacks, show_metrics, batch_size=128, extra_callbacks=(LangModelLogger(), ),
              validation_split=0., validation_data=None, shuffle=False, verbose=1):
        self.fit(X, y, callbacks, show_metrics, batch_size=batch_size, nb_epoch=1, verbose=verbose,
                 extra_callbacks=extra_callbacks, validation_split=validation_split,
                 validation_data=validation_data, shuffle=shuffle, show_accuracy=False)

    def train_from_dir(self, dir_, data_regex=re.compile(r'\d{3}.bz2'), callbacks=LangHistory(),
                       show_metrics=('loss', 'ppl'), *args, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if data_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            # y = np.zeros((X.shape[0], X.shape[1], self.vocab_size), dtype=np.int8)
            tmp = np.eye(self.vocab_size, dtype='int8')
            y = tmp[X]
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         idx = X[i, j]
            #         y[i, j, idx] = 1
            logger.info('Training on %s' % f)
            self.train(X, y, callbacks, show_metrics, *args, **kwargs)

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
        self.y = T.zeros_like(self.y_train)

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

        # train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)),
        #                         dtype=theano.config.floatX)
        # test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)),
        #                        dtype=theano.config.floatX)

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
        predict_ins = [self.X_test]

        self._train = theano.function(train_ins, [train_loss, train_ce, nb_trn_wrd], updates=updates,
                                      allow_input_downcast=True)
        self._train.out_labels = ['loss', 'encode_len', 'nb_words']
        self._predict = theano.function(predict_ins, self.y_test, allow_input_downcast=True)
        self._predict.out_labels = ['predicted']
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
        # noinspection PyUnresolvedReferences
        self.fit = self._Sequential__fit_unweighted


class Identity(Layer):
    # todo: mask support
    def __init__(self, inputs):
        super(Identity, self).__init__()
        self.inputs = inputs

    def get_output(self, train=False):
        return self.inputs[train]

    def get_input(self, train=False):
        return self.inputs[train]


class Split(LayerList):
    def __init__(self, split_at, split_axis=-1, keep_dim=False, slot_names=('head', 'tail')):
        """ Split a layer into to parallel layers.
            :param split_at: the index to split the layer. the first layer is 0:split_at, the second layer is split_at:
            :param split_axis: split axis.
        """
        super(Split, self).__init__()
        self.split_axis = split_axis
        self.split_at = split_at
        self.input_layer = None
        self.output_layer_names = slot_names
        self.input_layer_names = ['whole']
        self.__output_slots = []
        self.keep_dim = keep_dim or (abs(split_at) > 1)

        # if broadcastable is None:
        #     broadcastable = ((), ())
        # broadcastable = tuple(() if x is None else x for x in broadcastable)
        # self.broadcastable_axes = [[idx for idx, v in enumerate(broadcastable[i]) if v]
        #                            for i in range(len(broadcastable))]

    @property
    def nb_output(self):
        return 2

    @property
    def nb_input(self):
        return 1

    def set_inputs(self, inputs):
        super(Split, self).set_inputs(inputs)
        self.input_layer = self.input_layers[0]
        self.get_output_layers()

    # def _set_input(self, idx, layer):
    #     raise NotImplementedError('The input layer must be given at instance construction time')

    def get_output_layers(self):
        if self.output_layers:
            return self.output_layers
        out0, out1 = self.get_output_slots()
        layer0 = Identity(out0)
        layer1 = Identity(out1)
        self.output_layers = [layer0, layer1]
        return self.output_layers

    def get_output_slots(self):
        # todo: mask support
        single = True if abs(self.split_at) == 1 else False
        if self.__output_slots:
            return self.__output_slots
        out = self.input_layer.get_output(train=True)
        if self.split_at >= 0:
            sz0 = self.split_at
            sz1 = out.shape[self.split_axis] - sz0
        else:
            sz0 = out.shape[self.split_axis] + self.split_at
            sz1 = -self.split_at
        split_size = T.stack([sz0, sz1]).flatten()
        out0trn, out1trn = T.split(out, split_size, n_splits=2, axis=self.split_axis)
        if not self.keep_dim and single:
            newshape = T.concatenate([out0trn.shape[:self.split_axis], out0trn.shape[self.split_axis+1:]])
            if self.split_at == 1:
                out0trn = T.reshape(out0trn, newshape, ndim=out0trn.ndim-1)
            else:
                out1trn = T.reshape(out1trn, newshape, ndim=out1trn.ndim-1)

        out = self.input_layer.get_output(train=False)
        if self.split_at >= 0:
            sz0 = self.split_at
            sz1 = out.shape[self.split_axis] - sz0
        else:
            sz0 = out.shape[self.split_axis] + self.split_at
            sz1 = -self.split_at
        split_size = T.stack([sz0, sz1]).flatten()
        out0tst, out1tst = T.split(out, split_size, n_splits=2, axis=self.split_axis)
        if not self.keep_dim and single:
            newshape = T.concatenate([out0tst.shape[:self.split_axis], out0tst.shape[self.split_axis+1:]])
            if self.split_at == 1:
                out0tst = T.reshape(out0tst, newshape, ndim=out0tst.ndim-1)
            else:
                out1tst = T.reshape(out1tst, newshape, ndim=out1tst.ndim-1)
        # out0tst = T.addbroadcast(out0tst, *self.broadcastable_axes[0])
        # out1tst = T.addbroadcast(out1tst, *self.broadcastable_axes[1])

        out0 = {True: out0trn, False: out0tst}
        out1 = {True: out1trn, False: out1tst}

        self.__output_slots = (out0, out1)
        return out0, out1


class PartialSoftmax(Dense, MultiInputLayer):
    def __init__(self, input_dim, output_dim, init='glorot_uniform', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):
        MultiInputLayer.__init__(self, slot_names=['idxes', 'features'])
        Dense.__init__(self, input_dim, output_dim, init=init, weights=weights, name=name, W_regularizer=W_regularizer,
                       b_regularizer=b_regularizer, activity_regularizer=activity_regularizer,
                       W_constraint=W_constraint, b_constraint=b_constraint)

    def get_input(self, train=False):
        return dict((name, layer.get_output(train)) for name, layer in zip(self.input_layer_names, self.input_layers))

    def get_output(self, train=False):
        ins = self.get_input(train)
        idxes = ins['idxes']
        features = ins['features']
        weights = self.W.T.take(idxes, axis=0)
        bias = self.b.T.take(idxes, axis=0)
        return T.exp(T.sum(weights * features, axis=-1) + bias)


class LookupProb(Layer):
    def __init__(self, table):
        super(LookupProb, self).__init__()
        self.table = table

    def get_output(self, train=False):
        idxes = self.get_input(train)
        return self.table[idxes]


class TableSampler(rv_discrete):
    def __init__(self, table):
        nk = np.arange(len(table))
        super(TableSampler, self).__init__(b=len(table)-1, values=(nk, table))

    def sample(self, shape, dtype='int32'):
        return self.rvs(size=shape).astype(dtype)


class NCELangModel(Graph):
    def __init__(self, vocab_size, nb_negative, embed_dims=128, negprob_table=None, optimizer='adam'):
        super(NCELangModel, self).__init__(weighted_inputs=False)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.optimizer = optimizers.get(optimizer)
        self.nb_negative = nb_negative

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
        seq.add(LangLSTMLayer(embed_dims, output_dim=128))
        # seq.add(Dropout(0.5))

        self.add_node(seq, name='seq')
        self.add_node(PartialSoftmax(input_dim=128, output_dim=vocab_size),
                      name='part_prob', inputs=('idxes', 'seq'))
        self.add_node(LookupProb(negprob_table), name='lookup_prob', inputs='idxes')

        test_node = Dense(input_dim=128, output_dim=vocab_size, activation='softmax')
        test_node.params = []
        test_node.W = self.nodes['part_prob'].W
        test_node.b = self.nodes['part_prob'].b
        self.add_node(test_node, name='true_prob', inputs='seq')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')

        # TODO: this is memory inefficiency for larg vocabulary
        self.word_labels = theano.shared(np.eye(vocab_size, dtype='int32'), borrow=True)

    @staticmethod
    def encode_length(y_true, y_pred, mask=None):
        # probs_ = T.sum(y_true * y_pred, axis=-1)
        probs_ = y_pred[y_true.nonzero()]

        if mask is None:
            nb_words = y_true.shape[0] * y_true.shape[1]
            probs = probs_.ravel() + 1.0e-37
        else:
            nb_words = mask.sum()
            probs = T.reshape(probs_, mask.shape)[mask.nonzero()] + 1.0e-37

        return T.sum(T.log(1.0/probs)), nb_words

    # noinspection PyMethodOverriding
    def compile(self, optimizer=None, theano_mode=None):
        if optimizer is not None:
            logger.info('compiling with %s' % optimizer)
            self.optimizer = optimizers.get(optimizer)

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

        true_labels = self.word_labels[self.inputs['idxes'].get_output()[0]]
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        self._train = theano.function([self.inputs['idxes'].get_output(True)], outputs=[train_loss],
                                      updates=updates, mode=theano_mode)
        self._test = theano.function([self.inputs['idxes'].get_output(False)],
                                     outputs=[test_loss, encode_len, nb_words], mode=theano_mode)

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

    def prepare_input(self, X, validation_split, validation_data):
        if validation_data:
            ins = X
            val_ins = validation_data
        elif 0 < validation_split < 1:
            split_at = max(int(len(X) * (1 - validation_split)), 1)
            ins, val_ins = X[:split_at], X[split_at:]
        else:
            ins = X
            val_ins = None

        ins = self.negative_sample(ins)
        if val_ins is not None:
            val_ins = val_ins[np.newaxis, ...]

        return [ins], [val_ins]

    def train(self, X, callbacks, show_metrics, batch_size=128, extra_callbacks=None,
              validation_split=0., validation_data=None, shuffle=False, verbose=1):

        val_f = None
        ins, val_ins = self.prepare_input(X, validation_split, validation_data)

        if val_ins[0] is not None:
            val_f = self._test
        f = self._train
        return self._fit(f, ins, callbacks, val_f=val_f, val_ins=val_ins, metrics=show_metrics,
                         batch_size=batch_size, nb_epoch=1, extra_callbacks=extra_callbacks,
                         shuffle=shuffle, verbose=verbose)

    def train_from_dir(self, dir_, trn_regex=re.compile(r'\d{3}.bz2'), tst_regex=re.compile(r'test.bz2'),
                       callbacks=LangHistory(), show_metrics=('loss', 'ppl'),
                       extra_callbacks=(LangModelLogger(), ), chunk_size=35000, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if trn_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]
        # test_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if tst_regex.match(f)]
        # test_files = [f for f in test_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            nb_samples = X.shape[0]
            logger.debug('%d samples loaded' % nb_samples)
            logger.info('Training on %s' % f)
            chunks = make_batches(nb_samples, chunk_size)
            nb_chunks = len(chunks)
            for chunk_id, (batch_start, batch_end) in enumerate(chunks):
                data = slice_X([X], batch_start, batch_end, axis=0)[0]
                print 'Chunk %d/%d' % (chunk_id+1, nb_chunks)
                self.train(data, callbacks, show_metrics, extra_callbacks=extra_callbacks, **kwargs)

    def _fit(self, f, ins, callbacks, val_f=None, val_ins=None, metrics=(),
             batch_size=128, nb_epoch=100, extra_callbacks=(), shuffle=True, verbose=1):
        """
            Abstract fit function for f(*ins). Assume that f returns a list, labelled by out_labels.
        """
        if f.n_returned_outputs == 0:
            raise ValueError('We can not evaluate the outputs with none outputs')

        # standardize_outputs = lambda outputs: [outputs] if f.n_returned_outputs == 1 else outputs
        extra_callbacks = list(extra_callbacks)
        nb_train_sample = ins[0].shape[1]  # shape: (k+1, ns, nt)

        # logger.debug('out_labels: %s' % str(f.out_labels))

        do_validation = False
        if val_f and val_ins:
            do_validation = True
            nb_val_samples = val_ins[0].shape[1]
            pre_train_info = "Train on %d samples, validate on %d samples" % (nb_train_sample, nb_val_samples)
        else:
            pre_train_info = "Train on %d samples." % nb_train_sample

        if verbose:
            logger.info(pre_train_info)

        index_array = np.arange(nb_train_sample)
        #  TODO: any good idea to have history as mandatory callback?
        # There is problems for setting history as mandatory callback, for not all metrics are calculated
        # as the way in the History class. So I deleted this function for now and ask the user to define
        # what the callback is.
        # history = cbks.History()
        # callbacks = [history, cbks.BaseLogger()] + callbacks if verbose else [history] + callbacks
        callbacks_ = callbacks
        callbacks = cbks.CallbackList([callbacks_] + extra_callbacks)

        metrics_ = ['val_'+x for x in metrics] + list(metrics)
        cndt_metrics = [m for m in self.all_metrics if m in metrics_]

        callbacks.set_model(self)
        callbacks.set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': list(cndt_metrics),
        })
        callbacks.on_train_begin()

        self.stop_training = False
        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            epoch_logs = {}
            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    ins_batch = slice_X(ins, batch_ids)
                except TypeError:
                    print('TypeError while preparing batch. \
                        If using HDF5 input data, pass shuffle="batch".\n')
                    raise

                batch_logs = {'batch': batch_index, 'size': len(batch_ids)}
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(*ins_batch)
                _logs = [(label, value) for label, value in zip(f.out_labels, outs)]
                batch_logs.update(_logs)
                callbacks.on_batch_end(batch_index, batch_logs)

                if batch_index == len(batches) - 1:  # last batch
                    # validation
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = self._test_loop(val_f, val_ins, batch_size=batch_size, verbose=0)
                        # val_outs = standardize_outputs(val_outs)
                        _logs = [('val_'+label, value) for label, value in zip(val_f.out_labels, val_outs)]
                        epoch_logs.update(_logs)
                        # logger.debug('\nEpoch logs: %s\n' % str(epoch_logs))

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return callbacks_

    @staticmethod
    def _test_loop(f, ins, batch_size=128, verbose=0):
        """
            Abstract method to loop over some data in batches.
        """
        progbar = None
        nb_sample = ins[0].shape[1]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(*ins_batch)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(len(batch_ids))

            if verbose == 1:
                progbar.update(batch_end)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class PartialSoftmaxV1(Dense, MultiInputLayer):
    def __init__(self, input_dim, output_dim, init='glorot_uniform', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):
        MultiInputLayer.__init__(self, slot_names=['unique_idxes', 'poses', 'features'])
        Dense.__init__(self, input_dim, output_dim, init=init, weights=weights, name=name, W_regularizer=W_regularizer,
                       b_regularizer=b_regularizer, activity_regularizer=activity_regularizer,
                       W_constraint=W_constraint, b_constraint=b_constraint)

    def get_input(self, train=False):
        return dict((name, layer.get_output(train)) for name, layer in zip(self.input_layer_names, self.input_layers))

    def get_output(self, train=False):
        ins = self.get_input(train)
        idxes = ins['unique_idxes']
        poses = ins['poses']
        features = ins['features']
        weights_ = self.W.T.take(idxes, axis=0)
        bias_ = self.b.T.take(idxes, axis=0)
        weights = weights_[poses]
        bias = bias_[poses]
        return T.exp(T.sum(weights * features, axis=-1) + bias)


class NCELangModelV1(Graph):
    def __init__(self, vocab_size, nb_negative, embed_dims=128, negprob_table=None, optimizer='adam'):
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
        seq.add(LangLSTMLayer(embed_dims, output_dim=128))
        # seq.add(Dropout(0.5))

        self.add_node(seq, name='seq')

        self.add_node(PartialSoftmaxV1(input_dim=128, output_dim=vocab_size),
                      name='part_prob', inputs=('unique_idxes', 'poses', 'seq'))

        test_node = Dense(input_dim=128, output_dim=vocab_size, activation='softmax')
        test_node.params = []
        test_node.W = self.nodes['part_prob'].W
        test_node.b = self.nodes['part_prob'].b
        self.add_node(test_node, name='true_prob', inputs='seq')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')

        # self.word_labels = theano.shared(np.eye(vocab_size, dtype='int32'), borrow=True)

    @staticmethod
    def encode_length(y_label, y_pred, mask=None):
        # probs_ = T.sum(y_true * y_pred, axis=-1)
        # TODO: it may be very slow when the vocabulary is very large.
        # probs_ = y_pred[y_true.nonzero()]
        # y_label = T.flatten(y_label)
        # y_pred = T.reshape(y_pred, (-1, y_pred.shape[-1]))
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

    # noinspection PyMethodOverriding
    def compile(self, optimizer=None, theano_mode=None):
        if optimizer is not None:
            logger.info('compiling with %s' % optimizer)
            self.optimizer = optimizers.get(optimizer)

        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        # pos_prob_tst = pos_prob_layer.get_output(train=False)
        # neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)

        eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = pos_prob_trn[0].size.astype(theano.config.floatX)
        sum_pos_neg_trn = pos_prob_trn + neg_prob_trn
        y_train = T.sum(T.log(eps + pos_prob_trn[0] / sum_pos_neg_trn[0])) / nb_words
        y_train += T.sum(T.log(eps + neg_prob_trn[1:] / sum_pos_neg_trn[1:])) / nb_words
        # y_test = T.sum(T.log(eps + pos_prob_tst[0] / (pos_prob_tst[0] + neg_prob_tst[0]))) / nb_words
        # y_test += T.sum(T.log(eps + neg_prob_tst[1:] / (pos_prob_tst[1:] + neg_prob_tst[1:]))) / nb_words

        true_labels = self.inputs['pos_sents'].get_output()
        encode_len, nb_words = self.encode_length(true_labels, pre_prob_tst)

        train_loss = -y_train
        # test_loss = -y_test
        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        train_inputs = [self.inputs['pos_sents'].get_output(True),
                        self.inputs['lookup_prob'].get_output(True),
                        self.inputs['unique_idxes'].get_output(True),
                        self.inputs['poses'].get_output(True)]
        test_inputs = [self.inputs['pos_sents'].get_output(False)]

        self._train = theano.function(train_inputs, outputs=[train_loss], updates=updates, mode=theano_mode)
        self._test = theano.function(test_inputs, outputs=[encode_len, nb_words], mode=theano_mode)

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
            encode_len, nb_words = out
            # batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            # smry_loss = np.sum(loss * batch_size)/batch_size.sum()
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            return [smry_encode_len, smry_nb_words]

        self._test.summarize_outputs = __summarize_outputs

    def negative_sample(self, X, order=0):
        if order == 0:
            ret = np.empty(shape=(self.nb_negative+1,) + X.shape, dtype=X.dtype)
            ret[0] = X
            ret[1:] = self.sampler.sample(shape=ret[1:].shape)
        else:
            raise NotImplementedError('Only support order=0 now')
        return ret

    def prepare_input(self, X, validation_split, validation_data):
        if validation_data:
            ins = X
            val_ins = validation_data
        elif 0 < validation_split < 1:
            split_at = max(int(len(X) * (1 - validation_split)), 1)
            ins, val_ins = X[:split_at], X[split_at:]
        else:
            ins = X
            val_ins = None

        neg_idxes = self.negative_sample(ins)
        neg_probs = self.neg_prob_table[neg_idxes]
        unique_idxes, indeces = np.unique(neg_idxes, return_inverse=True)
        indeces = np.reshape(indeces, neg_probs.shape)
        unique_idxes = unique_idxes.astype(X.dtype)
        indeces = indeces.astype(X.dtype)
        return [ins, neg_probs, unique_idxes, indeces], [val_ins]

    def train(self, X, callbacks, show_metrics, batch_size=128, extra_callbacks=None,
              validation_split=0., validation_data=None, verbose=1):

        val_f = None
        do_validation = False
        f = self._train

        if validation_data is not None or validation_split > 0:
            val_f = self._test
            do_validation = True
        if f.n_returned_outputs == 0:
            raise ValueError('We can not evaluate the outputs with none outputs')

        nb_val_samples = 0
        val_ins = None
        if validation_data is None and 0 < validation_split < 1:
            split_at = max(int(X.shape[0] * (1 - validation_split)), 1)
            ins, val_ins = X[:split_at], X[split_at:]
            X = ins
            nb_val_samples = val_ins.shape[0]

        nb_train_sample = X.shape[0]
        if do_validation:
            pre_train_info = "Train on %d samples, validate on %d samples" % (nb_train_sample, nb_val_samples)
        else:
            pre_train_info = "Train on %d samples." % nb_train_sample

        metrics_ = ['val_'+x for x in show_metrics] + list(show_metrics)
        cndt_metrics = [m for m in self.all_metrics if m in metrics_]

        callbacks_ = callbacks
        callbacks = cbks.CallbackList([callbacks_] + list(extra_callbacks))
        callbacks.set_model(self)
        callbacks.set_params({
            'batch_size': batch_size,
            'nb_epoch': 1,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': list(cndt_metrics),
        })
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)

        logger.info(pre_train_info)
        epoch_logs = {}
        batches = make_batches(nb_train_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            data = slice_X([X], batch_start, batch_end, axis=0)
            ins, _ = self.prepare_input(data[0], validation_split=0, validation_data=None)
            batch_logs = {'batch': batch_index, 'size': data[0].shape[0]}
            callbacks.on_batch_begin(batch_index, batch_logs)
            outs = f(*ins)
            _logs = [(label, value) for label, value in zip(f.out_labels, outs)]
            batch_logs.update(_logs)
            callbacks.on_batch_end(batch_index, batch_logs)

        # validation
        if do_validation:
            val_outs = self._test_loop(val_f, [val_ins], batch_size=batch_size, verbose=0)
            _logs = [('val_'+label, value) for label, value in zip(val_f.out_labels, val_outs)]
            epoch_logs.update(_logs)
            # logger.debug('\nEpoch logs: %s\n' % str(epoch_logs))
        callbacks.on_epoch_end(0, epoch_logs)
        callbacks.on_train_end()
        return callbacks_

    def train_from_dir(self, dir_, trn_regex=re.compile(r'\d{3}.bz2'), tst_regex=re.compile(r'test.bz2'),
                       callbacks=LangHistory(), show_metrics=('loss', 'ppl'),
                       extra_callbacks=(LangModelLogger(), ), chunk_size=35000, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if trn_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]
        # test_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if tst_regex.match(f)]
        # test_files = [f for f in test_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            nb_samples = X.shape[0]
            logger.debug('%d samples loaded' % nb_samples)
            logger.info('Training on %s' % f)
            chunks = make_batches(nb_samples, chunk_size)
            nb_chunks = len(chunks)
            for chunk_id, (batch_start, batch_end) in enumerate(chunks):
                data = slice_X([X], batch_start, batch_end, axis=0)[0]
                print 'Chunk %d/%d' % (chunk_id+1, nb_chunks)
                self.train(data, callbacks, show_metrics, extra_callbacks=extra_callbacks, **kwargs)

    @staticmethod
    def _test_loop(f, ins, batch_size=128, verbose=0):
        """
            Abstract method to loop over some data in batches.
        """
        progbar = None
        nb_sample = ins[0].shape[0]
        outs = [[] for _ in range(f.n_returned_outputs)]
        batch_info = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids, axis=0)

            batch_outs = f(*ins_batch)
            for idx, v in enumerate(batch_outs):
                outs[idx].append(v)
            batch_info.append(len(batch_ids))

            if verbose == 1:
                progbar.update(batch_end)

        outs = f.summarize_outputs(outs, batch_info)
        return outs


class TreeLogSoftmax(Embedding, MultiInputLayer):
    eps = 10e-37

    def __init__(self, input_dim, embed_dim, init='uniform', W_regularizer=None,
                 activity_regularizer=None, W_constraint=None, weights=None):
        # The order to call base __init__ functions is important
        Embedding.__init__(self, input_dim, output_dim=embed_dim, init=init,
                           W_regularizer=W_regularizer, activity_regularizer=activity_regularizer,
                           W_constraint=W_constraint, mask_zero=False, weights=weights)
        # self.b = theano.shared(np.zeros(input_dim, dtype=floatX), borrow=True)
        # self.params.append(self.b)
        MultiInputLayer.__init__(self, slot_names=('features', 'cls_idx', 'word_bitstr_mask'))

    def supports_masked_input(self):
        return True

    def get_output_mask(self, train=None):
        self.name2layer['features'].get_output_mask(train=train)

    def get_output(self, train=False):
        ins = self.get_input(train=train)
        features = ins['features']
        cls_idx = ins['cls_idx']
        word_bits_mask = ins['word_bitstr_mask']

        node_embeds = self.W[cls_idx]                           # (n_s, n_t, n_n, d_l)
        # node_bias = self.b[cls_idx]                           # (n_s, n_t, n_n)
        features = features.dimshuffle(0, 1, 'x', 2)            # (n_s, n_t, 1,   d_l)
        # score = T.sum(features * node_embeds, axis=-1) + node_bias         # (n_s, n_t, n_n)
        score = T.sum(features * node_embeds, axis=-1)          # (n_s, n_t, n_n)
        prob_ = T.nnet.sigmoid(score * word_bits_mask)          # (n_s, n_t, n_n)
        prob = T.switch(T.eq(word_bits_mask, 0.0), 1.0, prob_)  # (n_s, n_t, n_n)
        log_prob = T.sum(T.log(self.eps+prob), axis=-1)         # (n_s, n_t)
        return log_prob

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "input_slot_names": self.input_layer_names,
                "init": self.init.__name__,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None}


class NotLinkedError(Exception):
    def __init__(self, message):
        super(NotLinkedError, self).__init__(message)
        # self.message = message


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
    def compile(self, optimizer=None, theano_mode=None):
        if optimizer is not None:
            logger.info('compiling with %s' % optimizer)
            self.optimizer = optimizers.get(optimizer)

        logprob_layer = self.outputs['tree_softmax']
        logprob_trn = logprob_layer.get_output(train=True)
        logprob_tst = logprob_layer.get_output(train=False)

        # eps = 1.0e-37
        #TODO: mask not supported here
        nb_words = logprob_trn.size.astype(theano.config.floatX)
        train_loss = -T.sum(logprob_trn) / nb_words
        test_loss = -T.sum(logprob_tst) / nb_words

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

        self._train = theano.function(train_inputs, outputs=[train_loss], updates=updates, mode=theano_mode)
        self._test = theano.function(test_inputs, outputs=[test_loss, encode_len, nb_words], mode=theano_mode)

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

    def prepare_input(self, X, validation_split, validation_data):
        ins = [None, None, None]
        val_ins = [None, None, None]
        if validation_data:
            ins[0] = X
            val_ins[0] = validation_data
        elif 0 < validation_split < 1:
            split_at = max(int(X.shape[0] * (1 - validation_split)), 1)
            ins[0], val_ins[0] = X[:split_at], X[split_at:]
        else:
            ins[0] = X
            val_ins[0] = None

        ins[1] = self.word2class[ins[0]]
        ins[2] = self.word2bitstr[ins[0]]
        if val_ins[0] is not None:
            val_ins[1] = self.word2class[val_ins[0]]
            val_ins[2] = self.word2bitstr[val_ins[0]]

        return ins, val_ins

    def train(self, X, callbacks, show_metrics, batch_size=128, extra_callbacks=None,
              validation_split=0., validation_data=None, shuffle=False, verbose=1, **kwargs):

        val_f = None
        ins, val_ins = self.prepare_input(X, validation_split, validation_data)

        if val_ins[0] is not None:
            val_f = self._test
        else:
            val_ins = None

        f = self._train
        return self._fit(f, ins, callbacks, val_f=val_f, val_ins=val_ins, metrics=show_metrics,
                         batch_size=batch_size, nb_epoch=1, extra_callbacks=extra_callbacks,
                         shuffle=shuffle, verbose=verbose)

    def train_from_dir(self, dir_, trn_regex=re.compile(r'\d{3}.bz2'), tst_regex=re.compile(r'test.bz2'),
                       callbacks=LangHistory(), show_metrics=('loss', 'ppl'),
                       extra_callbacks=(LangModelLogger(), ), chunk_size=35000, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if trn_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]
        # test_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if tst_regex.match(f)]
        # test_files = [f for f in test_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            nb_samples = X.shape[0]
            logger.debug('%d samples loaded' % nb_samples)
            logger.info('Training on %s' % f)
            chunks = make_batches(nb_samples, chunk_size)
            nb_chunks = len(chunks)
            for chunk_id, (batch_start, batch_end) in enumerate(chunks):
                data = slice_X([X], batch_start, batch_end, axis=0)[0]
                print 'Chunk %d/%d' % (chunk_id+1, nb_chunks)
                self.train(data, callbacks, show_metrics, extra_callbacks=extra_callbacks, **kwargs)


class LBLayer(Layer):
    def __init__(self, context_size, embed_dim, init='glorot_uniform', weights=None, name=None,
                 W_regularizer=None, activity_regularizer=None, W_constraint=None):
        super(LBLayer, self).__init__()
        self.context_size = context_size
        self.embed_dim = embed_dim
        self.init = initializations.get(init)
        # self.tidx = theano.shared(np.arange(max_sent_len).reshape((max_sent_len, 1)).astype('int16') +
        #                           np.arange(context_size).reshape((1, context_size)).astype('int16'), borrow=True)

        W = np.empty(shape=(context_size, embed_dim, embed_dim), dtype=floatX)
        for i in range(context_size):
            W[i] = self.init((embed_dim, embed_dim), device='tmp').get_value(borrow=True)
        self.W = theano.shared(W, name='cntx_w', borrow=True)
        self.pad = theano.shared(np.zeros((1, self.context_size, self.context_size, self.embed_dim), dtype=floatX),
                                 borrow=True)

        self.params = [self.W, self.pad]
        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

    def get_output(self, train=False):
        ins = self.get_input(train=train)
        ns = ins.shape[0]
        nt = ins.shape[1]

        y = T.dot(ins, self.W)
        x = T.repeat(self.pad, ns, axis=0)
        z = T.concatenate([x, y], axis=1)

        sidx = T.arange(ns, dtype='int32').dimshuffle(0, 'x', 'x')
        tidx = T.arange(nt+1, dtype='int32').dimshuffle(0, 'x') + \
               T.arange(self.context_size, dtype='int32').dimshuffle('x', 0)
        tidx = tidx.dimshuffle('x', 0, 1)
        cidx = T.arange(self.context_size-1, -1, -1, dtype='int32').dimshuffle('x', 'x', 0)

        d = z[sidx, tidx, cidx]
        c = T.sum(d, axis=2)
        return c

    def supports_masked_input(self):
        return None


class LBLScore(MultiInputLayer):
    def __init__(self, vocab_size):
        super(LBLScore, self).__init__(slot_names=('context', 'word'))
        self.vocab_size = vocab_size
        # self.b = T.zeros((vocab_size, 1), dtype=floatX)
        self.b = theano.shared(np.zeros((vocab_size, 1), dtype=floatX), borrow=True)
        self.params = [self.b]

    def get_output(self, train=False):
        ins = self.get_input(train)
        cntxt_vec = ins['context']
        wrd_vec = ins['word'].dimshuffle(0, 1, 'x')
        prob_ = T.exp(T.dot(cntxt_vec, wrd_vec) + self.b)
        prob_ = T.addbroadcast(prob_, 3)
        prob_ = prob_.dimshuffle(0, 1, 2)
        prob = prob_/T.sum(prob_, axis=-1, keepdims=True)

        return prob


class EmbeddingParam(Layer):
    def __init__(self):
        super(EmbeddingParam, self).__init__()

    def get_output(self, train=False):
        return self.previous.params[0]

    def get_input(self, train=False):
        return self.previous.params[0]


class LBLangModelV0(Graph):
    def __init__(self, vocab_size, context_size, embed_dims=128,
                 loss='categorical_crossentropy', optimizer='adam'):
        super(LBLangModelV0, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.loss = objectives.get(loss)
        self.loss_fnc = objective_fnc(self.loss)
        self.optimizer = optimizers.get(optimizer)
        self.context_size = context_size
        # self.max_sent_len = max_sent_len

        self.add_input(name='idxes', ndim=2, dtype='int32')
        self.add_node(Split(split_at=-1, split_axis=1), name=('idxes_cnt', ''), inputs='idxes')
        self.add_node(Embedding(vocab_size, embed_dims), name='embedding', inputs='idxes_cnt')
        self.add_node(EmbeddingParam(), name='embedding_param', inputs='embedding')
        self.add_node(LBLayer(context_size, embed_dims), name='context_vec', inputs='embedding')
        self.add_node(LBLScore(vocab_size), 'score', inputs=('context_vec', 'embedding_param'))

        self.add_output('prob', 'score')

    @staticmethod
    def encode_length(y_true, y_pred, mask):
        # probs_ = T.sum(y_true * y_pred, axis=-1)
        probs_ = y_pred[y_true.nonzero()]

        if mask is None:
            nb_words = y_true.shape[0] * y_true.shape[1]
            probs = probs_.ravel() + 1.0e-30
        else:
            nb_words = mask.sum()
            probs = T.reshape(probs_, mask.shape)[mask.nonzero()] + 1.0e-37

        return T.sum(T.log(1.0/probs)), nb_words

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
        self.y = T.zeros_like(self.y_train)

        self.weights = None

        # if hasattr(self.layers[-1], "get_output_mask"):
        #     mask = self.layers[-1].get_output_mask()
        # else:
        #     mask = None
        # todo: mask support
        mask = None
        train_loss = self.loss_fnc(self.y, self.y_train, mask)
        test_loss = self.loss_fnc(self.y, self.y_test, mask)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        self.y.name = 'y'

        # train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)),
        #                         dtype=theano.config.floatX)
        # test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)),
        #                        dtype=theano.config.floatX)

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
        # predict_ins = [self.X_test]

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

    def train(self, X, y, callbacks, show_metrics, batch_size=128, extra_callbacks=(LangModelLogger(), ),
              validation_split=0., validation_data=None, shuffle=False, verbose=1):
        data = {'idxes': X, 'prob': y}
        self.fit(data, callbacks, show_metrics, batch_size=batch_size, nb_epoch=1, verbose=verbose,
                 extra_callbacks=extra_callbacks, validation_split=validation_split,
                 validation_data=validation_data, shuffle=shuffle)

    def train_from_dir(self, dir_, data_regex=re.compile(r'\d{3}.bz2'), callbacks=LangHistory(),
                       show_metrics=('loss', 'ppl'), *args, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if data_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            # y = np.zeros((X.shape[0], X.shape[1], self.vocab_size), dtype=np.int8)
            tmp = np.eye(self.vocab_size, dtype='int8')
            y = tmp[X]
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         idx = X[i, j]
            #         y[i, j, idx] = 1
            logger.info('Training on %s' % f)
            self.train(X, y, callbacks, show_metrics, *args, **kwargs)


class FFNNLangModelV0(Graph):
    def __init__(self, vocab_size, context_size, embed_dims=128, context_dims=128,
                 loss='categorical_crossentropy', optimizer='adam'):
        super(FFNNLangModelV0, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.loss = objectives.get(loss)
        self.loss_fnc = objective_fnc(self.loss)
        self.optimizer = optimizers.get(optimizer)
        self.context_size = context_size
        # self.max_sent_len = max_sent_len

        self.add_input(name='idxes', ndim=2, dtype='int32')
        self.add_node(Split(split_at=-1, split_axis=1), name=('idxes_cnt', ''), inputs='idxes')
        self.add_node(Embedding(vocab_size, embed_dims), name='embedding', inputs='idxes_cnt')
        # self.add_node(EmbeddingParam(), name='embedding_param', inputs='embedding')
        self.add_node(LBLayer(context_size, context_dims), name='context_vec', inputs='embedding')
        self.add_node(Dense(context_dims, vocab_size, activation='softmax'), name='score',
                      inputs='context_vec')

        self.add_output('prob', 'score')

    @staticmethod
    def encode_length(y_true, y_pred, mask):
        # probs_ = T.sum(y_true * y_pred, axis=-1)
        probs_ = y_pred[y_true.nonzero()]

        if mask is None:
            nb_words = y_true.shape[0] * y_true.shape[1]
            probs = probs_.ravel() + 1.0e-30
        else:
            nb_words = mask.sum()
            probs = T.reshape(probs_, mask.shape)[mask.nonzero()] + 1.0e-37

        return T.sum(T.log(1.0/probs)), nb_words

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
        self.y = T.zeros_like(self.y_train)

        self.weights = None

        # if hasattr(self.layers[-1], "get_output_mask"):
        #     mask = self.layers[-1].get_output_mask()
        # else:
        #     mask = None
        # todo: mask support
        mask = None
        train_loss = self.loss_fnc(self.y, self.y_train, mask)
        test_loss = self.loss_fnc(self.y, self.y_test, mask)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        self.y.name = 'y'

        # train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)),
        #                         dtype=theano.config.floatX)
        # test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)),
        #                        dtype=theano.config.floatX)

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
        # predict_ins = [self.X_test]

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

    def train(self, X, y, callbacks, show_metrics, batch_size=128, extra_callbacks=(LangModelLogger(), ),
              validation_split=0., validation_data=None, shuffle=False, verbose=1):
        data = {'idxes': X, 'prob': y}
        self.fit(data, callbacks, show_metrics, batch_size=batch_size, nb_epoch=1, verbose=verbose,
                 extra_callbacks=extra_callbacks, validation_split=validation_split,
                 validation_data=validation_data, shuffle=shuffle)

    def train_from_dir(self, dir_, data_regex=re.compile(r'\d{3}.bz2'), callbacks=LangHistory(),
                       show_metrics=('loss', 'ppl'), *args, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if data_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            # y = np.zeros((X.shape[0], X.shape[1], self.vocab_size), dtype=np.int8)
            tmp = np.eye(self.vocab_size, dtype='int8')
            y = tmp[X]
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         idx = X[i, j]
            #         y[i, j, idx] = 1
            logger.info('Training on %s' % f)
            self.train(X, y, callbacks, show_metrics, *args, **kwargs)


class LBLScoreV1(MultiInputLayer):
    def __init__(self, vocab_size):
        super(LBLScoreV1, self).__init__(slot_names=('context', 'word'))
        self.vocab_size = vocab_size
        # self.b = T.zeros((vocab_size, 1), dtype=floatX)
        self.b = theano.shared(np.zeros((vocab_size, 1), dtype=floatX), borrow=True)
        self.params = [self.b]

    def get_output(self, train=False):
        ins = self.get_input(train)
        cntxt_vec = ins['context']
        wrd_vec = ins['word'][:self.vocab_size].dimshuffle(0, 1, 'x')
        prob_ = T.exp(T.dot(cntxt_vec, wrd_vec) + self.b)
        prob_ = T.addbroadcast(prob_, 2)
        prob_ = prob_.dimshuffle(0, 1)
        prob = prob_/T.sum(prob_, axis=-1, keepdims=True)

        return prob


class LBLangModelV1(Graph):
    def __init__(self, vocab_size, context_size, embed_dims=128,
                 loss='categorical_crossentropy', optimizer='adam'):
        super(LBLangModelV1, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.loss = objectives.get(loss)
        self.loss_fnc = objective_fnc(self.loss)
        self.optimizer = optimizers.get(optimizer)
        self.context_size = context_size
        # self.max_sent_len = max_sent_len

        self.add_input(name='ngrams', ndim=2, dtype='int32')

        self.add_node(Embedding(vocab_size+context_size, embed_dims), name='embedding', inputs='ngrams')
        self.add_node(EmbeddingParam(), name='embedding_param', inputs='embedding')
        self.add_node(Reshape(-1), name='reshape', inputs='embedding')
        composer_node = Dense(context_size*embed_dims, embed_dims)
        composer_node.params = [composer_node.W]   # drop the bias parameters
        # del composer_node.b
        # replace the default behavior of Dense
        composer_node.get_output = lambda train: node_get_output(composer_node, train)
        self.add_node(composer_node, name='context_vec', inputs='reshape')
        self.add_node(LBLScoreV1(vocab_size), name='score', inputs=('context_vec', 'embedding_param'))

        self.add_output('prob', 'score')

        def node_get_output(layer, train=False):
            X = layer.get_input(train)
            output = layer.activation(T.dot(X, layer.W))
            return output

    @staticmethod
    def encode_length(y_true, y_pred, mask=None):
        # probs_ = T.sum(y_true * y_pred, axis=-1)
        probs_ = y_pred[y_true.nonzero()]

        nb_words = y_true.shape[0]
        probs = probs_.ravel() + 1.0e-30

        return T.sum(T.log(1.0/probs)), nb_words

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
        self.y = T.zeros_like(self.y_train)

        self.weights = None

        # if hasattr(self.layers[-1], "get_output_mask"):
        #     mask = self.layers[-1].get_output_mask()
        # else:
        #     mask = None
        # todo: mask support
        mask = None
        train_loss = self.loss_fnc(self.y, self.y_train, mask)
        test_loss = self.loss_fnc(self.y, self.y_test, mask)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        self.y.name = 'y'

        # train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)),
        #                         dtype=theano.config.floatX)
        # test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)),
        #                        dtype=theano.config.floatX)

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
        # predict_ins = [self.X_test]

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

    def train(self, X, y, callbacks, show_metrics, batch_size=128, extra_callbacks=(LangModelLogger(), ),
              validation_split=0., validation_data=None, shuffle=False, verbose=1):
        data = {'ngrams': X, 'prob': y}
        self.fit(data, callbacks, show_metrics, batch_size=batch_size, nb_epoch=1, verbose=verbose,
                 extra_callbacks=extra_callbacks, validation_split=validation_split,
                 validation_data=validation_data, shuffle=shuffle)

    def train_from_dir(self, dir_, data_regex=re.compile(r'\d{3}.bz2'), callbacks=LangHistory(),
                       show_metrics=('loss', 'ppl'), *args, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if data_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            # y = np.zeros((X.shape[0], X.shape[1], self.vocab_size), dtype=np.int8)
            pad_idx = np.arange(self.vocab_size, self.vocab_size+self.context_size).reshape((1, -1))
            pad_idx = pad_idx.repeat(X.shape[0], axis=0)
            idxes = np.hstack((pad_idx, X))

            ns = X.shape[0]
            nt = X.shape[1]
            nb_ele = X.size
            X = np.empty(shape=(nb_ele, self.context_size), dtype='int32')
            y_label = np.empty(shape=(nb_ele, ), dtype='int32')
            start_end = np.array([0, 0], dtype='int32')
            k = 0
            for i in range(ns):
                start_end[0], start_end[1] = 0, self.context_size
                for j in range(nt):
                    X[k] = idxes[i, start_end[0]:start_end[1]]
                    y_label[k] = idxes[i, start_end[1]]
                    k += 1
                    start_end += 1

            tmp = np.eye(self.vocab_size, dtype='int8')
            y = tmp[y_label]
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         idx = X[i, j]
            #         y[i, j, idx] = 1
            logger.info('Training on %s' % f)
            self.train(X, y, callbacks, show_metrics, *args, **kwargs)


class FFNNLangModelV1(Graph):
    def __init__(self, vocab_size, context_size, embed_dims=128, context_dim=128,
                 loss='categorical_crossentropy', optimizer='adam'):
        super(FFNNLangModelV1, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims
        self.loss = objectives.get(loss)
        self.loss_fnc = objective_fnc(self.loss)
        self.optimizer = optimizers.get(optimizer)
        self.context_size = context_size
        # self.max_sent_len = max_sent_len

        self.add_input(name='ngrams', ndim=2, dtype='int32')

        self.add_node(Embedding(vocab_size+context_size, embed_dims), name='embedding', inputs='ngrams')
        # self.add_node(EmbeddingParam(), name='embedding_param', inputs='embedding')
        self.add_node(Reshape(-1), name='reshape', inputs='embedding')
        composer_node = Dense(context_size*embed_dims, context_dim)
        composer_node.params = [composer_node.W]   # drop the bias parameters
        # del composer_node.b
        # replace the default behavior of Dense
        composer_node.get_output = lambda train: node_get_output(composer_node, train)
        self.add_node(composer_node, name='context_vec', inputs='reshape')
        # self.add_node(Dropout(0.5), name='dropout', inputs='context_vec')
        self.add_node(Dense(context_dim, vocab_size, activation='softmax'), name='score',
                      inputs='context_vec')

        self.add_output('prob', 'score')

        def node_get_output(layer, train=False):
            X = layer.get_input(train)
            output = layer.activation(T.dot(X, layer.W))
            return output

    @staticmethod
    def encode_length(y_true, y_pred, mask=None):
        # probs_ = T.sum(y_true * y_pred, axis=-1)
        probs_ = y_pred[y_true.nonzero()]

        nb_words = y_true.shape[0]
        probs = probs_.ravel() + 1.0e-30

        return T.sum(T.log(1.0/probs)), nb_words

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
        self.y = T.zeros_like(self.y_train)

        self.weights = None

        # if hasattr(self.layers[-1], "get_output_mask"):
        #     mask = self.layers[-1].get_output_mask()
        # else:
        #     mask = None
        # todo: mask support
        mask = None
        train_loss = self.loss_fnc(self.y, self.y_train, mask)
        test_loss = self.loss_fnc(self.y, self.y_test, mask)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        self.y.name = 'y'

        # train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)),
        #                         dtype=theano.config.floatX)
        # test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)),
        #                        dtype=theano.config.floatX)

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
        # predict_ins = [self.X_test]

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

    def train(self, X, y, callbacks, show_metrics, batch_size=128, extra_callbacks=(LangModelLogger(), ),
              validation_split=0., validation_data=None, shuffle=False, verbose=1):
        data = {'ngrams': X, 'prob': y}
        self.fit(data, callbacks, show_metrics, batch_size=batch_size, nb_epoch=1, verbose=verbose,
                 extra_callbacks=extra_callbacks, validation_split=validation_split,
                 validation_data=validation_data, shuffle=shuffle)

    def train_from_dir(self, dir_, data_regex=re.compile(r'\d{3}.bz2'), callbacks=LangHistory(),
                       show_metrics=('loss', 'ppl'), *args, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if data_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            # y = np.zeros((X.shape[0], X.shape[1], self.vocab_size), dtype=np.int8)
            pad_idx = np.arange(self.vocab_size, self.vocab_size+self.context_size).reshape((1, -1))
            pad_idx = pad_idx.repeat(X.shape[0], axis=0)
            idxes = np.hstack((pad_idx, X))

            ns = X.shape[0]
            nt = X.shape[1]
            nb_ele = X.size
            X = np.empty(shape=(nb_ele, self.context_size), dtype='int32')
            y_label = np.empty(shape=(nb_ele, ), dtype='int32')
            start_end = np.array([0, 0], dtype='int32')
            k = 0
            for i in range(ns):
                start_end[0], start_end[1] = 0, self.context_size
                for j in range(nt):
                    X[k] = idxes[i, start_end[0]:start_end[1]]
                    y_label[k] = idxes[i, start_end[1]]
                    k += 1
                    start_end += 1

            tmp = np.eye(self.vocab_size, dtype='int8')
            y = tmp[y_label]
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         idx = X[i, j]
            #         y[i, j, idx] = 1
            logger.info('Training on %s' % f)
            self.train(X, y, callbacks, show_metrics, *args, **kwargs)


class SimpAttLayer(Recurrent):
    """ Recurrent Layer with simple attention mechanics implemented.
    """

    def __init__(self, input_dim, output_dim=128, train_init_cell=True, train_init_h=True,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 input_activation='tanh', gate_activation='hard_sigmoid', output_activation='tanh',
                 weights=None, truncate_gradient=-1, attention_len=10):

        super(SimpAttLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)
        self.input = T.tensor3()
        self.time_range = None
        self.attention_len = attention_len

        W_z = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_z = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_z = shared_zeros(self.output_dim)

        W_i = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_i = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_i = shared_zeros(self.output_dim)

        W_f = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_f = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_f = self.forget_bias_init(self.output_dim)

        W_o = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_o = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_o = shared_zeros(self.output_dim)

        self.W_s = self.init((self.output_dim, self.input_dim))
        self.b_s = theano.shared(np.zeros(shape=(self.input_dim,), dtype=floatX), name='bias_trans', borrow=True)

        self.h_m1 = shared_zeros(shape=(1, self.output_dim), name='h0')
        self.c_m1 = shared_zeros(shape=(1, self.output_dim), name='c0')

        W = np.vstack((W_z[np.newaxis, :, :],
                       W_i[np.newaxis, :, :],
                       W_f[np.newaxis, :, :],
                       W_o[np.newaxis, :, :]))  # shape = (4, input_dim, output_dim)
        R = np.vstack((R_z[np.newaxis, :, :],
                       R_i[np.newaxis, :, :],
                       R_f[np.newaxis, :, :],
                       R_o[np.newaxis, :, :]))  # shape = (4, output_dim, output_dim)
        self.W = theano.shared(W, name='Input to hidden weights (zifo)', borrow=True)
        self.R = theano.shared(R, name='Recurrent weights (zifo)', borrow=True)
        self.b = theano.shared(np.zeros(shape=(4, self.output_dim), dtype=theano.config.floatX),
                               name='bias', borrow=True)
        np_b_pos = np.zeros((self.attention_len+1, ), dtype=floatX)
        np_b_pos[0] = 1.0
        self.b_pos = theano.shared(np_b_pos, name='b_pos', borrow=True)
        self.bos_vec = theano.shared(np.zeros((1, self.input_dim), dtype=floatX), name='Begin of sentence', borrow=True)

        self.params = [self.W, self.R, self.b, self.W_s, self.b_s, self.b_pos, self.bos_vec]
        if train_init_cell:
            self.params.append(self.c_m1)
        if train_init_h:
            self.params.append(self.h_m1)

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              t,                                # sequence. t: scalar
              h_tm1, c_tm1,                     # output_info. h_tm1: (n_s, d_c), c_tm1: (n_s, d_c)
              R, W, bias, E_w, b_pos, W_s, b_s):      # non_sequence. R:(d_c, d_c), E_w:(t_max, n_s, d_e),
                                                      # W: (4, d_e, d_c)
        if t == 1:
            X_t = E_w[0]
        else:
            s_t = T.dot(h_tm1, W_s) + b_s    # (n_s, d_e)
            pos = T.arange(t-1, -1, -1, dtype='int16')    # (t, )
            pos = T.switch(pos > self.attention_len, self.attention_len, pos)    # (t, )
            embeds = E_w[:t]  # (t, n_s, d_e)
            score_t = T.sum(s_t*embeds, axis=-1).dimshuffle(1, 0) + b_pos[pos]   # (n_s, t)
            # alpha_t = T.nnet.softmax(score_t)   # (n_s, t)
            e_score = T.exp(score_t - T.max(score_t, axis=1, keepdims=True))
            alpha_t = e_score / T.sum(e_score, axis=1, keepdims=True)
            X_t = T.sum(alpha_t.dimshuffle(1, 0, 'x') * embeds, axis=0)   # (n_s, d_e)

        Y_t = T.dot(X_t, W) + bias   # (n_s, 4, d_c)
        G_tm1 = T.dot(h_tm1, R)      # (n_s, 4, d_c)
        M_t = Y_t + G_tm1            # (n_s, 4, d_c)

        z_t = self.input_activation(M_t[:, 0, :])
        ifo_t = self.gate_activation(M_t[:, 1:, :])
        i_t = ifo_t[:, 0, :]
        f_t = ifo_t[:, 1, :]
        o_t = ifo_t[:, 2, :]
        c_t = f_t * c_tm1 + i_t * z_t
        h_t = o_t * self.output_activation(c_t)
        return h_t, c_t

    def get_output_mask(self, train=None):
        return None

    def _get_output_with_mask(self, train=False):
        raise NotImplementedError('mask not supported for now')
        # X = self.get_input(train)
        # # mask = self.get_padded_shuffled_mask(train, X, pad=0)
        # mask = self.get_input_mask(train=train)
        # ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32').ravel()
        # max_time = T.max(ind) - 1   # drop the last frame
        # X = X.dimshuffle((1, 0, 2))
        # Y = T.dot(X, self.W) + self.b
        # # h0 = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        # h0 = T.repeat(self.h_m1, X.shape[1], axis=0)
        # c0 = T.repeat(self.c_m1, X.shape[1], axis=0)
        #
        # [outputs, _], updates = theano.scan(
        #     self._step,
        #     sequences=Y,
        #     outputs_info=[h0, c0],
        #     non_sequences=[self.R], n_steps=max_time,
        #     truncate_gradient=self.truncate_gradient, strict=True,
        #     allow_gc=theano.config.scan.allow_gc)
        #
        # res = T.concatenate([h0.dimshuffle('x', 0, 1), outputs], axis=0).dimshuffle((1, 0, 2))
        # return res

    def _get_output_without_mask(self, train=False):
        X = self.get_input(train)  # (n_s, n_t, d_e)
        max_time, ns = X.shape[1], X.shape[0]
        X = X[:, :-1]  # drop the last frame:          # (n_s, n_t-1, d_e)
        bos = T.repeat(self.bos_vec, ns, axis=0)       # (n_s, d_e)
        bos = T.reshape(bos, (ns, 1, self.input_dim))  # (n_s, 1, d_e)
        X = T.concatenate([bos, X], axis=1)            # (n_s, n_t, d_e)
        X = X.dimshuffle(1, 0, 2)                      # (n_t, n_s, d_e)

        h0 = T.repeat(self.h_m1, ns, axis=0)
        c0 = T.repeat(self.c_m1, ns, axis=0)

        [outputs, _], updates = theano.scan(
            self._step,
            sequences=T.arange(1, max_time+1, dtype='int16'),
            outputs_info=[h0, c0],
            # R, W, bias, E_w, b_pos, W_s, b_s)
            non_sequences=[self.R, self.W, self.b, X, self.b_pos, self.W_s, self.b_s],
            n_steps=max_time, strict=True,
            truncate_gradient=self.truncate_gradient,
            allow_gc=theano.config.scan.allow_gc)

        res = outputs.dimshuffle(1, 0, 2)
        return res

    def get_output(self, train=False):
        mask = self.get_input_mask(train=train)
        if mask is None:
            return self._get_output_without_mask(train=train)
        else:
            return self._get_output_with_mask(train=train)

    def set_init_cell_parameter(self, is_param=True):
        if is_param:
            if self.c_m1 not in self.params:
                self.params.append(self.c_m1)
        else:
            self.params.remove(self.c_m1)

    def set_init_h_parameter(self, is_param=True):
        if is_param:
            if self.h_m1 not in self.params:
                self.params.append(self.h_m1)
        else:
            self.params.remove(self.h_m1)

    def get_time_range(self, train):
        mask = self.get_input_mask(train=train)
        ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32')
        self.time_range = ind
        return ind

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "forget_bias_init": self.forget_bias_init.__name__,
                "input_activation": self.input_activation.__name__,
                "gate_activation": self.gate_activation.__name__,
                "truncate_gradient": self.truncate_gradient}


class ParallelAttLayerV0(Recurrent):
    """ Recurrent Layer with parallel attention mechanics implemented.
        This version may be more memory efficient than V1 with some restrictions.
    """

    def __init__(self, input_dim, output_dim=128, train_init_cell=True, train_init_h=True,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 input_activation='tanh', gate_activation='hard_sigmoid', output_activation='tanh',
                 weights=None, truncate_gradient=-1, attention_len=10, max_sent_len=64, max_min_batch=2048):

        super(ParallelAttLayerV0, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)
        self.input = T.tensor3()
        self.time_range = None
        self.attention_len = attention_len
        self._score_table = theano.shared(np.zeros((max_min_batch, max_sent_len), dtype=floatX), borrow=True)
        self._exp_score_table = theano.shared(np.zeros((max_min_batch, max_sent_len), dtype=floatX), borrow=True)
        self._alpha_table = theano.shared(np.zeros((max_min_batch, max_sent_len), dtype=floatX), borrow=True)
        self.max_sent_len = max_sent_len

        logger.warn('Only support mini-batch smaller than or equal to %d' % max_min_batch)

        W_z = self.init((self.input_dim, 2*self.output_dim)).get_value(borrow=True)
        R_z = self.inner_init((self.output_dim, 2*self.output_dim)).get_value(borrow=True)
        # self.b_z = shared_zeros(self.output_dim)

        W_i = self.init((self.input_dim, 2*self.output_dim)).get_value(borrow=True)
        R_i = self.inner_init((self.output_dim, 2*self.output_dim)).get_value(borrow=True)
        # self.b_i = shared_zeros(self.output_dim)

        W_f = self.init((self.input_dim, 2*self.output_dim)).get_value(borrow=True)
        R_f = self.inner_init((self.output_dim, 2*self.output_dim)).get_value(borrow=True)
        # self.b_f = self.forget_bias_init(self.output_dim)

        W_o = self.init((self.input_dim, 2*self.output_dim)).get_value(borrow=True)
        R_o = self.inner_init((self.output_dim, 2*self.output_dim)).get_value(borrow=True)
        # self.b_o = shared_zeros(self.output_dim)

        self.W_s = self.init((self.output_dim, self.input_dim))
        self.b_s = theano.shared(np.zeros(shape=(self.input_dim,), dtype=floatX), name='bias_trans', borrow=True)

        self.h_m1 = shared_zeros(shape=(1, 2*self.output_dim), name='h0')
        self.c_m1 = shared_zeros(shape=(1, 2*self.output_dim), name='c0')

        W = np.vstack((W_z[np.newaxis, :, :],
                       W_i[np.newaxis, :, :],
                       W_f[np.newaxis, :, :],
                       W_o[np.newaxis, :, :]))  # shape = (4, input_dim, 2*output_dim)
        R = np.vstack((R_z[np.newaxis, :, :],
                       R_i[np.newaxis, :, :],
                       R_f[np.newaxis, :, :],
                       R_o[np.newaxis, :, :]))  # shape = (4, output_dim, 2*output_dim)
        self.W = theano.shared(W, name='Input to hidden weights (zifo)', borrow=True)
        self.R = theano.shared(R, name='Recurrent weights (zifo)', borrow=True)
        self.b = theano.shared(np.zeros(shape=(4, 2*self.output_dim), dtype=theano.config.floatX),
                               name='bias', borrow=True)
        np_b_pos = np.zeros((self.attention_len+1, ), dtype=floatX)
        np_b_pos[0] = 1.0
        self.b_pos = theano.shared(np_b_pos, name='b_pos', borrow=True)
        self.bos_vec = theano.shared(np.zeros((1, self.input_dim), dtype=floatX), name='Begin of sentence', borrow=True)

        self.params = [self.W, self.R, self.b, self.W_s, self.b_s, self.b_pos, self.bos_vec]
        if train_init_cell:
            self.params.append(self.c_m1)
        if train_init_h:
            self.params.append(self.h_m1)

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              t,                                # sequence. t: scalar, t=1,2,...,t_max
              h_tm1, c_tm1,                     # output_info. h_tm1: (n_s, 2*d_c), c_tm1: (n_s, 2*d_c)
              R, W, bias, E_w, b_pos, W_s, b_s, score_table, exp_score_table, alpha_table):
              # non_sequence. R:(2*d_c, 2*d_c), E_w:(t_max, n_s, d_e),
              # W: (4, d_e, 2*d_c)
        if t == 1:
            X_t = E_w[0]
        else:
            s_t = T.dot(h_tm1[:, :self.output_dim], W_s) + b_s    # (n_s, d_e)
            pos = T.arange(t-1, -1, -1, dtype='int16')    # (t, )
            pos = T.switch(pos > self.attention_len, self.attention_len, pos)    # (t, )
            embeds = E_w[:t]  # (t, n_s, d_e)
            n_s = embeds.shape[1]
            # score_t = T.sum(s_t*embeds, axis=-1).dimshuffle(1, 0) + b_pos[pos]   # (n_s, t)
            s_tbl = T.set_subtensor(score_table[:n_s, :t], T.sum(s_t*embeds, axis=-1).dimshuffle(1, 0) + b_pos[pos])
                                    #inplace=True)
            # alpha_t = T.nnet.softmax(score_t)   # (n_s, t)
            # e_score = T.exp(score_t - T.max(score_t, axis=1, keepdims=True))
            exp_tbl = T.set_subtensor(exp_score_table[:n_s, :t],
                                      T.exp(s_tbl[:n_s, :t] - T.max(s_tbl[:n_s, :t], axis=1, keepdims=True)))
                                      # inplace=True)
            # alpha_t = e_score / T.sum(e_score, axis=1, keepdims=True)
            alpha_tbl = T.set_subtensor(alpha_table[:n_s, :t],
                                        exp_tbl[:n_s, :t] / T.sum(exp_tbl[:n_s, :t], axis=1, keepdims=True))
                                        # inplace=True)
            X_t = T.sum(alpha_tbl[:n_s, :t].dimshuffle(1, 0, 'x') * embeds, axis=0)   # (n_s, d_e)

        Y_t = T.dot(X_t, W) + bias   # (n_s, 4, 2*d_c)
        G_tm1 = T.dot(h_tm1[:, self.output_dim:], R)      # (n_s, 4, 2*d_c)
        M_t = Y_t + G_tm1            # (n_s, 4, 2*d_c)

        z_t = self.input_activation(M_t[:, 0, :])
        ifo_t = self.gate_activation(M_t[:, 1:, :])
        i_t = ifo_t[:, 0, :]
        f_t = ifo_t[:, 1, :]
        o_t = ifo_t[:, 2, :]
        c_t = f_t * c_tm1 + i_t * z_t
        h_t = o_t * self.output_activation(c_t)
        return h_t, c_t

    def get_output_mask(self, train=None):
        return None

    def _get_output_with_mask(self, train=False):
        raise NotImplementedError('mask not supported for now')
        # X = self.get_input(train)
        # # mask = self.get_padded_shuffled_mask(train, X, pad=0)
        # mask = self.get_input_mask(train=train)
        # ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32').ravel()
        # max_time = T.max(ind) - 1   # drop the last frame
        # X = X.dimshuffle((1, 0, 2))
        # Y = T.dot(X, self.W) + self.b
        # # h0 = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        # h0 = T.repeat(self.h_m1, X.shape[1], axis=0)
        # c0 = T.repeat(self.c_m1, X.shape[1], axis=0)
        #
        # [outputs, _], updates = theano.scan(
        #     self._step,
        #     sequences=Y,
        #     outputs_info=[h0, c0],
        #     non_sequences=[self.R], n_steps=max_time,
        #     truncate_gradient=self.truncate_gradient, strict=True,
        #     allow_gc=theano.config.scan.allow_gc)
        #
        # res = T.concatenate([h0.dimshuffle('x', 0, 1), outputs], axis=0).dimshuffle((1, 0, 2))
        # return res

    def _get_output_without_mask(self, train=False):
        X = self.get_input(train)  # (n_s, n_t, d_e)
        max_time, ns = X.shape[1], X.shape[0]
        X = X[:, :-1]  # drop the last frame:          # (n_s, n_t-1, d_e)
        bos = T.repeat(self.bos_vec, ns, axis=0)       # (n_s, d_e)
        bos = T.reshape(bos, (ns, 1, self.input_dim))  # (n_s, 1, d_e)
        X = T.concatenate([bos, X], axis=1)            # (n_s, n_t, d_e)
        X = X.dimshuffle(1, 0, 2)                      # (n_t, n_s, d_e)

        h0 = T.repeat(self.h_m1, ns, axis=0)
        c0 = T.repeat(self.c_m1, ns, axis=0)

        [outputs, _], updates = theano.scan(
            self._step,
            sequences=T.arange(1, max_time+1, dtype='int16'),
            outputs_info=[h0, c0],
            # R, W, bias, E_w, b_pos, W_s, b_s)
            non_sequences=[self.R, self.W, self.b, X, self.b_pos, self.W_s, self.b_s,
                           self._score_table, self._exp_score_table, self._alpha_table],
            n_steps=max_time, strict=True,
            truncate_gradient=self.truncate_gradient,
            allow_gc=theano.config.scan.allow_gc)

        res = outputs.dimshuffle(1, 0, 2)
        return res[:, :, self.output_dim:]

    def get_output(self, train=False):
        mask = self.get_input_mask(train=train)
        if mask is None:
            return self._get_output_without_mask(train=train)
        else:
            return self._get_output_with_mask(train=train)

    def set_init_cell_parameter(self, is_param=True):
        if is_param:
            if self.c_m1 not in self.params:
                self.params.append(self.c_m1)
        else:
            self.params.remove(self.c_m1)

    def set_init_h_parameter(self, is_param=True):
        if is_param:
            if self.h_m1 not in self.params:
                self.params.append(self.h_m1)
        else:
            self.params.remove(self.h_m1)

    def get_time_range(self, train):
        mask = self.get_input_mask(train=train)
        ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32')
        self.time_range = ind
        return ind

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "forget_bias_init": self.forget_bias_init.__name__,
                "input_activation": self.input_activation.__name__,
                "gate_activation": self.gate_activation.__name__,
                "truncate_gradient": self.truncate_gradient}


class ParallelAttLayer(Recurrent):
    """ Recurrent Layer with parallel attention mechanics implemented.
    """

    def __init__(self, input_dim, output_dim=128, train_init_cell=True, train_init_h=True,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 input_activation='tanh', gate_activation='hard_sigmoid', output_activation='tanh',
                 weights=None, truncate_gradient=-1, attention_len=10):

        super(ParallelAttLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)
        self.input = T.tensor3()
        self.time_range = None
        self.attention_len = attention_len

        W_z = self.init((self.input_dim, 2*self.output_dim)).get_value(borrow=True)
        R_z = self.inner_init((self.output_dim, 2*self.output_dim)).get_value(borrow=True)
        # self.b_z = shared_zeros(self.output_dim)

        W_i = self.init((self.input_dim, 2*self.output_dim)).get_value(borrow=True)
        R_i = self.inner_init((self.output_dim, 2*self.output_dim)).get_value(borrow=True)
        # self.b_i = shared_zeros(self.output_dim)

        W_f = self.init((self.input_dim, 2*self.output_dim)).get_value(borrow=True)
        R_f = self.inner_init((self.output_dim, 2*self.output_dim)).get_value(borrow=True)
        # self.b_f = self.forget_bias_init(self.output_dim)

        W_o = self.init((self.input_dim, 2*self.output_dim)).get_value(borrow=True)
        R_o = self.inner_init((self.output_dim, 2*self.output_dim)).get_value(borrow=True)
        # self.b_o = shared_zeros(self.output_dim)

        self.W_s = self.init((self.output_dim, self.input_dim))
        self.b_s = theano.shared(np.zeros(shape=(self.input_dim,), dtype=floatX), name='bias_trans', borrow=True)

        self.h_m1 = shared_zeros(shape=(1, 2*self.output_dim), name='h0')
        self.c_m1 = shared_zeros(shape=(1, 2*self.output_dim), name='c0')

        W = np.vstack((W_z[np.newaxis, :, :],
                       W_i[np.newaxis, :, :],
                       W_f[np.newaxis, :, :],
                       W_o[np.newaxis, :, :]))  # shape = (4, input_dim, 2*output_dim)
        R = np.vstack((R_z[np.newaxis, :, :],
                       R_i[np.newaxis, :, :],
                       R_f[np.newaxis, :, :],
                       R_o[np.newaxis, :, :]))  # shape = (4, output_dim, 2*output_dim)
        self.W = theano.shared(W, name='Input to hidden weights (zifo)', borrow=True)
        self.R = theano.shared(R, name='Recurrent weights (zifo)', borrow=True)
        self.b = theano.shared(np.zeros(shape=(4, 2*self.output_dim), dtype=theano.config.floatX),
                               name='bias', borrow=True)
        np_b_pos = np.zeros((self.attention_len+1, ), dtype=floatX)
        np_b_pos[0] = 1.0
        self.b_pos = theano.shared(np_b_pos, name='b_pos', borrow=True)
        self.bos_vec = theano.shared(np.zeros((1, self.input_dim), dtype=floatX), name='Begin of sentence', borrow=True)

        self.params = [self.W, self.R, self.b, self.W_s, self.b_s, self.b_pos, self.bos_vec]
        if train_init_cell:
            self.params.append(self.c_m1)
        if train_init_h:
            self.params.append(self.h_m1)

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              t,                                # sequence. t: scalar, t=1,2,...,t_max
              h_tm1, c_tm1,                     # output_info. h_tm1: (n_s, 2*d_c), c_tm1: (n_s, 2*d_c)
              R, W, bias, E_w, b_pos, W_s, b_s):
              # non_sequence. R:(2*d_c, 2*d_c), E_w:(t_max, n_s, d_e),
              # W: (4, d_e, 2*d_c)
        if t == 1:
            X_t = E_w[0]
        else:
            s_t = T.tanh(T.dot(h_tm1[:, :self.output_dim], W_s) + b_s)    # (n_s, d_e)
            pos = T.arange(t-1, -1, -1, dtype='int16')    # (t, )
            pos = T.switch(pos > self.attention_len, self.attention_len, pos)    # (t, )
            embeds = E_w[:t]  # (t, n_s, d_e)
            score_t = T.sum(s_t*embeds, axis=-1).dimshuffle(1, 0) + b_pos[pos]   # (n_s, t)
            # alpha_t = T.nnet.softmax(score_t)   # (n_s, t)
            e_score = T.exp(score_t - T.max(score_t, axis=1, keepdims=True))     # (n_s, t)
            alpha_t = e_score / T.sum(e_score, axis=1, keepdims=True)            # (n_s, t)
            X_t = T.sum(alpha_t.dimshuffle(1, 0, 'x') * embeds, axis=0)          # (n_s, d_e)

        Y_t = T.dot(X_t, W) + bias   # (n_s, 4, 2*d_c)
        G_tm1 = T.dot(h_tm1[:, self.output_dim:], R)      # (n_s, 4, 2*d_c)
        M_t = Y_t + G_tm1            # (n_s, 4, 2*d_c)

        z_t = self.input_activation(M_t[:, 0, :])
        ifo_t = self.gate_activation(M_t[:, 1:, :])
        i_t = ifo_t[:, 0, :]
        f_t = ifo_t[:, 1, :]
        o_t = ifo_t[:, 2, :]
        c_t = f_t * c_tm1 + i_t * z_t
        h_t = o_t * self.output_activation(c_t)
        return h_t, c_t

    def get_output_mask(self, train=None):
        return None

    def _get_output_with_mask(self, train=False):
        raise NotImplementedError('mask not supported for now')
        # X = self.get_input(train)
        # # mask = self.get_padded_shuffled_mask(train, X, pad=0)
        # mask = self.get_input_mask(train=train)
        # ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32').ravel()
        # max_time = T.max(ind) - 1   # drop the last frame
        # X = X.dimshuffle((1, 0, 2))
        # Y = T.dot(X, self.W) + self.b
        # # h0 = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        # h0 = T.repeat(self.h_m1, X.shape[1], axis=0)
        # c0 = T.repeat(self.c_m1, X.shape[1], axis=0)
        #
        # [outputs, _], updates = theano.scan(
        #     self._step,
        #     sequences=Y,
        #     outputs_info=[h0, c0],
        #     non_sequences=[self.R], n_steps=max_time,
        #     truncate_gradient=self.truncate_gradient, strict=True,
        #     allow_gc=theano.config.scan.allow_gc)
        #
        # res = T.concatenate([h0.dimshuffle('x', 0, 1), outputs], axis=0).dimshuffle((1, 0, 2))
        # return res

    def _get_output_without_mask(self, train=False):
        X = self.get_input(train)  # (n_s, n_t, d_e)
        max_time, ns = X.shape[1], X.shape[0]
        X = X[:, :-1]  # drop the last frame:          # (n_s, n_t-1, d_e)
        bos = T.repeat(self.bos_vec, ns, axis=0)       # (n_s, d_e)
        bos = T.reshape(bos, (ns, 1, self.input_dim))  # (n_s, 1, d_e)
        X = T.concatenate([bos, X], axis=1)            # (n_s, n_t, d_e)
        X = X.dimshuffle(1, 0, 2)                      # (n_t, n_s, d_e)

        h0 = T.repeat(self.h_m1, ns, axis=0)
        c0 = T.repeat(self.c_m1, ns, axis=0)

        [outputs, _], updates = theano.scan(
            self._step,
            sequences=T.arange(1, max_time+1, dtype='int16'),
            outputs_info=[h0, c0],
            # R, W, bias, E_w, b_pos, W_s, b_s)
            non_sequences=[self.R, self.W, self.b, X, self.b_pos, self.W_s, self.b_s],
            n_steps=max_time, strict=True,
            truncate_gradient=self.truncate_gradient,
            allow_gc=theano.config.scan.allow_gc)

        res = outputs.dimshuffle(1, 0, 2)
        return res[:, :, self.output_dim:]

    def get_output(self, train=False):
        mask = self.get_input_mask(train=train)
        if mask is None:
            return self._get_output_without_mask(train=train)
        else:
            return self._get_output_with_mask(train=train)

    def set_init_cell_parameter(self, is_param=True):
        if is_param:
            if self.c_m1 not in self.params:
                self.params.append(self.c_m1)
        else:
            self.params.remove(self.c_m1)

    def set_init_h_parameter(self, is_param=True):
        if is_param:
            if self.h_m1 not in self.params:
                self.params.append(self.h_m1)
        else:
            self.params.remove(self.h_m1)

    def get_time_range(self, train):
        mask = self.get_input_mask(train=train)
        ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32')
        self.time_range = ind
        return ind

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "forget_bias_init": self.forget_bias_init.__name__,
                "input_activation": self.input_activation.__name__,
                "gate_activation": self.gate_activation.__name__,
                "truncate_gradient": self.truncate_gradient}

class SimpAttLangModel(Sequential):
    def __init__(self, vocab_size, embed_dims=128, context_dim=128, attention_len=10,
                 loss='categorical_crossentropy', optimizer='adam'):
        super(SimpAttLangModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims

        self.optimizer = optimizers.get(optimizer)
        self.loss = objectives.get(loss)
        self.loss_fnc = objective_fnc(self.loss)

        self.add(Embedding(input_dim=vocab_size, output_dim=embed_dims))
        self.add(SimpAttLayer(input_dim=embed_dims, output_dim=context_dim, attention_len=attention_len))
        self.add(Dense(input_dim=context_dim, output_dim=vocab_size, activation='softmax'))

    @staticmethod
    def encode_length(y_true, y_pred, mask):
        probs_ = T.sum(y_true * y_pred, axis=-1)

        if mask is None:
            nb_words = y_true.shape[0] * y_true.shape[1]
            probs = probs_.ravel() + 1.0e-30
        else:
            nb_words = mask.sum()
            probs = probs_[mask.nonzero()] + 1.0e-30

        return T.sum(T.log(1.0/probs)), nb_words

    def train(self, X, y, callbacks, show_metrics, batch_size=128, extra_callbacks=(LangModelLogger(), ),
              validation_split=0., validation_data=None, shuffle=False, verbose=1):
        self.fit(X, y, callbacks, show_metrics, batch_size=batch_size, nb_epoch=1, verbose=verbose,
                 extra_callbacks=extra_callbacks, validation_split=validation_split,
                 validation_data=validation_data, shuffle=shuffle, show_accuracy=False)

    def train_from_dir(self, dir_, data_regex=re.compile(r'\d{3}.bz2'), callbacks=LangHistory(),
                       show_metrics=('loss', 'ppl'), *args, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if data_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            # y = np.zeros((X.shape[0], X.shape[1], self.vocab_size), dtype=np.int8)
            tmp = np.eye(self.vocab_size, dtype='int8')
            y = tmp[X]
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         idx = X[i, j]
            #         y[i, j, idx] = 1
            logger.info('Training on %s' % f)
            self.train(X, y, callbacks, show_metrics, *args, **kwargs)

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
        self.y = T.zeros_like(self.y_train)

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

        # train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)),
        #                         dtype=theano.config.floatX)
        # test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)),
        #                        dtype=theano.config.floatX)

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
        predict_ins = [self.X_test]

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

        self.fit = self._Sequential__fit_unweighted


class ParallelAttLangModel(Sequential):
    def __init__(self, vocab_size, embed_dims=128, context_dim=128, attention_len=10,
                 loss='categorical_crossentropy', optimizer='adam'):
        super(ParallelAttLangModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dims

        self.optimizer = optimizers.get(optimizer)
        self.loss = objectives.get(loss)
        self.loss_fnc = objective_fnc(self.loss)

        self.add(Embedding(input_dim=vocab_size, output_dim=embed_dims))
        self.add(ParallelAttLayer(input_dim=embed_dims, output_dim=context_dim, attention_len=attention_len))
        self.add(Dense(input_dim=context_dim, output_dim=vocab_size, activation='softmax'))

    @staticmethod
    def encode_length(y_true, y_pred, mask):
        probs_ = T.sum(y_true * y_pred, axis=-1)

        if mask is None:
            nb_words = y_true.shape[0] * y_true.shape[1]
            probs = probs_.ravel() + 1.0e-30
        else:
            nb_words = mask.sum()
            probs = probs_[mask.nonzero()] + 1.0e-30

        return T.sum(T.log(1.0/probs)), nb_words

    def train(self, X, y, callbacks, show_metrics, batch_size=128, extra_callbacks=(LangModelLogger(), ),
              validation_split=0., validation_data=None, shuffle=False, verbose=1):
        self.fit(X, y, callbacks, show_metrics, batch_size=batch_size, nb_epoch=1, verbose=verbose,
                 extra_callbacks=extra_callbacks, validation_split=validation_split,
                 validation_data=validation_data, shuffle=shuffle, show_accuracy=False)

    def train_from_dir(self, dir_, data_regex=re.compile(r'\d{3}.bz2'), callbacks=LangHistory(),
                       show_metrics=('loss', 'ppl'), *args, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if data_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]

        for f in train_files:
            logger.info('Loading training data from %s' % f)
            X = np.loadtxt(f, dtype='int32')
            # y = np.zeros((X.shape[0], X.shape[1], self.vocab_size), dtype=np.int8)
            tmp = np.eye(self.vocab_size, dtype='int8')
            y = tmp[X]
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         idx = X[i, j]
            #         y[i, j, idx] = 1
            logger.info('Training on %s' % f)
            self.train(X, y, callbacks, show_metrics, *args, **kwargs)

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
        self.y = T.zeros_like(self.y_train)

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

        # train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)),
        #                         dtype=theano.config.floatX)
        # test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)),
        #                        dtype=theano.config.floatX)

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
        predict_ins = [self.X_test]

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

        self.fit = self._Sequential__fit_unweighted


def slice_X(X, start_, end_=None, axis=1):
    if end_ is None:
        return [x.take(start_, axis=axis) for x in X]
    else:
        ret = []
        for y in X:
            s = [slice(None) for _ in range(y.ndim)]
            s[axis] = slice(start_, end_)
            s = tuple(s)
            ret.append(y[s])
        return ret


if __name__ == '__main__':
    # from keras.layers.core import Dropout, Dense
    # from keras.layers.core import MaskedLayer
    pass