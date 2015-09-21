#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

import theano
from theano import tensor as T
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix
from keras import activations, initializations
from keras.layers.recurrent import Recurrent
from keras.layers.core import MaskedLayer
import numpy as np
import os
import re
# from keras.layers.core import MaskedLayer, Layer


class LangLSTMLayerV0(Recurrent):
    """
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

    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(LangLSTMLayerV0, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_i = self.init((self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros(self.output_dim)

        self.W_f = self.init((self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init(self.output_dim)

        self.W_c = self.init((self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros(self.output_dim)

        self.W_o = self.init((self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros(self.output_dim)

        self.h00 = shared_zeros(shape=(1, self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.h00
        ]

        if weights is not None:
            self.set_weights(weights)

    def get_padded_shuffled_mask(self, train, X, pad=0):
        mask = self.get_input_mask(train)
        if mask is None:
            mask = T.ones_like(X.sum(axis=-1))  # is there a better way to do this without a sum?

        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        # return mask.astype('int8')
        return mask.astype(theano.config.floatX)

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask,  # sequence
              h_tm1, c_tm1,  # output_info
              u_i, u_f, u_o, u_c):  # non_sequence
        # h_mask_tm1 = mask_tm1 * h_tm1
        # c_mask_tm1 = mask_tm1 * c_tm1

        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t_cndt = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t_cndt = o_t * self.activation(c_t_cndt)
        h_t = mask * h_t_cndt + (1 - mask) * h_tm1
        c_t = mask * c_t_cndt + (1 - mask) * c_tm1
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=0)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        # h0 = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        h0 = T.repeat(self.h00, X.shape[1], axis=0)

        [outputs, _], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
            outputs_info=[h0, T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return (T.concatenate(h0.dimshuffle('x', 0, 1), outputs, axis=0).dimshuffle((1, 0, 2)),
                    padded_mask[1:].dimshuffle(1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "forget_bias_init": self.forget_bias_init.__name__,
                "activation": self.activation.__name__,
                "inner_activation": self.inner_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class LSTMLayer(Recurrent):
    """
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
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(LSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

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

        self.params = [self.W, self.R]
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

    def get_output(self, train=False):
        X = self.get_input(train)
        # mask = self.get_padded_shuffled_mask(train, X, pad=0)
        mask = self.get_input_mask(train=train)
        ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32').ravel()
        max_time = T.max(ind)
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
        if self.return_sequences:
            return res
        # return outputs[-1]
        return res[T.arange(mask.shape[0], dtype='int32'), ind]

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
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class LangLSTMLayerV2(Recurrent):
    """
        Only h_0 are set to be parameters
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

    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 input_activation='tanh', gate_activation='hard_sigmoid', output_activation='tanh',
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(LangLSTMLayerV2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)
        self.input = T.tensor3()

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

        self.h_m1 = shared_zeros(shape=(1, self.output_dim))

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

        self.params = [self.W, self.R, self.h_m1]
        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              Y_t, mask,  # sequence
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
        c_t_cndt = f_t * c_tm1 + i_t * z_t
        h_t_cndt = o_t * self.output_activation(c_t_cndt)
        h_t = mask * h_t_cndt + (1 - mask) * h_tm1
        c_t = mask * c_t_cndt + (1 - mask) * c_tm1
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        mask = self.get_padded_shuffled_mask(train, X, pad=0)
        X = X.dimshuffle((1, 0, 2))
        Y = T.dot(X, self.W) + self.b
        # h0 = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        h0 = T.repeat(self.h_m1, X.shape[1], axis=0)

        [outputs, _], updates = theano.scan(
            self._step,
            sequences=[Y, mask],
            outputs_info=[h0, T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            non_sequences=[self.R],
            truncate_gradient=self.truncate_gradient, strict=True,
            allow_gc=theano.config.scan.allow_gc)

        if self.return_sequences:
            return (T.concatenate(h0.dimshuffle('x', 0, 1), outputs, axis=0).dimshuffle((1, 0, 2)),
                    mask[1:].dimshuffle(1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "forget_bias_init": self.forget_bias_init.__name__,
                "input_activation": self.input_activation.__name__,
                "gate_activation": self.gate_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class LSTMLayerV0(Recurrent):
    """
        optimized version of LSTM: tensorized computation.
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
                 weights=None, truncate_gradient=-1, return_sequences=False):

        super(LSTMLayerV0, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)
        self.input = T.tensor3()

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

        self.params = [self.W, self.R]
        if train_init_cell:
            self.params.append(self.c_m1)
        if train_init_h:
            self.params.append(self.h_m1)

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
              Y_t, mask,  # sequence
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
        c_t_cndt = f_t * c_tm1 + i_t * z_t
        h_t_cndt = o_t * self.output_activation(c_t_cndt)
        h_t = mask * h_t_cndt + (1 - mask) * h_tm1
        c_t = mask * c_t_cndt + (1 - mask) * c_tm1
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        mask = self.get_padded_shuffled_mask(train, X, pad=0)
        X = X.dimshuffle((1, 0, 2))
        Y = T.dot(X, self.W) + self.b
        # h0 = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        h0 = T.repeat(self.h_m1, X.shape[1], axis=0)
        c0 = T.repeat(self.c_m1, X.shape[1], axis=0)

        [outputs, _], updates = theano.scan(
            self._step,
            sequences=[Y, mask],
            outputs_info=[h0, c0],
            non_sequences=[self.R],
            truncate_gradient=self.truncate_gradient, strict=True,
            allow_gc=theano.config.scan.allow_gc)

        if self.return_sequences:
            return (T.concatenate(h0.dimshuffle('x', 0, 1), outputs, axis=0).dimshuffle((1, 0, 2)),
                    mask[1:].dimshuffle(1, 0, 2))
        return outputs[-1]

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

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "forget_bias_init": self.forget_bias_init.__name__,
                "input_activation": self.input_activation.__name__,
                "gate_activation": self.gate_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_sequences": self.return_sequences}


class MeanPooling(MaskedLayer):
    """
        Global Mean Pooling Layer
    """

    def __init__(self, start=1):
        super(MeanPooling, self).__init__()
        self.start = start

    # def supports_masked_input(self):
    # return False
    def get_output_mask(self, train=False):
        return None

    def get_output(self, train=False):
        data = self.get_input(train=train)
        mask = self.get_input_mask(train=train)
        mask = mask.dimshuffle((0, 1, 'x'))
        return (data[:, self.start:] * mask).mean(axis=1)

    def get_config(self):
        return {"name": self.__class__.__name__}


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

        self.params = [self.W, self.R]
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
        max_time = T.max(ind)
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
        return res[:, :-1, :]  # drop the last frame

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

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Dense, Activation
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('LangModel')


class LangModel(object):
    def __init__(self, vocab_size, embed_dim=128, lstm_outdim=128):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim))
        self.model.add(LangLSTMLayer(input_dim=embed_dim, output_dim=lstm_outdim))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(input_dim=lstm_outdim, output_dim=vocab_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode='categorical')

    def train(self, X, y, nb_epoch=1, *args, **kwargs):
        self.model.fit(X, y, nb_epoch=nb_epoch, *args, **kwargs)

    def train_from_dir(self, dir_, data_regex=re.compile(r'\d{3}.bz2'), *args, **kwargs):
        train_files_ = [os.path.join(dir_, f) for f in os.listdir(dir_) if data_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]

        for f in train_files:
            X = np.loadtxt(f)
            y = np.zeros((X.shape[0], X.shape[1], 15), dtype=np.int8)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    idx = X[i, j]
                    y[i, j, idx] = 1
            logger.info('training on %s' % f)
            self.train(X, y, *args, **kwargs)



if __name__ == '__main__':
    from keras.models import Sequential
    from keras.layers.embeddings import Embedding
    from keras.layers.core import Dropout, Dense, Activation
    from keras.datasets import imdb
    from keras.preprocessing import sequence
    # from keras.regularizers import l1l2

    max_features = 20000
    maxlen = 100  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # print('Build standard model (initial output as parameters)...')
    # model = Sequential()
    # # model.add(Embedding(max_features, 128, mask_zero=True, W_regularizer=l1l2(l1=0.0001, l2=0.00001)))
    # model.add(Embedding(max_features, 128, mask_zero=True))
    # # model.add(LSTM(128, 128))  # try using a GRU instead, for fun
    # model.add(LangLSTMLayer(128, 128))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, 1))
    # model.add(Activation('sigmoid'))
    # # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
    #
    # print("Train...")
    # model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test), show_accuracy=True)
    # score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    # # =======================================================================================
    # print('Build model V1(Sequence Version with mean pool. initial cell and outputs as parameters)... ')
    #     model = Sequential()
    #     # model.add(Embedding(max_features, 128, mask_zero=True, W_regularizer=l1l2(l1=0.0001, l2=0.00001)))
    #     model.add(Embedding(max_features, 128, mask_zero=True))
    #     # model.add(LSTM(128, 128))  # try using a GRU instead, for fun
    #     model.add(LangLSTMLayerV1(128, 128, return_sequences=True))
    #     model.add(MeanPooling())
    #     model.add(Dropout(0.5))
    #     model.add(Dense(128, 1))
    #     model.add(Activation('sigmoid'))
    #     # try using different optimizers and different optimizer configs
    #     model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
    #
    #     print("Train...")
    #     model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(X_test, y_test), show_accuracy=True)
    #     score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    #     print('Test score:', score)
    #     print('Test accuracy:', acc)
    # =======================================================================================
    print('Build model V1(initial cell and outputs as parameters)... ')
    model = Sequential()
    # model.add(Embedding(max_features, 128, mask_zero=True, W_regularizer=l1l2(l1=0.0001, l2=0.00001)))
    model.add(Embedding(max_features, 128, mask_zero=True))
    # model.add(LSTM(128, 128))  # try using a GRU instead, for fun
    model.add(LSTMLayer(128, 128, train_init_cell=False, train_init_h=False))
    model.add(Dropout(0.5))
    model.add(Dense(128, 1))
    model.add(Activation('sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    print("Train...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(X_test, y_test),
              show_accuracy=True)
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)

    # # =======================================================================================
    #     print('Build model with LSTMLayer (initial cell and outputs as parameters)... ')
    #     model = Sequential()
    #     # model.add(Embedding(max_features, 128, mask_zero=True, W_regularizer=l1l2(l1=0.0001, l2=0.00001)))
    #     model.add(Embedding(max_features, 128, mask_zero=True))
    #     # model.add(LSTM(128, 128))  # try using a GRU instead, for fun
    #     model.add(LSTMLayer(128, 128))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(128, 1))
    #     model.add(Activation('sigmoid'))
    #     # try using different optimizers and different optimizer configs
    #     model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
    #
    #     print("Train...")
    #     model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(X_test, y_test), show_accuracy=True)
    #     score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    #     print('Test score:', score)
    #     print('Test accuracy:', acc)

    # =======================================================================================