#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'
from keras.layers.core import Layer, LayerList, Dense, MultiInputLayer
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import Recurrent
from keras import constraints, regularizers
from keras import initializations, activations
import theano
import theano.tensor as T
import theano.sparse as tsp
import numpy as np
from keras.utils.theano_utils import shared_zeros
from utils import floatX as float_t, epsilon


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


class LangLSTMLayerV5(Recurrent):
    """ Modified from LSTMLayer: do not transform inputs. adaptation for Language modelling
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

    def __init__(self, embed_dim, train_init_cell=True, train_init_h=True,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 input_activation='tanh', gate_activation='hard_sigmoid', output_activation='tanh',
                 weights=None, truncate_gradient=-1):

        super(LangLSTMLayerV5, self).__init__()
        self.input_dim = embed_dim
        self.output_dim = embed_dim
        self.truncate_gradient = truncate_gradient
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)
        self.input = T.tensor3()
        self.time_range = None

        # W_z = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_z = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_z = shared_zeros(self.output_dim)

        # W_i = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_i = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_i = shared_zeros(self.output_dim)

        # W_f = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_f = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_f = self.forget_bias_init(self.output_dim)

        # W_o = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_o = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_o = shared_zeros(self.output_dim)

        self.h_m1 = shared_zeros(shape=(1, self.output_dim), name='h0')
        self.c_m1 = shared_zeros(shape=(1, self.output_dim), name='c0')

        # W = np.vstack((W_z[np.newaxis, :, :],
        #                W_i[np.newaxis, :, :],
        #                W_f[np.newaxis, :, :],
        #                W_o[np.newaxis, :, :]))  # shape = (4, input_dim, output_dim)
        R = np.vstack((R_z[np.newaxis, :, :],
                       R_i[np.newaxis, :, :],
                       R_f[np.newaxis, :, :],
                       R_o[np.newaxis, :, :]))  # shape = (4, output_dim, output_dim)
        # self.W = theano.shared(W, name='Input to hidden weights (zifo)', borrow=True)
        self.R = theano.shared(R, name='Recurrent weights (zifo)', borrow=True)
        self.b = theano.shared(np.zeros(shape=(4, self.output_dim), dtype=theano.config.floatX),
                               name='bias', borrow=True)

        # self.params = [self.W, self.R, self.b]
        self.params = [self.R, self.b]
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
        G_tm1 = T.dot(h_tm1, R)
        M_t = Y_t + G_tm1
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

    def _get_output_without_mask(self, train=False):
        X = self.get_input(train)
        max_time = X.shape[1] - 1  # drop the last frame
        X = X.dimshuffle((1, 0, 2))
        Y = X.dimshuffle((0, 1, 'x', 2)) + self.b
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
            raise NotImplementedError('mask not supported yet')

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


class LangLSTMLayerV6(Recurrent):
    """ Modified from LSTMLayer: do not transform inputs. adaptation for Language modelling
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

    def __init__(self, embed_dim, train_init_cell=True, train_init_h=True,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 input_activation='tanh', gate_activation='hard_sigmoid', output_activation='tanh',
                 weights=None, truncate_gradient=-1):

        super(LangLSTMLayerV6, self).__init__()
        self.input_dim = embed_dim
        self.output_dim = embed_dim
        self.truncate_gradient = truncate_gradient
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)
        self.input = T.tensor3()
        self.time_range = None

        # W_z = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_z = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_z = shared_zeros(self.output_dim)

        # W_i = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_i = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_i = shared_zeros(self.output_dim)

        # W_f = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_f = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_f = self.forget_bias_init(self.output_dim)

        # W_o = self.init((self.input_dim, self.output_dim)).get_value(borrow=True)
        R_o = self.inner_init((self.output_dim, self.output_dim)).get_value(borrow=True)
        # self.b_o = shared_zeros(self.output_dim)

        self.h_m1 = shared_zeros(shape=(1, self.output_dim), name='h0')
        self.c_m1 = shared_zeros(shape=(1, self.output_dim), name='c0')

        # W = np.vstack((W_z[np.newaxis, :, :],
        #                W_i[np.newaxis, :, :],
        #                W_f[np.newaxis, :, :],
        #                W_o[np.newaxis, :, :]))  # shape = (4, input_dim, output_dim)
        R = np.vstack((R_z[np.newaxis, :, :],
                       R_i[np.newaxis, :, :],
                       R_f[np.newaxis, :, :],
                       R_o[np.newaxis, :, :]))  # shape = (4, output_dim, output_dim)
        # self.W = theano.shared(W, name='Input to hidden weights (zifo)', borrow=True)
        self.R = theano.shared(R, name='Recurrent weights (zifo)', borrow=True)
        self.b = theano.shared(np.zeros(shape=(4, self.output_dim), dtype=theano.config.floatX),
                               name='bias', borrow=True)

        # self.params = [self.W, self.R, self.b]
        self.params = [self.R, self.b]
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
        G_tm1 = T.dot(h_tm1, R)
        M_t = Y_t + G_tm1
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

    def _get_output_without_mask(self, train=False):
        X = self.get_input(train)
        ns = X.shape[0]
        max_time = X.shape[1] - 1  # drop the last frame
        X = X.dimshuffle((1, 0, 2, 3))
        Y = X + self.b
        h0 = T.repeat(self.h_m1, ns, axis=0)
        c0 = T.repeat(self.c_m1, ns, axis=0)

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
            raise NotImplementedError('mask not supported yet')

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

        self.__input_slots = None

    def get_input(self, train=False):
        if self.__input_slots is None:
            self.__input_slots = {True: dict((name, layer.get_output(True)) for name, layer in
                                             zip(self.input_layer_names, self.input_layers)),
                                  False: dict((name, layer.get_output(False)) for name, layer in
                                              zip(self.input_layer_names, self.input_layers))}
        return self.__input_slots[train]

    def get_output(self, train=False):
        ins = self.get_input(train)
        idxes = ins['idxes']
        features = ins['features']
        weights = self.W.T.take(idxes, axis=0)
        bias = self.b.T.take(idxes, axis=0)
        return T.exp(T.sum(weights * features, axis=-1) + bias)


class PartialSoftmaxV4(Dense, MultiInputLayer):
    def __init__(self, input_dim, base_size, init='glorot_uniform', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):
        MultiInputLayer.__init__(self, slot_names=['idxes', 'sparse_codings', 'features'])
        Dense.__init__(self, base_size, input_dim, init=init, weights=weights, name=name, W_regularizer=W_regularizer,
                       b_regularizer=b_regularizer, activity_regularizer=activity_regularizer,
                       W_constraint=W_constraint, b_constraint=b_constraint)
        self.params.remove(self.b)
        self.b = shared_zeros((base_size, 1), dtype=float_t)
        self.params.append(self.b)

        self.__input_slots = None

    def get_input(self, train=False):
        if self.__input_slots is None:
            self.__input_slots = {True: dict((name, layer.get_output(True)) for name, layer in
                                             zip(self.input_layer_names, self.input_layers)),
                                  False: dict((name, layer.get_output(False)) for name, layer in
                                              zip(self.input_layer_names, self.input_layers))}
        return self.__input_slots[train]

    def get_output(self, train=False):
        ins = self.get_input(train)
        idxes = ins['idxes']
        sparse_codings = ins['sparse_codings']  # (M, B+1)
        features = ins['features']   # (ns, nt, dl)
        detectors_flat = tsp.structured_dot(sparse_codings, self.W)   # (M, dl)
        bias_flat = tsp.structured_dot(sparse_codings, self.b)
        bias = T.reshape(bias_flat, idxes.shape, ndim=idxes.ndim)
        detec_shape = T.concatenate([idxes.shape, [-1]])
        detectors = T.reshape(detectors_flat, detec_shape, ndim=idxes.ndim+1)   # (ns, nt, dl)
        return T.exp(T.sum(detectors * features, axis=-1) + bias)
        # return T.exp(T.sum(detectors * features, axis=-1))


class PartialSoftmaxV7(Dense, MultiInputLayer):
    def __init__(self, input_dim, base_size, vocab_size, init='glorot_uniform', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):
        MultiInputLayer.__init__(self, slot_names=['idxes', 'sparse_codings', 'features'])
        Dense.__init__(self, base_size, input_dim, init=init, weights=weights, name=name, W_regularizer=W_regularizer,
                       b_regularizer=b_regularizer, activity_regularizer=activity_regularizer,
                       W_constraint=W_constraint, b_constraint=b_constraint)
        self.params.remove(self.b)
        self.b = shared_zeros((vocab_size, ), dtype=float_t)
        self.params.append(self.b)

        self.__input_slots = None

    def get_input(self, train=False):
        if self.__input_slots is None:
            self.__input_slots = {True: dict((name, layer.get_output(True)) for name, layer in
                                             zip(self.input_layer_names, self.input_layers)),
                                  False: dict((name, layer.get_output(False)) for name, layer in
                                              zip(self.input_layer_names, self.input_layers))}
        return self.__input_slots[train]

    def get_output(self, train=False):
        ins = self.get_input(train)
        idxes = ins['idxes']
        sparse_codings = ins['sparse_codings']  # (M, B+1)
        features = ins['features']   # (ns, nt, dl)
        detectors_flat = tsp.structured_dot(sparse_codings, self.W)   # (M, dl)
        bias = self.b[idxes]
        detec_shape = T.concatenate([idxes.shape, [-1]])
        detectors = T.reshape(detectors_flat, detec_shape, ndim=idxes.ndim+1)   # (ns, nt, dl)
        return T.exp(T.sum(detectors * features, axis=-1) + bias)
        # return T.exp(T.sum(detectors * features, axis=-1))


class PartialSoftmaxV8(Dense, MultiInputLayer):
    def __init__(self, input_dim, base_size, init='glorot_uniform', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):
        MultiInputLayer.__init__(self, slot_names=['idxes', 'sparse_codings', 'features'])
        Dense.__init__(self, base_size, input_dim, init=init, weights=weights, name=name, W_regularizer=W_regularizer,
                       b_regularizer=b_regularizer, activity_regularizer=activity_regularizer,
                       W_constraint=W_constraint, b_constraint=b_constraint)
        self.params.remove(self.b)
        self.b = shared_zeros((base_size+1, ), dtype=float_t)
        self.params.append(self.b)
        self.base_size = base_size

        self.__input_slots = None

    def get_input(self, train=False):
        if self.__input_slots is None:
            self.__input_slots = {True: dict((name, layer.get_output(True)) for name, layer in
                                             zip(self.input_layer_names, self.input_layers)),
                                  False: dict((name, layer.get_output(False)) for name, layer in
                                              zip(self.input_layer_names, self.input_layers))}
        return self.__input_slots[train]

    def get_output(self, train=False):
        ins = self.get_input(train)
        idxes = ins['idxes']
        sparse_codings = ins['sparse_codings']                        # (M, B+1)
        features = ins['features']                                    # (ns, nt, dl)
        detectors_flat = tsp.structured_dot(sparse_codings, self.W)   # (M, dl)
        detec_shape = T.concatenate([idxes.shape, [-1]])
        detectors = T.reshape(detectors_flat, detec_shape, ndim=idxes.ndim+1)   # (ns, nt, dl)
        exceed_idxes = (idxes > self.base_size).nonzero()
        bias_idxes = T.set_subtensor(idxes[exceed_idxes], self.base_size)
        bias = self.b[bias_idxes]
        return T.exp(T.sum(detectors * features, axis=-1) + bias)


class SharedWeightsDense(Layer):
    def __init__(self, W, b, sparse_codes, activation='linear'):
        super(SharedWeightsDense, self).__init__()
        self.params = []
        self.W = W
        self.b = b
        self.__input_slots = None
        self.sparse_codes = tsp.as_sparse_variable(sparse_codes)
        self.activation = activations.get(activation)

    def get_output(self, train=False):
        ins = self.get_input(train)
        W = tsp.structured_dot(self.sparse_codes, self.W).T
        b = tsp.structured_dot(self.sparse_codes, self.b).T
        b = T.addbroadcast(b, 0)
        return self.activation(T.dot(ins, W) + b)


class SharedWeightsDenseV7(Layer):
    def __init__(self, W, b, sparse_codes, activation='linear'):
        super(SharedWeightsDenseV7, self).__init__()
        self.params = []
        self.W = W
        self.b = b
        self.__input_slots = None
        self.sparse_codes = tsp.as_sparse_variable(sparse_codes)
        self.activation = activations.get(activation)

    def get_output(self, train=False):
        ins = self.get_input(train)
        W = tsp.structured_dot(self.sparse_codes, self.W).T
        b = self.b
        return self.activation(T.dot(ins, W) + b)


class SharedWeightsDenseV8(Layer):
    def __init__(self, W, b, sparse_codes, activation='linear'):
        super(SharedWeightsDenseV8, self).__init__()
        self.params = []
        self.W = W  # (B+1, dl)
        self.vocab_size = sparse_codes.shape[0]
        tmp_b = np.zeros((self.vocab_size,), dtype=float_t)
        b_value = b.get_value(borrow=True)
        tmp_b[:b_value.shape[0]] = b_value
        self.b = theano.shared(tmp_b, borrow=True)
        self.__input_slots = None
        self.sparse_codes = tsp.as_sparse_variable(sparse_codes)
        self.activation = activations.get(activation)

    def get_output(self, train=False):
        ins = self.get_input(train)
        W = tsp.structured_dot(self.sparse_codes, self.W).T
        return self.activation(T.dot(ins, W) + self.b)


class LookupProb(Layer):
    def __init__(self, table):
        super(LookupProb, self).__init__()
        self.table = table

    def get_output(self, train=False):
        idxes = self.get_input(train)
        return self.table[idxes]


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


class SparseEmbedding(MultiInputLayer):
    """
        Turn rows of sparse representations of words into dense vectors of fixed size

        @input_dim: size of the sparse reprsentation
        @out_dim: size of dense representation
    """
    def __init__(self, input_dim, output_dim, init='uniform',
                 W_regularizer=None, activity_regularizer=None, W_constraint=None, weights=None):

        # super(Embedding, self).__init__()
        MultiInputLayer.__init__(self, slot_names=['codes', 'shape'])
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        codes = tsp.csr_matrix('sparse-codes', dtype=float_t)
        shape = T.ivector('sents-shape')

        self.set_previous(layers=[Identity(inputs={True: codes, False: codes}),
                                  Identity(inputs={True: shape, False: shape})])

        self.W = self.init((self.input_dim, self.output_dim))

        self.params = [self.W]

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.regularizers = []

        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if weights is not None:
            self.set_weights(weights)

        self.__output_slots = None

    def get_output_mask(self, train=False):
        return None

    def supports_masked_input(self):
        return False

    def get_output(self, train=False):
        if self.__output_slots is None:
            train_in = self.get_input(True)
            test_in = self.get_input(False)
            trn_codes = train_in['codes']
            trn_shape = T.concatenate([train_in['shape'], np.array([-1], dtype=train_in['shape'].dtype)])
            tst_codes = test_in['codes']
            tst_shape = T.concatenate([test_in['shape'], np.array([-1], dtype=test_in['shape'].dtype)])

            trn_features = tsp.structured_dot(trn_codes, self.W)
            tst_features = tsp.structured_dot(tst_codes, self.W)

            self.__output_slots = {True: T.reshape(trn_features, trn_shape, ndim=trn_features.ndim+1),
                                   False: T.reshape(tst_features, tst_shape, ndim=tst_features.ndim+1)}
        out = self.__output_slots[train]
        return out

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None}


class SparseEmbeddingV6(MultiInputLayer):
    """
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    """
    def __init__(self, input_dim, output_dim, init='uniform',
                 W_regularizer=None, activity_regularizer=None, W_constraint=None, weights=None):

        # super(Embedding, self).__init__()
        MultiInputLayer.__init__(self, slot_names=['codes', 'shape'])
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        codes = tsp.csr_matrix('sparse-codes', dtype=float_t)
        shape = T.ivector('sents-shape')

        self.set_previous(layers=[Identity(inputs={True: codes, False: codes}),
                                  Identity(inputs={True: shape, False: shape})])

        W0 = self.init((self.input_dim, self.output_dim))
        W1 = self.init((self.input_dim, self.output_dim))
        W2 = self.init((self.input_dim, self.output_dim))
        W3 = self.init((self.input_dim, self.output_dim))

        Wi = [None] * 4
        Wi[0] = W0.get_value()
        Wi[1] = W1.get_value()
        Wi[2] = W2.get_value()
        Wi[3] = W3.get_value()

        if weights is not None:
            for t, s in zip(Wi, weights):
                t[:] = s

        W = np.vstack((Wi[0][np.newaxis, :, :],
                       Wi[1][np.newaxis, :, :],
                       Wi[2][np.newaxis, :, :],
                       Wi[3][np.newaxis, :, :]))  # shape = (4, input_dim, output_dim)
        self.W = theano.shared(W)

        del W0
        del W1
        del W2
        del W3

        self.params = [self.W]

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.regularizers = []

        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        # if weights is not None:
        #     self.set_weights(weights)

        self.__output_slots = None

    def get_output_mask(self, train=False):
        return None

    def supports_masked_input(self):
        return False

    def get_output(self, train=False):
        if self.__output_slots is None:
            train_in = self.get_input(True)
            test_in = self.get_input(False)
            trn_codes = train_in['codes']
            trn_shape = T.concatenate([train_in['shape'], np.array([4, -1], dtype=train_in['shape'].dtype)])
            tst_codes = test_in['codes']
            tst_shape = T.concatenate([test_in['shape'], np.array([4, -1], dtype=test_in['shape'].dtype)])

            trn_features0 = tsp.structured_dot(trn_codes, self.W[0])
            trn_features1 = tsp.structured_dot(trn_codes, self.W[1])
            trn_features2 = tsp.structured_dot(trn_codes, self.W[2])
            trn_features3 = tsp.structured_dot(trn_codes, self.W[3])

            tst_features0 = tsp.structured_dot(tst_codes, self.W[0])
            tst_features1 = tsp.structured_dot(tst_codes, self.W[1])
            tst_features2 = tsp.structured_dot(tst_codes, self.W[2])
            tst_features3 = tsp.structured_dot(tst_codes, self.W[3])

            trn_features = T.stack([trn_features0, trn_features1, trn_features2, trn_features3], axis=1)
            tst_features = T.stack([tst_features0, tst_features1, tst_features2, tst_features3], axis=1)

            self.__output_slots = {True: T.reshape(trn_features, trn_shape, ndim=trn_features.ndim+1),
                                   False: T.reshape(tst_features, tst_shape, ndim=tst_features.ndim+1)}
        out = self.__output_slots[train]
        return out

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None}


class ActivationLayer(Layer):
    def __init__(self, name='linear'):
        super(ActivationLayer, self).__init__()
        self.name = name
        self.activation = activations.get(name)

    def get_output(self, train=False):
        ins = self.get_input(train)
        return self.activation(ins)

    def get_config(self):
        w = super(ActivationLayer, self).get_config()
        w['name'] = self.__class__.__name__
        w['activation'] = self.name


class EmbeddingParam(Layer):
    def __init__(self):
        super(EmbeddingParam, self).__init__()

    def get_output(self, train=False):
        return self.previous.params[0]

    def get_input(self, train=False):
        return self.previous.params[0]

    def get_max_norm(self):
        embeds = self.previous.params[0]
        norms = T.sqrt(T.sum(embeds * embeds, axis=1))
        return T.max(norms)


class LBLScoreV1(MultiInputLayer):
    def __init__(self, vocab_size, b_regularizer=None):
        super(LBLScoreV1, self).__init__(slot_names=('context', 'word'))
        self.vocab_size = vocab_size
        # self.b = T.zeros((vocab_size, 1), dtype=floatX)
        self.b = theano.shared(np.zeros((vocab_size, 1), dtype=float_t), borrow=True)
        self.params = [self.b]

        self.regularizers = []
        if b_regularizer is not None:
            self.b_regularizer = regularizers.get(b_regularizer)
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

    def get_output(self, train=False):
        ins = self.get_input(train)
        cntxt_vec = ins['context']
        wrd_vec = ins['word'][:self.vocab_size].dimshuffle(0, 1, 'x')
        prob_ = T.exp(T.dot(cntxt_vec, wrd_vec) + self.b)
        prob_ = T.addbroadcast(prob_, 2)
        prob_ = prob_.dimshuffle(0, 1)
        prob_ /= T.sum(prob_, axis=-1, keepdims=True) + epsilon
        prob = T.clip(prob_, epsilon, 1.0-epsilon)
        prob /= T.sum(prob_, axis=-1, keepdims=True) + epsilon
        # cntx_norm = T.max(T.sqrt(T.sum(cntxt_vec * cntxt_vec, axis=1)))
        return prob  # cntx_norm


class PartialSoftmaxLBL(MultiInputLayer):
    """ this layer is designed specifically for LBL language model
    """
    def __init__(self, base_size, word_vecs, b_regularizer=None):
        MultiInputLayer.__init__(self, slot_names=['idxes', 'sparse_codings', 'features'])
        self.b = shared_zeros((base_size, 1), dtype=float_t)
        self.params = [self.b]
        self.W = word_vecs[:base_size]
        self.regularizers = []
        if b_regularizer is not None:
            self.b_regularizer = regularizers.get(b_regularizer)
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        self.__input_slots = None

    def get_input(self, train=False):
        if self.__input_slots is None:
            self.__input_slots = {True: dict((name, layer.get_output(True)) for name, layer in
                                             zip(self.input_layer_names, self.input_layers)),
                                  False: dict((name, layer.get_output(False)) for name, layer in
                                              zip(self.input_layer_names, self.input_layers))}
        return self.__input_slots[train]

    def get_output(self, train=False):
        ins = self.get_input(train)
        idxes = ins['idxes']                                                    # (k+1, ns)
        sparse_codings = ins['sparse_codings']                                  # (M, B+1), where M = ns*(k+1)
        features = ins['features']                                              # (ns, dc)
        detectors_flat = tsp.structured_dot(sparse_codings, self.W)             # (M, dc)
        bias_flat = tsp.structured_dot(sparse_codings, self.b)                  # (M, 1)
        bias = T.reshape(bias_flat, idxes.shape, ndim=idxes.ndim)               # (k+1, ns)
        detec_shape = T.concatenate([idxes.shape, [-1]])                        # = (k+1, ns, -1)
        detectors = T.reshape(detectors_flat, detec_shape, ndim=idxes.ndim+1)   # (k+1, ns, dc)
        return T.exp(T.sum(detectors * features, axis=-1) + bias)               # (k+1, ns)


class PartialSoftmaxLBLV4(Dense, MultiInputLayer):
    def __init__(self, input_dim, output_dim, word_vecs, init='glorot_uniform', weights=None,
                 name=None, b_regularizer=None, activity_regularizer=None, b_constraint=None):
        MultiInputLayer.__init__(self, slot_names=['idxes', 'features'])

        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        # self.W = self.init((self.output_dim, self.input_dim))  # (V, dc)
        self.W = word_vecs
        self.b = shared_zeros(self.output_dim)                   # (V, )

        self.params = [self.b]

        self.regularizers = []

        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.b_constraint]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

        self.__input_slots = None

    def get_input(self, train=False):
        if self.__input_slots is None:
            self.__input_slots = {True: dict((name, layer.get_output(True)) for name, layer in
                                             zip(self.input_layer_names, self.input_layers)),
                                  False: dict((name, layer.get_output(False)) for name, layer in
                                              zip(self.input_layer_names, self.input_layers))}
        return self.__input_slots[train]

    def get_output(self, train=False):
        ins = self.get_input(train)
        idxes = ins['idxes']                                     # (k+1, ns)
        features = ins['features']                               # (ns, dc)
        weights = self.W.take(idxes, axis=0)                     # (k+1, ns, dc)
        bias = self.b.take(idxes, axis=0)                        # (k+1, ns)
        return T.exp(T.sum(weights * features, axis=-1) + bias)  # (k+1, ns)


class SharedWeightsDenseLBLV4(Layer):
    def __init__(self, W, b, activation='linear'):
        super(SharedWeightsDenseLBLV4, self).__init__()
        self.params = []
        self.W = W  # (V, dc)
        self.b = b  # (V, )
        self.__input_slots = None
        self.activation = activations.get(activation)

    def get_output(self, train=False):
        ins = self.get_input(train)  # (ns, dc)
        return self.activation(T.dot(ins, self.W.T) + self.b)


class PartialSoftmaxFFNN(Dense, MultiInputLayer):
    def __init__(self, input_dim, base_size, init='glorot_uniform', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):
        MultiInputLayer.__init__(self, slot_names=['idxes', 'sparse_codings', 'features'])
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.base_size = base_size

        self.input = T.matrix()
        self.W = self.init((base_size, input_dim))            # (B+1, dc)
        self.b = shared_zeros((self.base_size, 1))            # (B+1, 1)

        self.params = [self.W, self.b]

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

        self.__input_slots = None

    def get_input(self, train=False):
        if self.__input_slots is None:
            self.__input_slots = {True: dict((name, layer.get_output(True)) for name, layer in
                                             zip(self.input_layer_names, self.input_layers)),
                                  False: dict((name, layer.get_output(False)) for name, layer in
                                              zip(self.input_layer_names, self.input_layers))}
        return self.__input_slots[train]

    def get_output(self, train=False):
        ins = self.get_input(train)
        idxes = ins['idxes']                                                    # (k+1, ns)
        features = ins['features']                                              # (ns, dc)
        sp_coding = ins['sparse_codings']                                       # (M, B+1)
        detectors_flat = tsp.structured_dot(sp_coding, self.W)                  # (M, dc)
        bias_flat = tsp.structured_dot(sp_coding, self.b)                       # (M, 1)
        bias = T.reshape(bias_flat, idxes.shape, ndim=idxes.ndim)               # (k+1, ns)
        detec_shape = T.concatenate([idxes.shape, [-1]])                        # = (k+1, ns, -1)
        detectors = T.reshape(detectors_flat, detec_shape, ndim=idxes.ndim+1)   # (k+1, ns, dc)
        return T.exp(T.sum(detectors * features, axis=-1) + bias)               # (k+1, ns)