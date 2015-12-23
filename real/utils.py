#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

import math
import os
import cPickle as pickle
from scipy.stats import rv_discrete
from keras.callbacks import History, BaseLogger
from keras.utils.generic_utils import Progbar
import theano
import theano.tensor as T
import numpy as np
import Queue

floatX = theano.config.floatX
if floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7


def categorical_crossentropy(y_true, y_pred):
    """
    :param y_true: true index labels with shape (ns, nt)
    :param y_pred: predicted probabilities with shape (ns, nt, V)
    :return: cce
    """
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True)

    ns = y_true.shape[0]
    nt = y_true.shape[1]
    sample_idx = T.reshape(T.arange(ns), (ns, 1))
    time_idx = T.reshape(T.arange(nt), (1, nt))
    probs_ = y_pred[sample_idx, time_idx, y_true]
    return -T.log(probs_)


def objective_fnc(fn):
    def symvar(y_true, y_pred, mask=None):
        obj_output = fn(y_true, y_pred)
        if mask is None:
            # return obj_output.mean(dtype=theano.config.floatX)
            return T.sum(obj_output) / obj_output.shape[0].astype(floatX)
        else:
            # obj_output = obj_output[mask.nonzero()]
            # return obj_output.mean(dtype=theano.config.floatX)
            obj_output = T.sum(obj_output * mask)
            return obj_output / mask.shape[0].astype(floatX)
    return symvar


def chunk_sentences(old_sentences, new_sentences, chunk_size, no_return=False, min_nb_ch=5):
    """
    :param old_sentences: [{nb_sents: x, sents: [...]}, ...]
    :param new_sentences:
    :param chunk_size:
    :param no_return:
    :return:
    """
    sent_len = new_sentences.shape[1]

    if old_sentences[sent_len]:
        nb_sents = old_sentences[sent_len]['nb_sents'] + new_sentences.shape[0]
        old_sentences[sent_len]['nb_sents'] = nb_sents
        old_sentences[sent_len]['sents'].append(new_sentences)

    else:
        nb_sents = new_sentences.shape[0]
        old_sentences[sent_len] = {'nb_sents': nb_sents,
                                   'sents': [new_sentences]}

    if nb_sents >= chunk_size*min_nb_ch and not no_return:
        nb_chunks = nb_sents // chunk_size
        nb_ret = nb_chunks * chunk_size
        tmp = np.vstack(old_sentences[sent_len]['sents'])
        old_sentences[sent_len]['sents'] = [tmp[nb_ret:]]
        old_sentences[sent_len]['nb_sents'] = old_sentences[sent_len]['sents'][0].shape[0]
        return tmp[:nb_ret]
    else:
        return None


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


def get_unigram_probtable(nb_words, wordmap='../data/wiki-wordmap.wp',
                          save_path='../data/wiki-unigram-prob-size10000.pkl'):
    if os.path.exists(save_path):
        with file(save_path, 'rb') as f:
            freq = pickle.load(f)
        return freq

    with file(wordmap, 'rb') as f:
        wp = pickle.load(f)

    idx2wc = wp['idx2wc']
    idx2wc[nb_words-1] = sum(idx2wc[nb_words-1:])
    nb_total = sum(idx2wc[:nb_words])

    freq = np.array(idx2wc[:nb_words], dtype=floatX)/nb_total
    freq_reduce = freq[nb_words-1] * 2.0/3.0
    freq[nb_words-1] -= freq_reduce
    pivot = nb_words // 2
    nb = nb_words - pivot
    gain = freq_reduce / nb
    freq[pivot:nb_words] += gain
    freq = freq / freq.sum()
    with file(save_path, 'wb') as f:
        pickle.dump(freq, f, -1)

    return freq


def prefix_generator(s, start=0, end=None):
    if end is None:
        end = len(s) + 1
    for idx in range(start, end):
        yield s[:idx]


def pad_bitstr(bitstr):
    """
    :param bitstr:
    :type bitstr: list
    :return: padded list of bits
    """
    max_bit_len = 0
    for bits in bitstr:
        if len(bits) > max_bit_len:
            max_bit_len = len(bits)
    for bits in bitstr:
        bits.extend([0] * (max_bit_len-len(bits)))

    return bitstr


def pad_virtual_class(clses, pad_value):
    max_cls_len = 0
    for nodes in clses:
        if len(nodes) > max_cls_len:
            max_cls_len = len(nodes)
    for nodes in clses:
        nodes.extend([pad_value] * (max_cls_len-len(nodes)))

    return clses


class HuffmanNode(object):
    def __init__(self, left=None, right=None, root=None):
        self.left = left
        self.right = right
        self.root = root     # Why?  Not needed for anything.

    def children(self):
        return self.left, self.right

    def preorder(self, path=None, left_code=0, right_code=1, collector=None):
        if collector is None:
            collector = []
        if path is None:
            path = []
        if self.left is not None:
            if isinstance(self.left[1], HuffmanNode):
                self.left[1].preorder(path+[left_code], left_code, right_code, collector)
            else:
                # print(self.left[1], path+[left_code])
                collector.append((self.left[1], self.left[0], path+[left_code]))
        if self.right is not None:
            if isinstance(self.right[1], HuffmanNode):
                self.right[1].preorder(path+[right_code], left_code, right_code, collector)
            else:
                # print(self.right[1], path+[right_code])
                collector.append((self.right[1], self.right[0], path+[right_code]))

        return collector


def create_tree(frequencies):
    p = Queue.PriorityQueue()
    for value in frequencies:     # 1. Create a leaf node for each symbol
        p.put(value)              #    and add it to the priority queue
    while p.qsize() > 1:          # 2. While there is more than one node
        l, r = p.get(), p.get()   # 2a. remove two highest nodes
        node = HuffmanNode(l, r)  # 2b. create internal node with children
        p.put((l[0]+r[0], node))  # 2c. add new node to queue
    return p.get()                # 3. tree is complete - return root node


def load_huffman_tree(prob_table):
    rel_freq = prob_table
    freq = zip(rel_freq, range(len(rel_freq)))
    tree = create_tree(freq)[1]
    x = tree.preorder(left_code=-1, right_code=1)
    y = sorted(x, key=lambda z: z[1], reverse=True)
    # bitstr = []
    # for _, _, bitstr_ in y:
    #     bitstr.append(bitstr_[:-1])

    z = [(wrdidx, bits, list(prefix_generator(bits, end=len(bits)))) for wrdidx, _, bits in y]
    clses = set()
    for _, _, ele in z:
        for i in ele:
            clses.add(''.join('%+d' % j for j in i))
    idx2clses = sorted(clses, key=lambda ele: len(ele))
    cls2idx = dict(((cls, idx) for idx, cls in enumerate(idx2clses)))
    w = map(lambda x: (x[0], x[1], [cls2idx[''.join('%+d' % j for j in p)] for p in x[2]]), z)

    tmp1, tmp2 = [], []
    for _, bits, cls_idx in w:
        tmp1.append(bits)
        tmp2.append(cls_idx)
    pad_bitstr(tmp1)
    pad_virtual_class(tmp2, pad_value=len(idx2clses)-1)
    assert len(freq) == len(w)
    idx2cls = [None] * len(freq)
    idx2bitstr = [None] * len(freq)
    for idx, bitstr_, cls_ in w:
        idx2cls[idx] = cls_
        idx2bitstr[idx] = bitstr_

    idx2cls = np.array(idx2cls, dtype='int32')
    idx2bitstr = np.array(idx2bitstr, dtype='int8')

    return idx2cls, idx2bitstr, idx2bitstr != 0


def save_tree(fn, idx2cls, idx2bitstr, mask):
    with file(fn, 'wb') as f:
        pickle.dump({'idx2cls': idx2cls, 'idx2bitstr': idx2bitstr, 'mask': mask}, f)


class TableSampler(rv_discrete):
    def __init__(self, table):
        nk = np.arange(len(table))
        super(TableSampler, self).__init__(b=len(table)-1, values=(nk, table))

    def sample(self, shape, dtype='int32'):
        return self.rvs(size=shape).astype(dtype)


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
        self.log_values = []

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