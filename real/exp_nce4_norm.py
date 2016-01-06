#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

from utils import get_unigram_probtable
import optparse
from keras.optimizers import adam, AdamAnneal
import cPickle as pickle
from time import time
import math
import numpy as np
import theano
import theano.sparse as tsp
import theano.tensor as T
from layers import SharedWeightsDense, ActivationLayer
from keras import optimizers
from utils import chunk_sentences, categorical_crossentropy, objective_fnc, TableSampler
from models import logger, LogInfo, floatX, Graph, LangModel, make_batches, slice_X, \
    Identity, PartialSoftmaxV4, SparseEmbedding, LangLSTMLayer, Dense, LookupProb
# noinspection PyUnresolvedReferences
from lm.utils.preprocess import import_wordmap, grouped_sentences

MAX_SETN_LEN = 65  # actually 64


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

        self.add_node(SharedWeightsDense(self.nodes['part_prob'].W, self.nodes['part_prob'].b, self.sparse_coding,
                                         activation='exponential'),
                      name='true_unnorm_prob', inputs='encoder')
        self.add_node(ActivationLayer(name='normalization'), name='true_prob', inputs='true_unnorm_prob')

        self.add_output('pos_prob', node='part_prob')
        self.add_output('neg_prob', node='lookup_prob')
        self.add_output('pred_prob', node='true_prob')
        self.add_output('normalizer', node='normalizer')
        self.add_output('unrm_prob', node='true_unnorm_prob')

    # noinspection PyMethodOverriding
    def compile(self):
        pos_prob_layer = self.outputs['pos_prob']
        neg_prob_layer = self.outputs['neg_prob']
        pre_prob_layer = self.outputs['pred_prob']
        normlzer_layer = self.outputs['normalizer']
        unrm_pro_layer = self.outputs['unrm_prob']

        pos_prob_trn = pos_prob_layer.get_output(train=True)
        neg_prob_trn = neg_prob_layer.get_output(train=True) * self.nb_negative
        pos_prob_tst = pos_prob_layer.get_output(train=False)
        neg_prob_tst = neg_prob_layer.get_output(train=False) * self.nb_negative
        pre_prob_tst = pre_prob_layer.get_output(train=False)
        unrm_pro_tst = unrm_pro_layer.get_output(train=False)

        nrm_const = normlzer_layer.get_output(train=True)
        nrm_const = T.reshape(nrm_const, (nrm_const.shape[0], nrm_const.shape[1]))
        nrm_const = nrm_const.dimshuffle('x', 0, 1)
        pos_prob_trn *= nrm_const

        nrm_const_tst_ = normlzer_layer.get_output(train=False)
        nrm_const_tst = T.reshape(nrm_const_tst_, (nrm_const_tst_.shape[0], nrm_const_tst_.shape[1]))
        nrm_const_tst = nrm_const_tst.dimshuffle('x', 0, 1)
        pos_prob_tst *= nrm_const_tst
        unrm_pro_tst *= T.addbroadcast(nrm_const_tst_, 2)
        partition = T.sum(unrm_pro_tst, axis=-1)
        sum_unrm = T.sum(partition)
        squre_urm = T.sum(partition * partition)

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
                                     outputs=[test_loss, encode_len, nb_words, sum_unrm, squre_urm])

        self._train.out_labels = ('loss', )
        self._test.out_labels = ('loss', 'encode_len', 'nb_words', 'sum_unnorm', 'square_unrom')
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
            loss, encode_len, nb_words, unrm, sq_urm = out
            batch_size = np.array(batch_sizes, dtype=theano.config.floatX)

            nb_total = batch_size.sum()
            smry_loss = np.sum(loss * batch_size)/nb_total
            smry_encode_len = encode_len.sum()
            smry_nb_words = nb_words.sum()
            smry_sum_urm = unrm.sum()
            smry_sq_urm = sq_urm.sum()
            return [smry_loss, smry_encode_len, smry_nb_words, smry_sum_urm, smry_sq_urm]

        self._test.summarize_outputs = __summarize_outputs

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

    def validation(self, val_sents, batch_size, log_file=None):
        """
        :param val_sents: validation sentences.
        :type val_sents: a list, each element a ndarray
        :return: tuple
        """
        code_len = 0.
        nb_words = 0.
        loss = 0.0
        unrm = 0.0
        sq_unrm = 0.0

        for sents in val_sents:
            x = [self.negative_sample(sents)]
            loss_, code_len_, nb_words_, unrm_, sq_unrm_ = self._test_loop(self._test, x, batch_size)
            nb_words += nb_words_
            code_len += code_len_
            loss += loss_ * nb_words_
            unrm += unrm_
            sq_unrm += sq_unrm_

        loss /= nb_words
        ppl = math.exp(code_len/nb_words)
        mean_unrm = unrm / nb_words
        mean_sq_unrm = sq_unrm / nb_words
        std_unrm = mean_sq_unrm - mean_unrm * mean_unrm
        logger.info('%s:Val val_loss: %.2f - val_ppl: %.2f - partition: mean: %.2f std: %.2f' %
                    (self.__class__.__name__, loss, ppl, mean_sq_unrm, std_unrm))
        log_file.info('%s:Val val_loss: %.6f - val_ppl: %.6f - partition: mean: %.6f std: %.6f' %
                      (self.__class__.__name__, loss, ppl, mean_sq_unrm, std_unrm))

        return loss, ppl, mean_unrm, std_unrm

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

model = NCELangModelV4(sparse_coding=sparse_coding, nb_negative=options.negative,
                       embed_dims=options.embed_size, context_dims=options.context_size,
                       init_embeddings=[ini_embeds], negprob_table=unigram_table, optimizer=opt)
model.compile()
model.train(data_file=DATA_PATH,
            save_path=save_path,
            batch_size=BATCH_SIZE, train_nb_words=nb_run_words,
            val_nb_words=nb_evaluate, train_val_nb=nb_run_val,
            validation_interval=options.interval, log_file=log_file)