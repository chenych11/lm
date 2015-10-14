#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bz2 import BZ2File
from copy import copy
import unittest
import os
import numpy as np
import cPickle as pickle
import logging
import re

__author__ = 'Yunchuan Chen'
logging.basicConfig(level=logging.INFO)


class ReadFileTest(unittest.TestCase):
    def test_prprcs_wrt(self):
        if not os.path.exists('../data/corpus/wiki-sg-norm-lc-drop.bz2'):
            return
        with BZ2File('../data/corpus/wiki-sg-norm-lc-drop.bz2') as f:
            f.readline()
            line = f.readline()
            self.failUnless('it was shortlisted for the booker prize and won several other awards .'.strip() == line.strip(),
                            'read line: %s not as expected.\n' % line)

    def test_ixport(self):
        wpx, flag = export_wordmap()
        wpi = import_wordmap()

        self.failUnless(flag is True, 'Failure flag received from export map')
        if wpx is not None:
            self.failUnless('word2idx' in wpx, 'word2idx key lost for the wordmap.')
            self.failUnless('idx2word' in wpx, 'idx2word key lost for the wordmap.')
            self.failUnless('idx2wc' in wpx, 'idx2wc key lost for the wordmap.')

        self.failUnless('word2idx' in wpi, 'word2idx key lost for the wordmap.')
        self.failUnless('idx2word' in wpi, 'idx2word key lost for the wordmap.')
        self.failUnless('idx2wc' in wpi, 'idx2wc key lost for the wordmap.')


def smart_open(fname, mode='rb', buffering=5*2**20):
    _, ext = os.path.splitext(fname)
    if ext == '.bz2':
        from bz2 import BZ2File
        return BZ2File(fname, mode, buffering)
    # if ext == '.gz':
    #     from gzip import GzipFile
    #     return GzipFile(fname, mode, buffering)
    return open(fname, mode, buffering)


def export_wordmap(dist_file='../data/wiki-wordmap.wp',
                   corpus_file='../data/corpus/wiki-sg-norm-lc.txt', rebuild=False):
    """
    :param dist_file: file name to store the wordmap
    :param corpus_file: corpus source to build wordmap against
    :param rebuild: whether rebuild wordmap if it already exists.
    :return: exported model and a flag.
    """
    if os.path.exists(dist_file) and not rebuild:
        return None, True
    word2cnt = dict()
    with smart_open(corpus_file, buffering=5*2**20) as f:
        for sent in f:
            words = sent.split()
            for w in words:
                try:
                    word2cnt[w] += 1
                except KeyError:
                    word2cnt[w] = 1
    kv = sorted(word2cnt.items(), key=lambda x: x[1], reverse=True)
    idx2word = [w for w, _ in kv]
    idx2wc = [c for _, c in kv]
    word2idx = dict((w, idx) for idx, (w, _) in enumerate(kv))
    model = {'idx2word': idx2word, 'idx2wc': idx2wc, 'word2idx': word2idx}
    with file(dist_file, 'wb') as f:
        pickle.dump(model, f, -1)
    return model, True


def import_wordmap(fname='../data/wiki-wordmap.wp'):
    """
    :param fname: a string indicate where the wordmap stores.
    :return: wordmap
    """
    with file(fname, 'rb') as f:
        wp = pickle.load(f)
    return wp


def preprocess_corpus(corpus_file='../data/corpus/wiki-sg-norm-lc.txt',
                      dist_file='../data/corpus/wiki-sg-norm-lc-drop.bz2'):
    """
    :param corpus_file: original corpus file name
    :type corpus_file: str
    :param dist_file: location to store the preprocessed corpus.
    :type dist_file: str
    :return: None
    Drop all sentences with length not in [3, 64]. Replace words that occurs less than 100 times
    with a special word __rare__.
    """
    corpus_file = file(corpus_file)
    dist_file = smart_open(dist_file, mode='w')

    assert corpus_file is not None and dist_file is not None
    wp = import_wordmap()
    for line in corpus_file:
        words = line.split()
        if not (3 <= len(words) <= 64):
            continue
        words_ = copy(words)
        for idx, w in enumerate(words):
            if w not in wp['word2idx']:
                words_[idx] = '__rare__'
        sentence = ' '.join(words_)
        dist_file.writelines([sentence, '\n'])

    corpus_file.close()
    dist_file.close()


def binarize_corpus(group_size=20000, corpus_file='../data/corpus/wiki-sg-norm-lc-drop.bz2',
                    dist_file='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2',
                    max_len=64, wordmap='../data/wiki-wordmap.wp'):
    """
    :param group_size: group size. We repeatedly read group size of sentences and
    convert and store them into binary format as a batch.
    :type group_size: int
    :param corpus_file: the corpus to be converted
    :type corpus_file: str
    :param dist_file: the file to store the converted corpus
    :param max_len: maximum length of sentence. Sentences exceeds this length will be dropped.
    :param wordmap: wordmap.
    :return: None
    """
    def _index_sentence(sent):
        """
        :param sent: a sentence as a string
        :type sent: str
        :return: a list of word index
        Represents a sentence using word indexes.
        """
        words = sent.split()
        return [word2idx[w] for w in words]

    def _commit_result():
        for idx_sent in result[3:]:
            if len(idx_sent) > 0:
                sents = np.array(idx_sent, dtype=np.int32)
                shape = np.array(sents.shape, dtype=np.int32)
                dist_file.write(shape.tobytes())
                dist_file.write(sents.tobytes())

        for j in range(len(result)):
            result[j] = []

    dist_file = smart_open(dist_file, 'wb')
    assert dist_file is not None
    if isinstance(wordmap, str):
        wp = import_wordmap(fname=wordmap)
    elif isinstance(wordmap, dict):
        wp = wordmap
    else:
        logging.error('can not recognize wordmap type')
        raise TypeError('wordamp must be dict or str')
    word2idx = wp['word2idx']
    result = [[] for _ in range(max_len + 1)]
    with smart_open(corpus_file) as f:
        for i, sent in enumerate(f, start=1):
            idxs = _index_sentence(sent)
            try:
                result[len(idxs)].append(idxs)
                if i % group_size == 0:
                    _commit_result()
            except IndexError:
                continue
        _commit_result()

    dist_file.close()


def grouped_sentences(binary_corpus='../data/corpus/wiki-sg-norm-lc-drop-bin.bz2'):
    with smart_open(binary_corpus) as f:
        while True:
            shape_data = f.read(2*4)
            if shape_data == "":
                return
            shape = np.frombuffer(shape_data, dtype=np.uint32)
            siz = shape[0] * shape[1] * 4
            sents = np.frombuffer(f.read(siz), dtype=np.uint32)
            # noinspection PyTypeChecker
            sents_ = np.reshape(sents, shape)
            yield sents_


def show_grouped_sentences(group_sents, wordmap='../data/wiki-wordmap.wp'):
    """
    :param group_sents: a matrix represents a set of sentences' indexes
    :type group_sents: numpy.ndarray
    :param wordmap: word_ to index_ map and vise versa
    :return: list, a list of string representation of the sentences.
    """
    if isinstance(wordmap, str):
        # import logging
        logger = logging.getLogger('Preprocess')
        logger.warn('It would be inefficient if repeatedly call this function with wordmap name')
        wordmap = import_wordmap(fname=wordmap)
        idx2word = wordmap['idx2word']
    elif isinstance(wordmap, dict):
        idx2word = wordmap['idx2word']
    else:
        raise TypeError('wordmap must be a string representing the map location or '
                        'a dictionary containing the map')
    ret = [None] * group_sents.shape[0]
    for i in range(len(ret)):
        ret[i] = [idx2word[j] for j in group_sents[i]]

    return ret


def get_fake_data_meta(fname='../data/fake', trn_regex=re.compile(r'\d{3}.bz2')):
    data_path = os.path.abspath(fname)
    meta_file = os.path.join(data_path, 'meta.pkl')
    if not os.path.isfile(meta_file):
        train_files_ = [os.path.join(data_path, f) for f in os.listdir(data_path) if trn_regex.match(f)]
        train_files = [f for f in train_files_ if os.path.isfile(f)]
        nb_total = 0
        nb_bin = np.zeros((15,), dtype='int32')

        for f in train_files:
            X = np.loadtxt(f, dtype='int32')
            nb_bin += np.bincount(X.ravel(), minlength=15)
            nb_total += np.prod(X.shape)

        rel_freq = nb_bin.astype('float32')/nb_total
        ret = {'freq': nb_bin, 'rel_freq': rel_freq, 'nb_total': nb_total}
        with file(meta_file, 'wb') as mf:
            pickle.dump(ret, mf)
    else:
        with file(meta_file, 'rb') as mf:
            ret = pickle.load(mf)

    return ret

if __name__ == '__main__':
    if not os.path.exists('../data/corpus/wiki-sg-norm-lc-drop.bz2'):
        export_wordmap()
        preprocess_corpus()
    if not os.path.exists('../data/corpus/wiki-sg-norm-lc-drop-bin.bz2'):
        binarize_corpus()

    unittest.main()