#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bz2 import BZ2File
from wordmaps import import_wordmap
from copy import copy
import unittest
import os

__author__ = 'Yunchuan Chen'


class ReadFileTest(unittest.TestCase):
    def test_prprcs_wrt(self):
        if not os.path.exists('../data/corpus/wiki-sg-norm-lc-drop.bz2'):
            return
        with BZ2File('../data/corpus/wiki-sg-norm-lc-drop.bz2') as f:
            f.readline()
            line = f.readline()
            self.failUnless('it was shortlisted for the booker prize and won several other awards .' == line,
                            'read line: %s not as expected.')


def preprocess_corpus(corpus_file='../data/corpus/wiki-sg-norm-lc.txt',
                      dist_file='../data/corpus/wiki-sg-norm-lc-drop.bz2'):
    corpus_file = file(corpus_file)
    dist_file = BZ2File(dist_file, mode='w', buffering=2**20)

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

if __name__ == '__main__':
    if not os.path.exists('../data/corpus/wiki-sg-norm-lc-drop.bz2'):
        preprocess_corpus()

    unittest.main()