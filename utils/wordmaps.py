#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import sys

sys.path.append('/home/cyc/Documents/workspace/rel2vec/rw2vec3')
from rw2vec import RelUtils, CorpusInfo, Rw2Vec
import cPickle as pickle


class ReadFileTest(unittest.TestCase):
    def test_ixport(self):
        wpx = export_wordmap()
        wpi = import_wordmap()
        self.failUnless('word2idx' in wpx, 'word2idx key lost for the wordmap.')
        self.failUnless('idx2word' in wpx, 'idx2word key lost for the wordmap.')
        self.failUnless('idx2wc' in wpx, 'idx2wc key lost for the wordmap.')

        self.failUnless('word2idx' in wpi, 'word2idx key lost for the wordmap.')
        self.failUnless('idx2word' in wpi, 'idx2word key lost for the wordmap.')
        self.failUnless('idx2wc' in wpi, 'idx2wc key lost for the wordmap.')


def export_wordmap(model_name='../data/models/rw2vec/rw2vec-2015-02-10.13-30-39-100.model6'):
    w2v = Rw2Vec.load(model_name)
    model = dict()
    model['word2idx'] = w2v.word2idx
    model['idx2word'] = w2v.idx2word
    model['idx2wc'] = w2v.idx2wc
    with file('../data/wiki-wordmap.wp', 'wb') as f:
        pickle.dump(model, f, -1)
    return model


def import_wordmap(fname='../data/wiki-wordmap.wp'):
    with file(fname, 'rb') as f:
        wp = pickle.load(f)
    return wp


if __name__ == '__main__':
    unittest.main()
