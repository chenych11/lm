#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'


def check_maps(map1, map2):
    for w1, w2 in zip(map1['idx2word'], map2['idx2word']):
        if w1 != w2:
            raise Exception('idx2word: map not equal')

    for i, m in enumerate([map1, map2]):
        for idx, w in enumerate(m['idx2word']):
            if idx != m['word2idx'][w]:
                raise Exception('map%d not consistent' % i)


if __name__ == '__main__':
    import cPickle as pickle
    wp_file = '../../data/wiki-wordmap-trunc300k.wp'
    embeds_file = '/home/cyc/Data/models/embeddings/rw2vec_embeddings-size200.pkl'

    with file(wp_file, 'rb') as f:
        wp = pickle.load(f)

    with file(embeds_file, 'rb') as f:
        em = pickle.load(f)

    check_maps(wp, em)

