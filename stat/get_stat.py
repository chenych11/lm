#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'
import numpy as np
from scipy.io import savemat


def get_sample_sent(number=10, min_len=200, corpus=r'../data/wiki-sg-norm-lc.txt'):
    samples = []
    with file(corpus) as f:
        for line in f:
            if len(line.split()) >= min_len:
                samples.append(line)
                if len(samples) == number:
                    break
    return samples

if __name__ == '__main__':
    datafile = r'../data/wiki-sg-norm-lc.txt'
    max_line = 5000000
    len_stat = np.zeros(max_line, dtype='int32')
    with file(datafile) as f:
        for idx, line in enumerate(f):
            if idx == max_line:
                break
            len_stat[idx] = len(line.split())

    savemat('../data/wiki-stats.mat', {'len_stat': len_stat}, oned_as='column')
