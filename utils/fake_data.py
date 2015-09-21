#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'
from preprocess import smart_open
import sys
import os


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def generate(dist_dir, corpus_file='../data/corpus/wiki-sg-norm-lc.tar.bz2', sent_len=64,
             max_size=100*2**20, file_size=2**20):
    def sentence_generator():
        with smart_open(corpus_file) as f:
            for sent in f:
                words_ = sent.split()
                words = [w for w in words_ if not w.startswith('_')]
                chars_ = list(''.join(words))
                chars = [c for c in chars_ if ord('a') <= ord(c) <= ord('z')]
                chunk_len = sent_len - 1
                if len(chars) < chunk_len:
                    continue

                num_chars = [(ord(x)-ord('a'))//2 + 1 for x in chars]

                def prefix_line(prefix_char, line):
                    tmp = [prefix_char]
                    for c in line:
                        tmp.append(str(c))
                    return ' '.join(tmp) + '\n'

                cnks = list(chunks(num_chars, chunk_len))
                line = cnks[0]
                yield prefix_line('0', line)
                for line in cnks[:-1]:
                    yield prefix_line('14', line)
                line = cnks[-1]
                if len(cnks) == chunk_len:
                    yield prefix_line('14', line)

    def file_name_generator(max_nb_file=100000, spec='%03d.bz2'):
        for idx in xrange(max_nb_file):
            dist_file_ = spec % idx
            yield os.path.join(dist_dir, dist_file_)

    dfn_gen = file_name_generator()
    dist_file_name = dfn_gen.next()
    dist_file = smart_open(dist_file_name, mode='wb', buffering=2**10)
    sentences = sentence_generator()

    last_size = 0
    nb_line = 0
    while True:
        try:
            next_line = sentences.next()
        except StopIteration:
            break
        dist_file.write(next_line)
        nb_line += 1
        if nb_line % 100 == 0:
            if dist_file.tell() >= file_size:
                last_size += dist_file.tell()
                dist_file.close()

                if last_size >= max_size:
                    break

                dist_file_name = dfn_gen.next()
                dist_file = smart_open(dist_file_name, mode='wb', buffering=2*10)












