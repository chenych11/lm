#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'
import numpy as np
import cPickle as pickle
import Queue


def prefix_generator(s, start=0, end=None):
    if end is None:
        end = len(s) + 1
    for idx in range(start, end):
        yield s[:idx]


#paths_line = re.compile(r'(\d+)\s+(\S+)\s+(\d+)')
def load_brown_tree(paths_file, word2idx, start=0, to_end=False):
    """
    :param paths_file: paths file which is the output of the wcluster program
    :param word2idx: a dictionary which maps each word in vocabulary to a index
    :type word2idx: dict
    :return: a tuple of word index to word cluster id and bit string and a mask
    """
    bit_namespace = set()
    idx2bitstr = [None] * len(word2idx)
    idx2cls = [None] * len(word2idx)
    idx2cls_name = [None] * len(word2idx)
    # cls_idx = -1
    with file(paths_file, 'r') as f:
        for line in f:
            try:
                bitstr_, word, _ = line.split()
            except ValueError:
                continue
            word_clses = []
            end_ = len(bitstr_) if not to_end else len(bitstr_) + 1
            for pre in prefix_generator(bitstr_, start=start, end=end_):
                if pre not in bit_namespace:
                    bit_namespace.add(pre)
                    # cls_idx += 1
                word_clses.append(pre)
            bitstr = [1 if x == '1' else -1 for x in bitstr_[:end_]]
            word_idx = word2idx[word]
            idx2bitstr[word_idx] = bitstr
            idx2cls_name[word_idx] = word_clses
    node_names = sorted(bit_namespace, key=lambda x: len(x))
    clsname2idx = dict(((n, idx) for idx, n in enumerate(node_names)))
    for i in range(len(idx2cls)):
        idx2cls[i] = [clsname2idx[x] for x in idx2cls_name[i]]

    idx2cls = np.array(pad_virtual_class(idx2cls, pad_value=len(node_names)), dtype='int32')
    idx2bitstr = np.array(pad_bitstr(idx2bitstr), dtype='int8')
    return idx2cls, idx2bitstr, idx2bitstr != 0


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


def save_tree(fn, idx2cls, idx2bitstr, mask):
    with file(fn, 'wb') as f:
        pickle.dump({'idx2cls': idx2cls, 'idx2bitstr': idx2bitstr, 'mask': mask}, f)


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


def load_huffman_tree(meta_file):
    import cPickle as pickle
    with file(meta_file, 'rb') as f:
        meta = pickle.load(f)
        rel_freq = meta['rel_freq']
    freq = zip(rel_freq, range(len(rel_freq)))
    tree = create_tree(freq)[1]
    x = tree.preorder(left_code=-1, right_code=1)
    y = sorted(x, key=lambda z: z[1], reverse=True)
    bitstr = []
    for _, _, bitstr_ in y:
        bitstr.append(bitstr_[:-1])

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
    pad_virtual_class(tmp2, pad_value=len(idx2clses))
    assert len(freq) == len(w)
    idx2cls = [None] * len(freq)
    idx2bitstr = [None] * len(freq)
    for idx, bitstr_, cls_ in w:
        idx2cls[idx] = cls_
        idx2bitstr[idx] = bitstr_

    idx2cls = np.array(idx2cls, dtype='int32')
    idx2bitstr = np.array(idx2bitstr, dtype='int8')

    return idx2cls, idx2bitstr, idx2bitstr != 0

if __name__ == '__main__':
    freq = [
        (8.167, 'a'), (1.492, 'b'), (2.782, 'c'), (4.253, 'd'),
        (12.702, 'e'),(2.228, 'f'), (2.015, 'g'), (6.094, 'h'),
        (6.966, 'i'), (0.153, 'j'), (0.747, 'k'), (4.025, 'l'),
        (2.406, 'm'), (6.749, 'n'), (7.507, 'o'), (1.929, 'p'),
        (0.095, 'q'), (5.987, 'r'), (6.327, 's'), (9.056, 't'),
        (2.758, 'u'), (1.037, 'v'), (2.365, 'w'), (0.150, 'x'),
        (1.974, 'y'), (0.074, 'z')]
    node = create_tree(freq)
    print(node)

