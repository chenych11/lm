#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

from tree_util import load_brown_tree

idx2cls, idx2bitstr, mask = load_brown_tree('../brown-cluster/fake-c15-p2.out/paths', dict((str(x), x) for x in range(15)))

print idx2cls
print idx2bitstr
print mask
