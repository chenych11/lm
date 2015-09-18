#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
__author__ = 'Yunchuan Chen'


class ReadFileTest(unittest.TestCase):
    def test_readlines(self):
        iter_lines = []
        read_lines = []
        with file('../data/test_data') as f:
            for line in f:
                iter_lines.append(line)

            f.seek(0)
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                read_lines.append(line)
        self.failUnless(len(iter_lines) == len(read_lines),
                        'Iterating over file is different from readlines\n'
                        'The result of iterating over lines: %s\n'
                        'The result of readlines: %s' % (str(iter_lines), str(read_lines)))


if __name__ == '__main__':
    unittest.main()
