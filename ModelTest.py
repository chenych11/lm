#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

from langLSTM import LangModel
import os

model = LangModel(15)
model.train_from_dir(os.path.abspath('data/fake'), validation_split=0.1, batch_size=256, show_accuracy=True)
