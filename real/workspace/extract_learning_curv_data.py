#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Yunchuan Chen'

import sys
import os
import re
from scipy.io import savemat
import numpy as np

log_dir = sys.argv[1]
pat = re.compile(r"main-nce4-.*-V(\d+)-N\d+.log")
file_pat = re.compile(sys.argv[2]) if len(sys.argv) >= 3 else pat
# INFO:NCELangModelV4:Train - time: 1453042236.299597 - loss: 4.672819
# INFO:NCELangModelV4:Val val_loss: 4.653410 - val_ppl: 351.053158
trn_pat = re.compile(r'.*:Train - time: (\d+\.\d+) - loss: (\d+\.\d+)')
val_pat = re.compile(r'.*:Val val_loss: (\d+\.\d+) - val_ppl: (\d+\.\d+)')
log_files = os.listdir(log_dir)

loss_data = {}
val_data = {}
for file_name in os.listdir(log_dir):
    m_k = pat.match(file_name)
    if m_k is None:
        continue
    k = m_k.group(1)
    loss_key = 'lossV'+k
    val_key = 'pplV' + k
    loss_data[loss_key] = []
    val_data[val_key] = []
    with file(log_dir+'/'+file_name, 'r') as f:
        for line in f:
            m = trn_pat.match(line)
            if m:
                loss_data[loss_key].append([float(m.group(1)), float(m.group(2))])
                continue
            m = val_pat.match(line)
            if m:
                val_data[val_key].append([float(m.group(1)), float(m.group(2))])

data = {}
for k in loss_data:
    data[k] = np.array(loss_data[k])
for k in val_data:
    data[k] = np.array(val_data[k])

savemat(log_dir+'/loss.mat', data)






