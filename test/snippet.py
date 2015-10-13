from models import NCELangModel
import os, re
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

trn_regex=re.compile(r'\d{3}.bz2')
dir_ = 'data/fake/test'
train_files = [os.path.join(dir_, f) for f in os.listdir(dir_) if trn_regex.match(f)]
X = np.loadtxt(train_files[0], dtype='int32')

model = NCELangModel(vocab_size=15, nb_negative=2, embed_dims=128)
ins, _ = model.prepare_input(X, 0, None)
data = {model.input['idxes']: ins[0]}
model.compile()
