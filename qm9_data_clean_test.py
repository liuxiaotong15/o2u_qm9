import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from ase.db import connect
from ase.io import read, write
# from ase.visualize import view
import random
import pymatgen.io.ase as pymatgen_io_ase
import numpy as np

from megnet.models import MEGNetModel
import numpy as np
from operator import itemgetter
import json

from base64 import b64encode, b64decode
#/home/inode01/xiaotong/code/megnet/mvl_models/qm9-2018.6.1 

from megnet.utils.molecule import get_pmg_mol_from_smiles

seed = 1234
random.seed(seed)
np.random.seed(seed)
filename = 'qm9.db'
commit_id = str(os.popen('git --no-pager log -1 --oneline --pretty=format:"%h"').read())

print('commit_id is: ', commit_id)
db = connect(filename)
# rows = list(db.select("id<500", sort='id'))
rows = list(db.select(sort='id'))

MODEL_NAME = 'G'

structures = []
targets = []

G = "free_energy"
def get_data_pp(idx, type):
    # extract properties
    prop = 'None'
    row = rows[idx-1]
    if(row.id != idx):
        1/0
    # extract from schnetpack source code
    shape = row.data["_shape_" + type]
    dtype = row.data["_dtype_" + type]
    prop = np.frombuffer(b64decode(row.data[type]), dtype=dtype)
    prop = prop.reshape(shape)
    return prop


def cvt_ase2pymatgen(atoms):
    atoms.set_cell(100 * np.identity(3)) # if don't set_cell, later converter will crash..
    return(pymatgen_io_ase.AseAtomsAdaptor.get_structure(atoms))

for row in rows:
    structures.append(cvt_ase2pymatgen(row.toatoms()))
    targets.append(get_data_pp(row.id, G))

# shuffle data
c = list(zip(structures, targets))
random.shuffle(c)
structures, targets = zip(*c)


Q1_s = structures[:100000]
Q1_t = []
Q1_tmp_t = targets[:100000]

noise_ratio = 0.01
for i in range(len(Q1_tmp_t)):
    Q1_t.append(np.array(Q1_tmp_t[i] * (1 + noise_ratio)))


Q2_s = structures[100000:130000]
Q2_t = list(targets[100000:130000])

Q3_s = structures[130000:]
Q3_t = list(targets[130000:])

# import pickle
# f = open('targets_' + commit_id + '.pickle', 'wb')
# pickle.dump(targets, f)
# f.close()

import tensorflow as tf
import tensorflow.keras.backend as K
keras = tf.keras

# === megnet start === #

from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from megnet.utils.preprocessing import StandardScaler

import numpy as np

def prediction(model):
    MAE = 0
    test_size = len(Q3_s)
    for i in range(test_size):
        MAE += abs(model.predict_structure(Q3_s[i]).ravel() - Q3_t[i])
    MAE /= test_size
    print('MAE is:', MAE)

train_s = Q1_s + Q2_s
train_t = Q1_t + Q2_t

gc = CrystalGraph(bond_converter=GaussianDistance(
        np.linspace(0, 5, 100), 0.5), cutoff=4)
model = MEGNetModel(100, 2, graph_converter=gc, lr=1e-3)
INTENSIVE = False # U0 is an extensive quantity
scaler = StandardScaler.from_training_data(train_s, train_t, is_intensive=INTENSIVE)
model.target_scaler = scaler

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

idx = int(0.8 * len(train_s))


model.train(train_s[:idx], train_t[:idx],
        validation_structures=train_s[idx:],
        validation_targets=train_t[idx:],
        callbacks=[callback],
        epochs=1000,
        save_checkpoint=False,
        automatic_correction=False)


print('Training finish..')

predict(model)
