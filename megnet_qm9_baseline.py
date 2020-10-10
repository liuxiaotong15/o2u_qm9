from ase.db import connect
from ase.io import read, write
# from ase.visualize import view
import random
import pymatgen.io.ase as pymatgen_io_ase
import os
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

db = connect(filename)
# rows = list(db.select("id<100", sort='id'))
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

print(len(structures), len(targets))

# === megnet start === #

from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from megnet.utils.preprocessing import StandardScaler
import numpy as np

gc = CrystalGraph(bond_converter=GaussianDistance(
        np.linspace(0, 5, 100), 0.5), cutoff=4)
model = MEGNetModel(100, 2, graph_converter=gc, lr=1e-3)
INTENSIVE = False # U0 is an extensive quantity
scaler = StandardScaler.from_training_data(structures, targets, is_intensive=INTENSIVE)
model.target_scaler = scaler

model.train(structures, targets, epochs=500, verbose=2)

print('finish..')
