import tensorflow as tf
# disable warnings and enhance performance
tf.compat.v1.disable_eager_execution()  

import json
with open('qm9_sample.json', 'r') as f:
    data = json.load(f)

from pymatgen import Molecule

qm9_ids = list(data.keys())

print('qm9 len is: ', len(qm9_ids))

molecules = [Molecule.from_dict(data[i]['molecule']) for i in qm9_ids]  # this gives a list of pymatgen Molecule

structures = molecules
targets = [data[i]['property']['U0'] for i in qm9_ids]  # We are training U0 herea

train_structures = structures[:80]
test_structures = structures[80:]
train_targets = targets[:80]
test_targets = targets[80:]

from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from megnet.utils.preprocessing import StandardScaler
import numpy as np

gc = CrystalGraph(bond_converter=GaussianDistance(
        np.linspace(0, 5, 100), 0.5), cutoff=4)
model = MEGNetModel(100, 2, graph_converter=gc, lr=1e-3)

INTENSIVE = False # U0 is an extensive quantity
scaler = StandardScaler.from_training_data(train_structures, train_targets, is_intensive=INTENSIVE)
model.target_scaler = scaler

model.train(train_structures, train_targets, epochs=500, verbose=2)

predicted_tests = []
for i in test_structures:
    predicted_tests.append(model.predict_structure(i).ravel()[0])

print(type(test_targets), type(predicted_tests))

for i in range(10):
    print(test_targets[i], predicted_tests[i])
