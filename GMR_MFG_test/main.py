import json
import pandas as pd
import numpy as np
import random

seed = 1234
random.seed(seed)
np.random.seed(seed)

items = ['pbe', 'hse', 'gllb-sc', 'scan']

structures = []
targets = []
data_size = []

from pymatgen.core.structure import Structure

for it in items:
    csv_name = 'data/' + it + '_cif.csv'
    df = pd.read_csv(csv_name)
    data_size.append(len(df))
    for i in range(len(df)):
        structures.append(Structure.from_str(df[it+'_structure'][i], fmt='cif'))
        targets.append(df[it+'_gap'][i])

print('4 data size is:', data_size)

### load exp data and shuffle

test_structures = []
test_targets = []

data_path = 'data/all_data.json' # put here the path to the json file
with open(data_path,'r') as fp:
    d = json.load(fp)

s = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
t = [x['band_gap'] for x in d['ordered_exp'].values()]

print('exp data size is:', len(s))
data_size.append(0)
for i in range(len(list(d['ordered_exp'].keys()))):
    if random.random() > 0.5:
        structures.append(s[i])
        targets.append(t[i])
        data_size[-1]+=1
    else:
        test_structures.append(s[i])
        test_targets.append(t[i])


from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel

import sys

training_mode = int(sys.argv[1])

model = MEGNetModel(10, 2, nblocks=1, lr=1e-3,
        n1=4, n2=4, n3=4, npass=1, ntarget=1,
        graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))

if training_mode == 0: # PBE -> HSE ... -> part EXP, one by one
    idx = 0
    for i in range(len(data_size)):
        model.train(structures[idx:idx+data_size[i]], targets[idx:idx+data_size[i]], epochs=20)
        idx += data_size[i]
        # model.save_model('test.hdf5')
        # model = MEGNetModel.from_file('test.hdf5')
elif training_mode == 1: # all training set together
    model.train(structures, targets, epochs=100)
    # model.save_model('test.hdf5')
    # model = MEGNetModel.from_file('test.hdf5')
elif trainning_mode == 2: # only part EXP
    model.train(structures[sum(data_size[0:len(data_size)-1]):], targets[sum(data_size[0:len(data_size)-1]):], epochs=100)
    pass
elif trainning_mode == 3: # all -> all-PBE -> all-PBE-HSE -> ... -> part EXP
    idx = 0
    for i in range(len(data_size)):
        model.train(structures[idx:], targets[idx:], epochs=20)
        idx += data_size[i]
else:
    pass

## model predict 
MAE = 0
test_size = len(test_structures)

for i in range(test_size):
    MAE += abs(model.predict_structure(test_structures[i]).ravel() - test_targets[i])

MAE /= test_size

print('MAE is:', MAE)
