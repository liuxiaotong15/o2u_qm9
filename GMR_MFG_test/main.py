import json
import pandas as pd
import numpy as np
import random
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 

seed = 1234
random.seed(seed)
np.random.seed(seed)
commit_id = str(os.popen('git --no-pager log -1 --oneline --pretty=format:"%h"').read())

print('commit_id is: ', commit_id)

items = ['pbe', 'hse', 'gllb-sc', 'scan']

structures = []
targets = []
data_size = []
sample_weights = []

from pymatgen.core.structure import Structure

for it in items:
    csv_name = 'data/' + it + '_cif.csv'
    df = pd.read_csv(csv_name)
    data_size.append(len(df))
    r = list(range(len(df)))
    random.shuffle(r)
    for i in r:
        structures.append(Structure.from_str(df[it+'_structure'][i], fmt='cif'))
        targets.append(df[it+'_gap'][i])
        sample_weights.append(1.0/len(r))

print('4 data size is:', data_size)

### load exp data and shuffle

test_structures = []
test_targets = []
test_input = []

data_path = 'data/all_data.json' # put here the path to the json file
with open(data_path,'r') as fp:
    d = json.load(fp)

s = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
t = [x['band_gap'] for x in d['ordered_exp'].values()]

print('exp data size is:', len(s))
data_size.append(0)
r = list(range(len(list(d['ordered_exp'].keys()))))
random.shuffle(r)
for i in r:
    if random.random() > 0.5:
        structures.append(s[i])
        targets.append(t[i])
        data_size[-1]+=1
        sample_weights.append(1.0)
    else:
        test_structures.append(s[i])
        test_targets.append(t[i])


from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel
from megnet.callbacks import XiaotongCB

import sys

def prediction(model):
    MAE = 0
    test_size = len(test_structures)
    for i in range(test_size):
        MAE += abs(model.predict_structure(test_structures[i]).ravel() - test_targets[i])
    MAE /= test_size
    print('MAE is:', MAE)

training_mode = int(sys.argv[1])

model = MEGNetModel(10, 2, nblocks=1, lr=1e-3,
        n1=4, n2=4, n3=4, npass=1, ntarget=1,
        graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))

ep = 1000
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

for s in test_structures:
    test_input.append(model.graph_converter.graph_to_input(model.graph_converter.convert(s)))

if training_mode == 0: # PBE -> HSE ... -> part EXP, one by one
    idx = 0
    for i in range(len(data_size)):
        model.train(structures[idx:idx+data_size[i]], targets[idx:idx+data_size[i]], epochs=ep)
        idx += data_size[i]
        prediction(model)
elif training_mode == 1: # all training set together
    model.train(structures, targets, epochs=ep*len(data_size))
    prediction(model)
elif training_mode == 2: # only part EXP
    model.train(structures[sum(data_size[0:len(data_size)-1]):], targets[sum(data_size[0:len(data_size)-1]):], epochs=ep*len(data_size))
    prediction(model)
elif training_mode == 3: # all -> all-PBE -> all-PBE-HSE -> ... -> part EXP
    idx = 0
    for i in range(len(data_size)):
        model.train(structures[idx:], targets[idx:], epochs=ep)
        idx += data_size[i]
        prediction(model)
elif training_mode == 4: # use E1 as validation dataset, P -> H -> G -> S one by one
    idx = 0
    for i in range(len(data_size)-1):
        print('vali dataset length: ', len(structures[sum(data_size[:-1]):]))
        model.train(structures[idx:idx+data_size[i]], targets[idx:idx+data_size[i]],
                validation_structures=structures[sum(data_size[:-1]):],
                validation_targets=targets[sum(data_size[:-1]):],
                callbacks=[callback],
                epochs=ep,
                save_checkpoint=False,
                automatic_correction=False)
        idx += data_size[i]
        prediction(model)
elif training_mode == 5: # use more accuracy dataset as validation dataset, P -> H -> G -> S one by one
    idx = 0
    for i in range(len(data_size)-1):
        model.train(structures[idx:idx+data_size[i]], targets[idx:idx+data_size[i]],
                validation_structures=structures[sum(data_size[:i+1]):],
                validation_targets=targets[sum(data_size[:i+1]):],
                callbacks=[callback],
                epochs=ep,
                save_checkpoint=False,
                automatic_correction=False)
        idx += data_size[i]
        prediction(model)
elif training_mode == 6: # PBE -> HSE ... -> part EXP, one by one, with 20% validation
    idx = 0
    for i in range(len(data_size)):
        model.train(structures[idx:idx+int(0.8*data_size[i])], targets[idx:idx+int(0.8*data_size[i])],
                validation_structures=structures[idx+int(0.8*data_size[i]):(idx+data_size[i])],
                validation_targets=targets[idx+int(0.8*data_size[i]):(idx+data_size[i])],
                callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
                epochs=ep,
                save_checkpoint=False,
                automatic_correction=False)
        idx += data_size[i]
        prediction(model)
elif training_mode == 7: # all training set together with 20% validation
    l = len(structures)
    c = list(zip(structures, targets))
    random.shuffle(c)
    structures, targets = zip(*c)
    model.train(structures[:int(0.8 * l)], targets[:int(0.8 * l)],
            validation_structures=structures[int(0.8 * l):],
            validation_targets=targets[int(0.8 * l):],
            callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
            epochs=ep*len(data_size),
            save_checkpoint=False,
            automatic_correction=False)
    prediction(model)
elif training_mode == 8: # only part EXP with 20% validation
    model.train(structures[-1*data_size[-1]:int(-0.2*data_size[-1])], targets[-1*data_size[-1]:int(-0.2*data_size[-1])],
            validation_structures=structures[int(-0.2*data_size[-1]):],
            validation_targets=targets[int(-0.2*data_size[-1]):],
            callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
            epochs=ep*len(data_size),
            save_checkpoint=False,
            automatic_correction=False)
    prediction(model)
elif training_mode == 9 or training_mode == 10: # all -> all-PBE -> all-PBE-HSE -> ... -> part EXP with 20% validation
    idx = 0
    for i in range(len(data_size)):
        s = structures[idx:]
        t = targets[idx:]
        sw = sample_weights[idx:]
        c = list(zip(s, t, sw))
        random.shuffle(c)
        s, t, sw = zip(*c)
        l = len(s)
        if training_mode == 9:
            sw = None
        model.train(s[:int(0.8*l)], t[:int(0.8*l)],
                validation_structures=s[int(0.8*l):],
                validation_targets=t[int(0.8*l):],
                callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
                save_checkpoint=False,
                automatic_correction=False,
                sample_weights=sw,
                epochs=ep)
        idx += data_size[i]
        prediction(model)
elif training_mode == 11: # PBE -> HSE -> part EXP, one by one, with 20% validation (no G no S)
    idx = 0
    for i in range(len(data_size)):
        if i > 2 and i < len(data_size) -1: 
            model.train(structures[idx:idx+int(0.8*data_size[i])], targets[idx:idx+int(0.8*data_size[i])],
                    validation_structures=structures[idx+int(0.8*data_size[i]):(idx+data_size[i])],
                    validation_targets=targets[idx+int(0.8*data_size[i]):(idx+data_size[i])],
                    callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
                    epochs=ep,
                    save_checkpoint=False,
                    automatic_correction=False)
        idx += data_size[i]
        prediction(model)
elif training_mode == 12: # all -> all-PBE -> all-PBE-HSE -> part EXP with 20% validation (no G no S)
    idx = 0
    for i in range(len(data_size)):
        s = structures[idx:]
        t = targets[idx:]
        sw = sample_weights[idx:]
        c = list(zip(s, t, sw))
        random.shuffle(c)
        s, t, sw = zip(*c)
        l = len(s)
        if i > 2 and i < len(data_size) -1: 
            model.train(s[:int(0.8*l)], t[:int(0.8*l)],
                    validation_structures=s[int(0.8*l):],
                    validation_targets=t[int(0.8*l):],
                    callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
                    save_checkpoint=False,
                    automatic_correction=False,
                    sample_weights=sw,
                    epochs=ep)
        idx += data_size[i]
        prediction(model)

else:
    pass

## model save and load
model.save_model(commit_id+str(training_mode)+'.hdf5')
# model = MEGNetModel.from_file('test.hdf5')
## model predict 

