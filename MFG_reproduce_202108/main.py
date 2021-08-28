import json
import pandas as pd
import numpy as np
import random
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed = 123
random.seed(seed)
np.random.seed(seed)
commit_id = str(os.popen('git --no-pager log -1 --oneline --pretty=format:"%h"').read())

print('commit_id is: ', commit_id)

# items = ['pbe', 'hse', 'gllb-sc', 'scan']
# items = ['gllb-sc', 'hse', 'scan', 'pbe']
# items = ['gllb-sc', 'pbe', 'scan', 'hse']
items = ['pbe', 'hse']

tau_modify_enable = False
# tau_dict = {'pbe': 1.297, 'hse': 1.066, 'scan': 1.257, 'gllb-sc': 0.744} # P, H, S, G # min(MSE)
tau_dict = {'pbe': 1/0.6279685889089127,
            'hse': 1/0.7774483582697933,
            'scan': 1/0.7430766771711287,
            'gllb-sc': 1/1.0419268013851504} # P, H, S, G # min(MAE)

load_old_model_enable = True
old_model_name = '7075e10_9_4.hdf5'
cut_value = 0.3

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
        if tau_modify_enable:
            targets.append(df[it+'_gap'][i] * tau_dict[it])
        else:
            targets.append(df[it+'_gap'][i])
        sample_weights.append(1.0/len(r))

print('data size is:', data_size)

### load exp data and shuffle

test_structures = []
test_targets = []
test_input = []

data_path = 'data/all_data.json' # put here the path to the json file
with open(data_path,'r') as fp:
    d = json.load(fp)

s_exp = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
t_exp = [x['band_gap'] for x in d['ordered_exp'].values()]

print('exp data size is:', len(s_exp))
data_size.append(0)
r = list(range(len(list(d['ordered_exp'].keys()))))
random.shuffle(r)
for i in r:
    if random.random() > 0.5:
        structures.append(s_exp[i])
        targets.append(t_exp[i])
        data_size[-1]+=1
        sample_weights.append(1.0)
    else:
        test_structures.append(s_exp[i])
        test_targets.append(t_exp[i])


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


dump_model_name = '{commit_id}_{training_mode}_{seed}'.format(commit_id=commit_id, 
        training_mode=training_mode,
        seed=seed)
print(dump_model_name)

# data preprocess part
if load_old_model_enable:
    import pickle
    # load the past if needed
    model = MEGNetModel.from_file(old_model_name)
    idx = 0
    print('-' * 100)
    diff_lst = []
    for i in range(len(s_exp)):
        diff_lst.append(model.predict_structure(s_exp[i]).ravel() - t_exp[i])
    print('std of the diff of  model output and exp data is: {std}, \
            mean diff is: {mean}'.format(std=np.std(diff_lst),
                mean=np.mean(diff_lst)))

    for sz in data_size[:-1]:
        error_lst = []
        prediction_lst = []
        targets_lst = []
        for i in range(idx, idx + sz):
            prdc = model.predict_structure(structures[i]).ravel()
            tgt = targets[i]
            prediction_lst.append(prdc)
            targets_lst.append(tgt)
            e = (prdc - tgt)
            error_lst.append(e)
            if abs(e) > cut_value:
                targets[i] = prdc
            # targets[i] = (model.predict_structure(structures[i]).ravel() + targets[i])/2
        print('std orig: {std_orig}, std_model: {std_model}'.format(
            std_orig=np.std(targets_lst), std_model=np.std(prediction_lst)))
        print('mean orig: {mean_orig}, mean_model: {mean_model}'.format(
            mean_orig=np.mean(targets_lst), mean_model=np.mean(prediction_lst)))
        f = open(dump_model_name + '_'+ str(sz) + '.txt', 'wb') # to store and analyze the error
        pickle.dump(error_lst, f)
        f.close()
        idx += sz

# model = MEGNetModel(10, 2, nblocks=3, lr=1e-3,
#         n1=4, n2=4, n3=4, npass=1, ntarget=1,
#         graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))

model = MEGNetModel(nfeat_edge=10, nfeat_global=2, graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))

ep = 5000
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

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
                automatic_correction=False,
                batch_size = 512)
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
                automatic_correction=False,
                batch_size = 512)
        idx += data_size[i]
        prediction(model)
elif training_mode == 6: # PBE -> HSE ... -> part EXP, one by one, with 20% validation
    idx = 0
    for i in range(len(data_size)):
        model.train(structures[idx:idx+int(0.8*data_size[i])], targets[idx:idx+int(0.8*data_size[i])],
                validation_structures=structures[idx+int(0.8*data_size[i]):(idx+data_size[i])],
                validation_targets=targets[idx+int(0.8*data_size[i]):(idx+data_size[i])],
                # callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
                callbacks=[callback],
                epochs=ep,
                save_checkpoint=False,
                batch_size = 512,
                automatic_correction=False)
        model.save_model(dump_model_name+'_'+str(i)+'.hdf5')
        idx += data_size[i]
        prediction(model)
elif training_mode == 11: # PBE -> HSE ... -> part EXP, one by one, with 20% of last dataset as validation
    idx = 0
    for i in range(len(data_size)):
        model.train(structures[idx:idx+int(0.8*data_size[i])], targets[idx:idx+int(0.8*data_size[i])],
                validation_structures=structures[sum(data_size[:-1])+int(0.8*data_size[-1]):],
                validation_targets=targets[sum(data_size[:-1])+int(0.8*data_size[-1]):],
                # validation_structures=structures[idx+int(0.8*data_size[i]):(idx+data_size[i])],
                # validation_targets=targets[idx+int(0.8*data_size[i]):(idx+data_size[i])],
                # callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
                callbacks=[callback],
                epochs=ep,
                save_checkpoint=False,
                batch_size = 512,
                automatic_correction=False)
        # model.save_model(commit_id+'_'+str(training_mode)+'_'+str(i)+'.hdf5')
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
            # callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
            callbacks=[callback],
            epochs=ep*len(data_size),
            batch_size = 512,
            save_checkpoint=False,
            automatic_correction=False)
    prediction(model)
elif training_mode == 8: # only part EXP with 20% validation
    model.train(structures[-1*data_size[-1]:int(-0.2*data_size[-1])], targets[-1*data_size[-1]:int(-0.2*data_size[-1])],
            validation_structures=structures[int(-0.2*data_size[-1]):],
            validation_targets=targets[int(-0.2*data_size[-1]):],
            # callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
            callbacks=[callback],
            epochs=ep*len(data_size),
            save_checkpoint=False,
            batch_size = 512,
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
                # callbacks=[callback, XiaotongCB((test_input, test_targets), commit_id)],
                callbacks=[callback],
                save_checkpoint=False,
                automatic_correction=False,
                sample_weights=sw,
                batch_size = 512,
                epochs=ep)
        model.save_model(dump_model_name+'_'+str(i)+'.hdf5')
        idx += data_size[i]
        prediction(model)
else:
    pass

## model save and load
# model.save_model(commit_id+str(training_mode)+'.hdf5')
# model = MEGNetModel.from_file('test.hdf5')
## model predict 

