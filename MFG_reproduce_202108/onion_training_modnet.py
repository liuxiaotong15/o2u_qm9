import json
import pandas as pd
import numpy as np
import random
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

import os
import gc

import sys

from modnet.models import MODNetModel
from modnet.preprocessing import MODData

training_mode = int(sys.argv[1])
seed = 123
GPU_seed = 11111
GPU_device = "0"
dump_prediction_cif = False
load_old_model_enable = False
training_new_model = True
swap_E1_test = False

trained_last_time = True

if training_mode in [0, 1]:
    swap_E1_test = bool(training_mode&1)
    # special_path = 'init_randomly_EGPHS'  # only full
    # special_path = 'init_randomly_EGPHS_EGPH_EGP_EG_E'  # worst1
    special_path = 'init_randomly_EGPHS_EPHS_EHS_EH_E'  # best
    last_commit_id = 'f6f98f0'
    if training_mode == 0:
        old_model_name = last_commit_id + '_0_123_' + special_path + '.hdf5'
        GPU_device = "0"
    elif training_mode == 1:
        old_model_name = last_commit_id + '_1_123_' + special_path + '.hdf5'
        GPU_device = "1"
    else:
        pass


items = ['gllb-sc', 'pbe', 'scan', 'hse']
cut_value = 0.3

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(GPU_seed)

commit_id = str(os.popen('git --no-pager log -1 --oneline --pretty=format:"%h"').read())

dump_model_name = '{commit_id}_{training_mode}_{seed}'.format(commit_id=commit_id, 
        training_mode=training_mode,
        seed=seed)

import logging
root_logger = logging.getLogger()
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename=dump_model_name+".log",
        format='%(asctime)s-%(pathname)s[line:%(lineno)d]-%(levelname)s: %(message)s',
        level=logging.INFO)

def prediction(model):
    MAE = 0
    test_data = MODData(materials=test_structures, targets=test_targets, target_names=['gap_eV'])
    test_data.featurize()
    test_data.feature_selection(n=-1)
    
    pred = model.predict(test_data)

    mae_test = np.absolute(pred.values-test_data.df_targets.values).mean()
    return mae_test
    # logging.info('MAE is: {mae}'.format(mae=MAE))


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_device

logging.info('onion training is running, the whole training process likes a tree, gogogo!')
logging.info('commit_id is: {cid}'.format(cid=commit_id))
logging.info('training_mode is: {tm}'.format(tm=training_mode))
logging.info('device number is: GPU_{d}'.format(d=GPU_device))
logging.info('GPU seed is: {d}'.format(d=GPU_seed))

logging.info('items is {it}'.format(it=str(items)))
logging.info('trained_last_time is {e}'.format(e=str(trained_last_time)))

logging.info('load_old_model_enable={l}, old_model_name={omn}, cut_value={cv}'.format(
    l=load_old_model_enable, omn=old_model_name, cv=cut_value))
logging.info('swap_E1_test={b}'.format(b=str(swap_E1_test)))


## start to load data ##
structures = {}
targets = {}

from pymatgen.core.structure import Structure
from collections import Counter
for it in items:
    structures[it] = []
    targets[it] = []
    csv_name = 'data/' + it + '_cif.csv'
    df = pd.read_csv(csv_name)
    r = list(range(len(df)))
    random.shuffle(r)
    sp_lst = []
    for i in r:
        structures[it].append(Structure.from_str(df[it+'_structure'][i], fmt='cif'))
        sp_lst.extend(list(set(structures[it][-1].species)))
        targets[it].append(df[it+'_gap'][i])
    logging.info('dataset {item}, element dict: {d}'.format(item=it, d=Counter(sp_lst)))

### load exp data and shuffle

test_structures = []
test_targets = []

data_path = 'data/all_data.json' # put here the path to the json file
with open(data_path,'r') as fp:
    d = json.load(fp)

s_exp = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
t_exp = [x['band_gap'] for x in d['ordered_exp'].values()]

logging.info('exp data size is: {s}'.format(s=len(s_exp)))
r = list(range(len(list(d['ordered_exp'].keys()))))
random.shuffle(r)
sp_lst=[]
structures['E1'] = []
targets['E1'] = []
for i in r:
    sp_lst.extend(list(set(s_exp[i].species)))
    if random.random() > 0.5:
        structures['E1'].append(s_exp[i])
        targets['E1'].append(t_exp[i])
    else:
        test_structures.append(s_exp[i])
        test_targets.append(t_exp[i])

if swap_E1_test:
    structures['E1'], test_structures = test_structures, structures['E1']
    targets['E1'], test_targets = test_targets, targets['E1']

logging.info('dataset EXP, element dict: {d}'.format(item=it, d=Counter(sp_lst)))

logging.info(str(structures.keys()) + str(targets.keys()))
for k in structures.keys():
    logging.info(str(len(structures[k])) + str(len(targets[k])))

# data preprocess part
if load_old_model_enable:
    import pickle
    # load the past if needed
    model = MEGNetModel.from_file(old_model_name)
    diff_lst = []
    for i in range(len(s_exp)):
        diff_lst.append(model.predict_structure(s_exp[i]).ravel() - t_exp[i])
    logging.info('Std of the list(model output - exp data) is: {std}, \
mean is: {mean}'.format(std=np.std(diff_lst),
                mean=np.mean(diff_lst)))

    for it in items:
        error_lst = []
        prediction_lst = []
        targets_lst = []
        for i in range(len(structures[it])):
            prdc = model.predict_structure(structures[it][i]).ravel()
            tgt = targets[it][i]
            prediction_lst.append(prdc)
            targets_lst.append(tgt)
            e = (prdc - tgt)
            error_lst.append(e)
            if abs(e) > cut_value:
                targets[it][i] = prdc
            # targets[i] = (model.predict_structure(structures[i]).ravel() + targets[i])/2
        logging.info('Data count: {dc}, std orig dft value: {std_orig}, std of model output: {std_model}'.format(
            dc=len(targets_lst), std_orig=np.std(targets_lst), std_model=np.std(prediction_lst)))
        logging.info('Data count: {dc}, Mean orig: {mean_orig}, Mean_model: {mean_model}'.format(
            dc=len(targets_lst), mean_orig=np.mean(targets_lst), mean_model=np.mean(prediction_lst)))
        f = open(dump_model_name + '_'+ it + '.txt', 'wb') # to store and analyze the error
        pickle.dump(error_lst, f)
        f.close()

model = MODNetModel([[['gap_eV']]],
                    weights={'gap_eV':1},
                    num_neurons = [[256], [128], [16], [16]],
                    n_feat = 150,
                    act =  "elu"
                   )


model.save(dump_model_name+'_init_randomly' + '.hdf5')
init_model_tag = 'EGPHS'
start_model_tag = 'EGPHS'

ep = 5000
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

db_short_full_dict = {'G': 'gllb-sc', 'H': 'hse', 'S': 'scan', 'P': 'pbe', 'E': 'E1'}

def construct_dataset_from_str(db_short_str):
    s = []
    t = []
    for i in range(len(db_short_str)):
        s.extend(structures[db_short_full_dict[db_short_str[i]]])
        t.extend(targets[db_short_full_dict[db_short_str[i]]])
    c = list(zip(s, t))
    random.shuffle(c)
    s, t = zip(*c)
    return s, t

def find_sub_tree(cur_tag, history_tag):
    global trained_last_time
    if init_model_tag == start_model_tag or trained_last_time == False:
        trained_last_time = False
        ###### load model #######
        father_model_name = dump_model_name + '_' + history_tag + '.hdf5'
        history_tag += '_'
        history_tag += cur_tag
        if special_path != '' and history_tag not in special_path:
            return
        else:
            pass

        cur_model_name = dump_model_name + '_' + history_tag + '.hdf5'
        cur_model = MODNetModel.load(father_model_name)
        ###### get dataset ######
        s, t = construct_dataset_from_str(cur_tag)
        data = MODData(materials=s, targets=t, target_names=['gap_eV'])
        data.featurize()
        data.feature_selection(n=-1)
        ###### train ############
        cur_model.fit(data,
                val_fraction = 0.2,
                lr = 0.0002,
                batch_size = 256,
                verbose = 1,
                loss = 'mae',
                epochs = ep,
                callbacks=[callback])
        mae = prediction(cur_model)
        logging.info('MAE of {tag} is: {mae}'.format(tag=history_tag, mae=mae))
        cur_model.save(cur_model_name)
        del s, t, l
        gc.collect()
    else:
        logging.info('cur_tag is {ct}, trained_last_time is {e}'.format(
            ct=str(cur_tag), e=str(trained_last_time)))
    ###### next level #######
    if len(cur_tag) > 1:
        for i in range(len(cur_tag)):
            next_tag = cur_tag[:i] + cur_tag[i+1:]
            find_sub_tree(next_tag, history_tag)
    else:
        pass
        

find_sub_tree(init_model_tag, 'init_randomly')

