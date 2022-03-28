import json
import pandas as pd
import numpy as np
import random
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

import os
import gc

from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel
# from megnet.callbacks import XiaotongCB

import sys
training_mode = int(sys.argv[1])
seed = 123
GPU_seed = 11111
GPU_device = "0"
dump_prediction_cif = False
load_old_model_enable = True
predict_before_dataclean = False
training_new_model = True
contain_e1_in_every_node = False
swap_E1_test = False
tau_modify_enable = False

trained_last_time = True


# best: 1706f35_0_123_init_randomly_EGPHS_EPHS_EHS_EH_E.hdf5
if training_mode in [0, 1]:
    swap_E1_test = bool(training_mode&1)
    # special_path = 'init_randomly_EGPHS_EGPH_EGP_EG_E'  # worst1
    # special_path = 'init_randomly_EGPHS_EPHS_EPH_EH_E'  # better
    # special_path = 'init_randomly_EGPHS_EPHS_EHS_EH_E'  # best
    special_path = '1by1_init_randomly_S_G_P_E_H'  # best
    last_commit_id = 'a367e11'
    if training_mode == 0:
        old_model_name = last_commit_id + '_0_123_' + special_path + '.hdf5'
        GPU_device = "0"
    elif training_mode == 1:
        old_model_name = last_commit_id + '_1_123_' + special_path + '.hdf5'
        GPU_device = "1"
    else:
        pass

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
    test_size = len(test_structures)
    err_lst = []
    sum0, sum02, sum2 = [], [], []
    for i in range(test_size):
        model_output = model.predict_structure(test_structures[i]).ravel()
        err = abs(model_output - test_targets[i])
        err_lst.append(err)
        if(test_targets[i] < 0.0001):
            sum0.append(err)
        elif(test_targets[i] < 2):
            sum02.append(err)
        else:
            sum2.append(err)

    MAE = sum(err_lst)/len(err_lst) 
    logging.info('MAE is: {mae}'.format(mae=MAE))
    logging.info('SUM and count {sum1}, {sum2}, {sum3}, {cnt1}, {cnt2}, {cnt3},'.format(sum1=sum(sum0), sum2=sum(sum02), sum3=sum(sum2), cnt1=len(sum0), cnt2=len(sum02), cnt3=len(sum2)))
    return MAE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_device

## start to load data ##
structures = {}
targets = {}


from pymatgen.core.structure import Structure
from collections import Counter
items = ['gllb-sc', 'pbe', 'scan', 'hse']
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
        if tau_modify_enable:
            targets[it].append(df[it+'_gap'][i] * tau_dict[it])
        else:
            targets[it].append(df[it+'_gap'][i])
    logging.info('dataset {item}, element dict: {d}'.format(item=it, d=Counter(sp_lst)))


test_structures = []
test_targets = []
test_input = []

data_path = 'data/all_data.json' # put here the path to the json file
with open(data_path,'r') as fp:
    d = json.load(fp)

s_exp = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
t_exp = [x['band_gap'] for x in d['ordered_exp'].values()]

r = list(range(len(list(d['ordered_exp'].keys()))))
random.shuffle(r)
sp_lst=[]
structures['E1'] = []
targets['E1'] = []
sum_zero = 0
for i in r:
    sp_lst.extend(list(set(s_exp[i].species)))
    if t_exp[i] < 0.001:
        sum_zero += 1
    if random.random() > 0.5:
        structures['E1'].append(s_exp[i])
        targets['E1'].append(t_exp[i])
    else:
        test_structures.append(s_exp[i])
        test_targets.append(t_exp[i])

if swap_E1_test:
    structures['E1'], test_structures = test_structures, structures['E1']
    targets['E1'], test_targets = test_targets, targets['E1']

# print(sum_zero)

# last_commit_id = 'a367e11'
# for p in ['1by1_init_randomly_S_G_P_E_H', '1by1_init_randomly_H_P_S_E_G', '1by1_init_randomly_H_P_G_S_E', '1by1_init_randomly_E']:
#     if training_mode in [0, 1]:
#         swap_E1_test = bool(training_mode&1)
#         special_path = p
#         if training_mode == 0:
#             old_model_name = last_commit_id + '_0_123_' + special_path + '.hdf5'
#             GPU_device = "0"
#         elif training_mode == 1:
#             old_model_name = last_commit_id + '_1_123_' + special_path + '.hdf5'
#             GPU_device = "1"
#         else:
#             pass
#     
#     model = MEGNetModel.from_file(old_model_name)
#     prediction(model)
# 
last_commit_id = '02923e5'
# for p in ['init_randomly_EGPHS', 'init_randomly_EGPHS_EPHS_EHS_EH_E', 'init_randomly_EGPHS_GPHS_GPH_GP_G', 'init_randomly_EGPHS_EGPH_EGP_EG_E']:
for p in ['init_randomly_EGPHS', 'init_randomly_EGPHS_EPHS_EHS_EH_E', 'init_randomly_EGPHS_GPHS_GPS_GP_G', 'init_randomly_EGPHS_EGPH_EGS_EG_E']:
    if training_mode in [0, 1]:
        swap_E1_test = bool(training_mode&1)
        special_path = p
        if training_mode == 0:
            old_model_name = last_commit_id + '_0_123_' + special_path + '.hdf5'
            GPU_device = "0"
        elif training_mode == 1:
            old_model_name = last_commit_id + '_1_123_' + special_path + '.hdf5'
            GPU_device = "1"
        else:
            pass
    
    model = MEGNetModel.from_file(old_model_name)
    prediction(model)

