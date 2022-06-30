import json
import pandas as pd
import numpy as np
import random
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

import os
import gc

from megnet.data.crystal import CrystalGraph #, CrystalGraphDisordered
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel
# from megnet.callbacks import XiaotongCB

import sys
training_mode = int(sys.argv[1])
seed = 123
GPU_seed = 11111
GPU_device = "0"
dump_prediction_cif = False
load_old_model_enable = False
predict_before_dataclean = False
training_new_model = True
contain_e1_in_every_node = False
swap_E1_test = False
tau_modify_enable = False

trained_last_time = True

if training_mode in [0, 1]:
    swap_E1_test = bool(training_mode&1)
    # special_path = 'init_randomly_EGPHS_EGPH_EGP_EG_E'  # worst1
    # special_path = 'init_randomly_EGPHS_EPHS_EPH_EH_E'  # better
    special_path = 'init_randomly_EGPHS_EPHS_EHS_EH_E'  # best
    # special_path = 'init_randomly_EGPHS_GPHS_GPH_GP_G'  # worst
    # special_path = 'init_randomly_EGPHS_EGPS_GPS_GS_S'  # worst
    # special_path = '1by1_init_randomly_S'
    # special_path = '1by1_init_randomly_S_G_P_E_H'
    # special_path = '1by1_init_randomly_P_S_G_H_E'
    # special_path = '1by1_init_randomly_H_P_S_E_G'
    # last_commit_id = '02923e5' # onion
    # last_commit_id = '30f5b2b' # cleaned onion
    # last_commit_id = 'a367e11'  # 1by1 
    # last_commit_id = '3efa225'  # 1by1 cleaned
    last_commit_id = '223d078'
    if training_mode == 0:
        old_model_name_0 = last_commit_id + '_0_123_' + special_path + '.hdf5'
        old_model_name_1 = last_commit_id + '_1_123_' + special_path + '.hdf5'
        GPU_device = "0"
    # elif training_mode == 1:
    #     old_model_name = last_commit_id + '_1_123_' + special_path + '.hdf5'
    #     GPU_device = "1"
    else:
        pass


tau_dict = {'pbe': 1.297, 'hse': 1.066, 'scan': 1.257, 'gllb-sc': 0.744} # P, H, S, G # min(MSE)
# tau_dict = {'pbe': 1/0.6279685889089127,
#             'hse': 1/0.7774483582697933,
#             'scan': 1/0.7430766771711287,
#             'gllb-sc': 1/1.0419268013851504} # P, H, S, G # min(MAE)

# items = ['pbe', 'hse', 'gllb-sc', 'scan']
# items = ['gllb-sc', 'hse', 'scan', 'pbe']
# items = ['gllb-sc', 'scan', 'hse', 'pbe']
items = ['gllb-sc', 'pbe', 'scan', 'hse']
# items = ['pbe', 'scan', 'hse', 'gllb-sc']
# items = ['pbe', 'hse']


# old_model_name = '7075e10_9_4.hdf5'
# old_model_name = '249acf2_9_123_4.hdf5'
# old_model_name = 'c5ddc72_9_123_4.hdf5'
# old_model_name = '1d8f4bd_9_123_4.hdf5'
# old_model_name = 'fe32ec4_12_1234_2.hdf5'
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_device

logging.info('onion training is running, the whole training process likes a tree, gogogo!')
logging.info('commit_id is: {cid}'.format(cid=commit_id))
logging.info('training_mode is: {tm}'.format(tm=training_mode))
logging.info('device number is: GPU_{d}'.format(d=GPU_device))
logging.info('GPU seed is: {d}'.format(d=GPU_seed))

logging.info('items is {it}'.format(it=str(items)))
logging.info('contain E1 in every node is {e}'.format(e=str(contain_e1_in_every_node)))
logging.info('trained_last_time is {e}'.format(e=str(trained_last_time)))

logging.info('tau_enable={t} and tau_dict is {td}'.format(
    t=str(tau_modify_enable), td=str(tau_dict)))

logging.info('swap_E1_test={b}'.format(b=str(swap_E1_test)))
logging.info('predict_before_dataclean={p}, training_new_model={t}'.format(
    p=predict_before_dataclean, t=training_new_model))


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
        tmp = Structure.from_str(df[it+'_structure'][i], fmt='cif')
        # tmp.remove_oxidation_states()
        # tmp.state=[0]
        structures[it].append(tmp)
        sp_lst.extend(list(set(structures[it][-1].species)))
        if tau_modify_enable:
            targets[it].append(df[it+'_gap'][i] * tau_dict[it])
        else:
            targets[it].append(df[it+'_gap'][i])
    logging.info('dataset {item}, element dict: {d}'.format(item=it, d=Counter(sp_lst)))

## start to load icsd-mpid mapping

icsd_mpid_mapping = {}
mapping_file_name = "data/intersection/E_multifidelity.csv"
df = pd.read_csv(mapping_file_name)
for i in range(len(df)):
    icsd_mpid_mapping[df["icsd_id"][i]] = df['mp_id'][i]

df_he = pd.read_csv("data/5set/H.csv")
df_pe = pd.read_csv("data/5set/P.csv")
df_se = pd.read_csv("data/5set/S.csv")
df_ge = pd.read_csv("data/5set/G.csv")
### load exp data and shuffle

test_structures = []
test_targets = []
test_dft = []

data_path = 'data/all_data.json' # put here the path to the json file
with open(data_path,'r') as fp:
    d = json.load(fp)

# "icsd_id": 261333, "is_ordered": true, "band_gap": 5.54

s_exp = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
s_icsd = [x['icsd_id'] for x in d['ordered_exp'].values()]
t_exp = [x['band_gap'] for x in d['ordered_exp'].values()]

s_exp_disordered = [Structure.from_dict(x['structure']) for x in d['disordered_exp'].values()]
t_exp_disordered = [x['band_gap'] for x in d['disordered_exp'].values()]

print(len(s_exp_disordered), len(s_exp))

# give a default but only single-fidelity
for i in range(len(s_exp)):
    s_exp[i].remove_oxidation_states()
    s_exp[i].state=[0]

for i in range(len(s_exp_disordered)):
    s_exp_disordered[i].remove_oxidation_states()
    s_exp_disordered[i].state=[0]

logging.info('exp data size is: {s}'.format(s=len(s_exp)))
r = list(range(len(list(d['ordered_exp'].keys()))))
random.shuffle(r)
sp_lst=[]
structures['E1'] = []
targets['E1'] = []
target_dft = []

icsd_1 = []
icsd_2 = []

zero_cnt_1 = 0
zero_cnt_2 = 0



############################# change this for diff dft dataset ##########
# df_dft = df_ge
# df_dft = df_se
# df_dft = df_he
df_dft = df_pe
#########################################################################

for i in r:
    sp_lst.extend(list(set(s_exp[i].species)))
    mpid = icsd_mpid_mapping["icsd-{0}".format(s_icsd[i])]
    if random.random() > 0.5:
        if mpid not in list(df_dft["mp_id"]):
            continue
        structures['E1'].append(s_exp[i])
        icsd_1.append(s_icsd[i])
        targets['E1'].append(t_exp[i])
        idx = df_dft[df_dft.mp_id==mpid].index
        target_dft.append(df_dft["gap"][idx].values[0])
        if t_exp[i] == 0:
            zero_cnt_1 += 1
    else:
        if mpid not in list(df_dft["mp_id"]):
            continue
        test_structures.append(s_exp[i])
        icsd_2.append(s_icsd[i])
        test_targets.append(t_exp[i])
        idx = df_dft[df_dft.mp_id==mpid].index
        test_dft.append(df_dft["gap"][idx].values[0])

        if t_exp[i] == 0:
            zero_cnt_2 += 1

print(targets['E1'])
print(target_dft)

# analyze MAE of all, metal, and non-metal
metal_err_lst, smaller2_err_lst, bigger2_err_lst, all_err_lst = [], [], [], []

from scipy import stats
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(loss="epsilon_insensitive", epsilon=0)
x = np.array(targets['E1'])
y = np.array(target_dft)
test_x = np.array(test_targets)
test_y = np.array(test_dft)

model.fit(x.reshape([-1,1]),y)
k1 = model.coef_[0]
b1 = model.intercept_[0]

# k1, b1, r_value, p_value, std_err = stats.linregress(targets['E1'], target_dft)
err1 = (np.array(test_dft) - b1)/k1 - test_targets

for xx, yy in zip(test_targets, list(err1)):
    if xx < 0.0001:
        metal_err_lst.append(abs(yy))
    elif xx < 2:
        smaller2_err_lst.append(abs(yy))
    else:
        bigger2_err_lst.append(abs(yy))
    
    all_err_lst.append(abs(yy))


model.fit(test_x.reshape([-1,1]), test_y)
k2 = model.coef_[0]
b2 = model.intercept_[0]
# k2, b2, r_value, p_value, std_err = stats.linregress(test_targets, test_dft)
err2 = (np.array(target_dft) - b2)/k2 - targets['E1']

for xx, yy in zip(targets['E1'], list(err2)):
    if xx < 0.0001:
        metal_err_lst.append(abs(yy))
    elif xx < 2:
        smaller2_err_lst.append(abs(yy))
    else:
        bigger2_err_lst.append(abs(yy))
    
    all_err_lst.append(abs(yy))


print(k1, b1)
print(k2, b2)
# print(err1, err2)

metal_err_lst, smaller2_err_lst, bigger2_err_lst, all_err_lst = np.array(metal_err_lst), np.array(smaller2_err_lst), np.array(bigger2_err_lst), np.array(all_err_lst)

print('MAE of metal, <2, >=2 and all', np.mean(metal_err_lst), np.mean(smaller2_err_lst), np.mean(bigger2_err_lst), np.mean(all_err_lst))
print('std of metal, <2, >=2 and all', np.std(metal_err_lst), np.std(smaller2_err_lst), np.std(bigger2_err_lst), np.std(all_err_lst))



