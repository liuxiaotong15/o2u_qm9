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

xxx, yyy = [], [] 

def plot_output_exp_err(model, structures, targets, ax):
    test_size = len(structures)
    output_lst = []
    for i in range(test_size):
        model_output = model.predict_structure(structures[i]).ravel()
        output_lst.append(model_output[0])
        xxx.append(targets[i])
        yyy.append(model_output[0])
    
    ax.scatter(targets, output_lst, alpha=0.5)
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 12])
    ax.plot([0, 1], [0, 1], 'k--', transform=ax.transAxes)

    a, b, r_value, p_value, std_err = stats.linregress(targets, output_lst)
    print("k, b: ", a, b)

        

def prediction(model, structures, targets):
    MAE = 0
    test_size = len(structures)
    for i in range(test_size):
        model_output = model.predict_structure(structures[i]).ravel()
        err = abs(model_output - targets[i])
        if dump_prediction_cif:
            name = '{ae}_{mo}_{target}.cif'.format(
                    ae=err, mo=model_output, target=targets[i])
            structures[i].to(filename=name)
        MAE += err
    MAE /= test_size
    return MAE
    # logging.info('MAE is: {mae}'.format(mae=MAE))


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

### load exp data and shuffle

test_structures = []
test_targets = []

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

icsd_1 = []
icsd_2 = []

zero_cnt_1 = 0
zero_cnt_2 = 0
for i in r:
    sp_lst.extend(list(set(s_exp[i].species)))
    if random.random() > 0.5:
        structures['E1'].append(s_exp[i])
        icsd_1.append(s_icsd[i])
        targets['E1'].append(t_exp[i])
        if t_exp[i] == 0:
            zero_cnt_1 += 1
    else:
        test_structures.append(s_exp[i])
        icsd_2.append(s_icsd[i])
        test_targets.append(t_exp[i])
        if t_exp[i] == 0:
            zero_cnt_2 += 1

all_icsd = icsd_2 + icsd_1

print("Zero counts in E1 and E2:", zero_cnt_1, zero_cnt_2)

if swap_E1_test:
    structures['E1'], test_structures = test_structures, structures['E1']
    targets['E1'], test_targets = test_targets, targets['E1']

logging.info('dataset EXP, element dict: {d}'.format(item=it, d=Counter(sp_lst)))

logging.info(str(structures.keys()) + str(targets.keys()))
for k in structures.keys():
    logging.info(str(len(structures[k])) + str(len(targets[k])))

# ordered structures only test
# model = MEGNetModel(nfeat_edge=10, nfeat_global=2, graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))

# ordered/disordered structures test together
# model = MEGNetModel(nfeat_edge=100, nfeat_node=16, ngvocal=1, global_embedding_dim=16, graph_converter=CrystalGraphDisordered(bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5)))

# model.save_model(dump_model_name+'_init_randomly' + '.hdf5')
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
    if contain_e1_in_every_node:
        s.extend(structures['E1'])
        t.extend(targets['E1'])
    c = list(zip(s, t))
    random.shuffle(c)
    s, t = zip(*c)
    return s, t

       
# pbe_energy = prediction(model, structures['pbe'], targets['pbe'])
# ordered_energy = prediction(model, test_structures, test_targets)
# disordered_energy = prediction(model, s_exp_disordered, t_exp_disordered)
# 
# logging.info('Prediction before trainnig, MAE of \
#         pbe: {pbe}; ordered: {ordered}; disordered: {disordered}.'.format(
#     pbe=pbe_energy, ordered=ordered_energy, disordered=disordered_energy))

# find_sub_tree(init_model_tag, 'init_randomly')


cur_model_0 = MEGNetModel.from_file(old_model_name_0)
cur_model_1 = MEGNetModel.from_file(old_model_name_1)

import matplotlib.pyplot as plt
plt.figure(0)
from scipy import stats
font = {'size': 16, 'family': 'Arial'}
plt.rc('font', **font)
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

fig, ax = plt.subplots()

plot_output_exp_err(cur_model_0, test_structures, test_targets, ax)
plot_output_exp_err(cur_model_1, structures['E1'], targets['E1'], ax)

# analyze MAE of all, metal, and non-metal

metal_err_lst, smaller2_err_lst, bigger2_err_lst, all_err_lst = [], [], [], []

for xx, yy in zip(xxx, yyy):
    if xx < 0.0001:
        metal_err_lst.append(abs(xx-yy))
    elif xx < 2:
        smaller2_err_lst.append(abs(xx-yy))
    else:
        bigger2_err_lst.append(abs(xx-yy))
    
    all_err_lst.append(abs(xx-yy))


metal_err_lst, smaller2_err_lst, bigger2_err_lst, all_err_lst = np.array(metal_err_lst), np.array(smaller2_err_lst), np.array(bigger2_err_lst), np.array(all_err_lst)

print('MAE of metal, <2, >=2 and all', np.mean(metal_err_lst), np.mean(smaller2_err_lst), np.mean(bigger2_err_lst), np.mean(all_err_lst))
print('std of metal, <2, >=2 and all', np.std(metal_err_lst), np.std(smaller2_err_lst), np.std(bigger2_err_lst), np.std(all_err_lst))


# plot
a, b, r_value, p_value, std_err = stats.linregress(xxx, yyy)

s = np.linspace(0, 12, 2)
ax.plot(s, s*a+b, "r:", label=f"k={a: .2f}; b={b: .2f}", color='red')
ax.set_ylabel(f"Model output band gap (eV)")
ax.set_xlabel(f"Experimental band gap (eV)")
ax.legend()

plt.subplots_adjust(bottom=0.125, right=0.978, left=0.105, top=0.973)
# plt.show()

# mae = prediction(cur_model, s_exp_disordered, t_exp_disordered)
# logging.info('Disordered structures MAE of {tag} is: {mae}'.format(tag=old_model_name, mae=mae))

zzz = np.array(xxx) - np.array(yyy)

# dump in csv format
import pandas as pd
dict = {'icsd': all_icsd, 'exp_target': xxx, 'model_output': yyy, 'Abs. Err.': list(np.abs(zzz))}
df = pd.DataFrame(dict)
df.to_csv(old_model_name_0 + '_MAE_' + str(np.mean(all_err_lst)) +'.csv')

import seaborn as sns
import numpy as np

plt.figure(1)
fig1, ax1 = plt.subplots()

sns.distplot(zzz, ax=ax1, hist=True, kde=True)

ax1.set_xlim((-3, 3))
ax1.set_yticks([])
# ax1.get_yaxis().set_visible(False)
ax1.set_ylabel('Gaussian kernel density (arb. units)')
ax1.set_xlabel('Error on band gap (eV)')

# from matplotlib.offsetbox import AnchoredText
# anchored_text = AnchoredText(r"$\mathrm{\mu: " + str(round(np.mean(zzz), 3)) + '\n' + r" \sigma: " + str(round(np.std(zzz), 3)) + r"}$", loc=2)
# ax1.add_artist(anchored_text)

plt.annotate(r"$\mathrm{\mu: }$" + str(round(np.mean(zzz), 3)) + "\n" + r"$\mathrm{\sigma: }$" + str(round(np.std(zzz), 3)), xy=(0.05, 0.85), xycoords='axes fraction')
# fig1.legend(labels=[r"$\mathrm{\mu: " + str(round(np.mean(zzz), 3)) + r'\\' + r" \sigma: " + str(round(np.std(zzz), 3)) + r"}$"])
plt.subplots_adjust(bottom=0.13, right=0.978, left=0.052, top=0.985)
plt.show()
