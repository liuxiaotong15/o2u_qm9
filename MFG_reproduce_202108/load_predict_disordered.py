import json
import pandas as pd
import numpy as np
import random
import tensorflow as tf

from pymatgen.core.structure import Structure
from collections import Counter

# tf.compat.v1.disable_eager_execution()

import os
import gc

from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel
# from megnet.callbacks import XiaotongCB
from megnet.data.crystal import (
    CrystalGraph,
    get_elemental_embeddings,
    CrystalGraphWithBondTypes,
    CrystalGraphDisordered,
)
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
    special_path = 'init_randomly_EGPHS_EPHS_EHS_EH_E'  # best
    last_commit_id = '1706f35'
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
    for i in range(test_size):
        # test_structures[i].state = [0]
        # cg = CrystalGraphDisordered(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5))
        graph = model.graph_converter.convert(test_structures[i])
        # graph['atom'] = cg.atom_converter.convert(graph['atom']) # .tolist()
        graph['atom'] = [embed[i] for i in graph['atom']]
        # print(graph['atom'])
        # model_output = model.predict_structure(test_structures[i]).ravel()
        model_output = model.predict_graph(graph).ravel()
        err = abs(model_output - test_targets[i])
        err_lst.append(err)
    MAE = sum(err_lst)/len(err_lst) 
    logging.info('MAE is: {mae}'.format(mae=MAE))
    return MAE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_device

## start to load data ##
structures = {}
targets = {}
test_structures = []
test_targets = []
test_input = []

data_path = 'data/all_data.json' # put here the path to the json file
with open(data_path,'r') as fp:
    d = json.load(fp)

s_exp = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
t_exp = [x['band_gap'] for x in d['ordered_exp'].values()]

s_exp_disordered = [Structure.from_dict(x['structure']) for x in d['disordered_exp'].values()]
t_exp_disordered = [x['band_gap'] for x in d['disordered_exp'].values()]

r = list(range(len(list(d['ordered_exp'].keys()))))
for i in r:
    s_exp[i].remove_oxidation_states()
    test_structures.append(s_exp[i])
    test_targets.append(t_exp[i])

# r = list(range(len(list(d['disordered_exp'].keys()))))
# for i in r:
#     s_exp_disordered[i].remove_oxidation_states()
#     test_structures.append(s_exp_disordered[i])
#     test_targets.append(t_exp_disordered[i])


model = MEGNetModel.from_file(old_model_name)
model.summary()
embed = model.get_weights()[0]
print(model.get_weights()[0].shape)

model_new = MEGNetModel(nfeat_edge=10, nfeat_global=2, graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))
model_new.summary()
# model_new.set_weights(model.get_weights()[0:]) 
# prediction(model_new)


model = MEGNetModel(nfeat_edge=100, nfeat_node=16, ngvocal=4, global_embedding_dim=16, graph_converter=CrystalGraphDisordered(bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5)))
model.summary()

# # print([type(i.species) for i in s_exp_disordered[0]])
# # data preprocess part
# if load_old_model_enable:
#     # model = MEGNetModel(nfeat_edge=10, nfeat_global=2, graph_converter=CrystalGraphDisordered(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))
#     # load the past if needed
#     # model = MEGNetModel.from_file(old_model_name)
#     # model = MEGNetModel.from_file('/home/ubuntu/code/megnet/mvl_models/mf_2020/pbe_gllb_hse_exp_disorder/0/best_model.hdf5')
#     # model = MEGNetModel.from_file('/home/ubuntu/code/megnet/mvl_models/mf_2020/pbe_gllb_hse_exp/0/best_model.hdf5')
#     # model = MEGNetModel(nfeat_edge=100, nfeat_node=16, ngvocal=4, global_embedding_dim=16, graph_converter=CrystalGraphDisordered(bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5)))
#     model = MEGNetModel(nfeat_edge=100, nfeat_node=16, ngvocal=4, global_embedding_dim=16, graph_converter=CrystalGraphDisordered(bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5)))
#     model.summary()
#     prediction(model)
