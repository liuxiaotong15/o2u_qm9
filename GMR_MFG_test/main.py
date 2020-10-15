import json
import pandas as pd
import numpy as np
import random

seed = 1234
random.seed(seed)

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

from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel

model = MEGNetModel(10, 2, nblocks=1, lr=1e-2,
        n1=4, n2=4, n3=4, npass=1, ntarget=1,
        graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))

model.train(structures, targets, epochs=2)

