import json
import pandas as pd
import numpy as np
items = ['pbe', 'hse', 'gllb-sc', 'scan']

structures = []
targets = []

from pymatgen import MPRester

with MPRester("zBbeX4qitlXrl2uBfIP") as mpr:
    structure = mpr.get_structure_by_material_id('mp-1143')
    print(type(structure))

from pymatgen.core.structure import Structure

for it in items:
    csv_name = 'data/' + it + '_cif.csv'
    df = pd.read_csv(csv_name)
    for i in range(len(df)):
        # structures.append(dict(df[it+'_structure'][i]))
        structures.append(Structure.from_str(df[it+'_structure'][i], fmt='cif'))
        targets.append(df[it+'_gap'][i])

from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel

model = MEGNetModel(10, 2, nblocks=1, lr=1e-2,
        n1=4, n2=4, n3=4, npass=1, ntarget=1,
        graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))

model.train(structures, targets, epochs=2)
