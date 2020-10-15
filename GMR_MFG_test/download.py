import json
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'data/all_data.json' # put here the path to the json file
with open(data_path,'r') as fp:
    d = json.load(fp)
## load structure from mp-id

from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser

import sys

items = ['pbe', 'hse', 'gllb-sc', 'scan']
task = items[int(sys.argv[1])]

structures = []
pbe_keys = list(d[task].keys())
print(len(pbe_keys))
batch_size = 1024

while len(structures) < len(pbe_keys):
    with MPRester("zBbeX4qitlXrl2uBfIP") as mpr:
        # if batch_size > 0:
        #     for i in range(int(len(pbe_keys)/batch_size)):
        #         lst = mpr.get_structures(pbe_keys[i*batch_size:min((i+1)*batch_size, len(pbe_keys))])
        #         structures += lst
        #         print(len(structures))
        # else:
        for i in range(len(structures), len(pbe_keys)):
            try:
                s1 = mpr.get_structure_by_material_id(pbe_keys[i])
                # or
                # data = mpr.query(criteria={"task_id": pbe_keys[i]}, properties=["final_energy", "cif"])
                # s2 = CifParser.from_string(data[0]["cif"]).get_structures()
            except:
                print('something wrong with index:', i, 'mp id is: ', pbe_keys[i])
                break
            else:
                structures.append(s1.to(fmt="cif"))
                if len(structures)%10 == 0:
                    print(len(structures))


# dump the data
df_pbe = pd.DataFrame({'mp_id':list(d[task].keys()), task+'_gap':list(d[task].values()), task+'_structure':structures})
df_pbe.set_index('mp_id',inplace=True)
df_pbe.to_csv('data/'+task+'_cif.csv')

