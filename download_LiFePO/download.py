import json
import pandas as pd

from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser

structures = []

with MPRester("zBbeX4qitlXrl2uBfIP") as mpr:
    # data = mpr.query(criteria={"elements": {"$all": ["Fe", "O"]}}, properties=["exp.tags", "icsd_ids"])
    # data = mpr.query(criteria={"elements": {"$in": ["Fe", "O", "Li", "P"]}}, properties=["task_id"])
    data = mpr.query(criteria={"elements": {"$in": ["Fe", "O", "Li", "P"]}}, properties=["task_id"])
    print(type(data))
    print(data[0]['task_id'])
    for i in range(len(data)):
        try:
            s1 = mpr.get_structure_by_material_id(data[i]['task_id'])
        except:
            print('something wrong with index:', i, 'mp id is: ', pbe_keys[i])
            break
        else:
            structures.append(s1.to(fmt="cif"))
            if len(structures)%10 == 0:
                print(len(structures))

# dump the data
df = pd.DataFrame({'mp_id':data[:len(structures)], 'structure':structures})
df.set_index('mp_id',inplace=True)
df.to_csv('data_cif.csv')

