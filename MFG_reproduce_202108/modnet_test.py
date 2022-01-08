import numpy as np
from matminer.datasets import load_dataset
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
import matplotlib.pyplot as plt 
from pymatgen.core import Composition


from matminer.datasets import load_dataset

df = load_dataset("matbench_expt_gap")
df["composition"] = df["composition"].map(Composition) # maps composition to a pymatgen composition object
df.head()


# This instantiates the MODData
data = MODData(
    materials=df["composition"], # you can provide composition objects to MODData
    targets=df["gap expt"],
    target_names=["gap_expt_eV"]
)



# Featurization of the moddata
# It will automatically apply composition only featurizers
data.featurize()


from sklearn.model_selection import train_test_split
split = train_test_split(range(100), test_size=0.1, random_state=1234)
train, test = data.split(split)


train.feature_selection(n=-1)
# if you want to use precomputed cross_nmi of the MP. This saves time :
# data.feature_selection(n=-1, use_precomputed_cross_nmi)


model = MODNetModel([[['gap_expt_eV']]],
                    weights={'gap_expt_eV':1},
                    num_neurons = [[256], [128], [16], [16]],
                    n_feat = 150,
                    act =  "elu"
                   )


model.fit(train,
          val_fraction = 0.1,
          lr = 0.0002,
          batch_size = 64,
          loss = 'mae',
          epochs = 100,
          verbose = 1,
         )


pred = model.predict(test)
pred.head()

mae_test = np.absolute(pred.values-test.df_targets.values).mean()
print(f'mae: {mae_test}')


