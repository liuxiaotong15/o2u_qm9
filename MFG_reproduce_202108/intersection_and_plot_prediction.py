import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from megnet.models import MEGNetModel

from pymatgen.core.structure import Structure

def plot_output_exp_err(model0, model1, structures, E_targets, dft_targets, ax):
    MAE = 0
    xxx, yyy = [], []
    test_size = len(structures)
    output_lst = []
    
    err_lst = []
    err_cutoff_lst = []
    for i in range(test_size):
        model_output0 = model0.predict_structure(structures[i]).ravel()
        model_output1 = model1.predict_structure(structures[i]).ravel()
        model_output = (model_output0 + model_output1)/2
        output_lst.append(model_output[0])
        xxx.append(E_targets[i])
        yyy.append(model_output[0])
        err_lst.append(model_output[0] - E_targets[i])
        if len(dft_targets) > 0:
            if abs(model_output[0] - dft_targets[i]) > 0.3:
                err_cutoff_lst.append(model_output[0] - E_targets[i])
            else:
                err_cutoff_lst.append(dft_targets[i] - E_targets[i])
        else:
            pass
        MAE += abs(model_output[0] - E_targets[i])
    MAE /= test_size
    err_lst = np.array(err_lst)
    err_cutoff_lst = np.array(err_cutoff_lst)
    print('MAE, mean and std of model output:', MAE, np.mean(err_lst), np.std(err_lst))
    if len(dft_targets) > 0:
        print('MAE, mean and std of >0.3 correct:', np.mean(np.abs(err_cutoff_lst)), np.mean(err_cutoff_lst), np.std(err_cutoff_lst))

    ax.scatter(E_targets, output_lst, alpha=0.5)
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 12])
    ax.plot([0, 1], [0, 1], 'k--', transform=ax.transAxes)

    a, b, r_value, p_value, std_err = stats.linregress(xxx, yyy)
    s = np.linspace(0, 12, 2)
    ax.plot(s, s*a+b, "r:", label=f"k={a: .2f}; b={b: .2f}; MAE={MAE: .2f} eV", color='red')
    ax.set_ylabel(f"Model output band gap (eV)")
    ax.set_xlabel(f"Experimental band gap (eV)")
    ax.legend()
    

df_pbe = pd.read_csv("data/5set/P.csv")
df_scan = pd.read_csv("data/5set/S.csv")
df_gllb = pd.read_csv("data/5set/G.csv")
df_hse = pd.read_csv("data/5set/H.csv")
df_exp = pd.read_csv("data/5set/E_with_structures.csv")

df_exp_no_dft = pd.concat([df_exp, df_pbe, df_pbe]).drop_duplicates(subset=['mp_id'], keep=False)
df_exp_no_dft = pd.concat([df_exp_no_dft, df_scan, df_scan]).drop_duplicates(subset=['mp_id'], keep=False)
df_exp_no_dft = pd.concat([df_exp_no_dft, df_gllb, df_gllb]).drop_duplicates(subset=['mp_id'], keep=False)
df_exp_no_dft = pd.concat([df_exp_no_dft, df_hse, df_hse]).drop_duplicates(subset=['mp_id'], keep=False)


df_hse = df_hse.set_index("mp_id")
df_gllb = df_gllb.set_index("mp_id")
df_scan = df_scan.set_index("mp_id")
df_exp = df_exp.set_index("mp_id")
df_pbe = df_pbe.set_index("mp_id")

df_pbe_expt = df_pbe.join(df_exp,how="inner")
df_hse_expt = df_hse.join(df_exp,how="inner")
df_scan_expt = df_scan.join(df_exp,how="inner")
df_gllb_expt = df_gllb.join(df_exp,how="inner")

print(df_exp_no_dft, df_pbe_expt, df_hse_expt, df_scan_expt, df_gllb_expt)


special_path = 'init_randomly_EGPHS_EPHS_EHS_EH_E'  # best
last_commit_id = '223d078'
old_model_name_0 = last_commit_id + '_0_123_' + special_path + '.hdf5'
old_model_name_1 = last_commit_id + '_1_123_' + special_path + '.hdf5'

cur_model_0 = MEGNetModel.from_file(old_model_name_0)
cur_model_1 = MEGNetModel.from_file(old_model_name_1)

for idx, df in enumerate([df_exp_no_dft, df_pbe_expt, df_hse_expt, df_scan_expt, df_gllb_expt]):
    plt.figure(idx)
    font = {'size': 16, 'family': 'Arial'}
    plt.rc('font', **font)
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    
    fig, ax = plt.subplots()

    test_structures = []
    E_targets = []
    DFT_targets = []
    for (E, dft, s) in zip(df['E_gap'], df['gap'], df['E_structure']):
        s = Structure.from_str(s, fmt='cif')
        s.remove_oxidation_states()
        s.state=[0]
        test_structures.append(s)
        E_targets.append(E)
        DFT_targets.append(dft)
    if idx == 0:
        DFT_targets = []
    plot_output_exp_err(cur_model_0, cur_model_1, test_structures, E_targets, DFT_targets, ax)
    plt.subplots_adjust(bottom=0.125, right=0.978, left=0.105, top=0.973)
    # plt.show()
    plt.savefig(str(idx) + '.pdf')

