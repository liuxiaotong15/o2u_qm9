import re
import numpy as np

root = './bak_files_20210929/'
filenames = ["80c5815_0_123.log", "fe977d4_0_123.log", "6a33607_0_123.log", "e19744f_0_123.log"]

for filename in filenames:
    print(root+filename)
    value_dict = {'E':[], 'P':[], 'G':[], 'H':[], 'S':[],}
    subtree_value_dict = {'GPHS':[], 'EPHS':[], 'EGHS':[], 'EGPS':[], 'EGPH':[],}
    all_number_lst = []
    with open(root+filename, 'r') as f:
        for line in f.readlines():
            status = re.match(r'.*(tau_enable|load_old_model_enable).*', line.strip())
            if status:
                print(line.strip())
            ending_digits = re.match(r'.* is:\ \[(\d+(\.\d+)?)', line.strip())
            if ending_digits:
                all_number_lst.append(float(ending_digits.group(1)))
            for key in value_dict.keys():
                match_obj = re.match(r'.*_['+key+r']\ is:\ \[(.*)\]', line.strip())
                if match_obj:
                    value_dict[key].append(float(match_obj.group(1)))

            for key in subtree_value_dict.keys():
                match_obj = re.match(r'.*_'+key+r'_.*\ is:\ \[(.*)\]', line.strip())
                if match_obj:
                    subtree_value_dict[key].append(float(match_obj.group(1)))

    print('Ending with char test:')
    for key in value_dict.keys():
        print(key, np.mean(value_dict[key]))
    print('Subtree analysis:')
    for key in subtree_value_dict.keys():
        print(key, np.mean(subtree_value_dict[key]))
    print('All data analysis:')
    print('Max: ', max(all_number_lst))
    print('Min: ', min(all_number_lst))
    print('np.std: ', np.std(all_number_lst))
    print('np.mean: ', np.mean(all_number_lst))
    print('~' * 20)
