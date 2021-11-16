import collections
import re
import numpy as np

# root = './bak_files_20210929/'
# filenames = ["80c5815_0_123.log", "fe977d4_0_123.log", "6a33607_0_123.log", "e19744f_0_123.log"]
root = './'
# filenames = ["02923e5_1_123.log", "02923e5_0_123.log"]
# filenames = ["ff14c38_0_123.log", "ff14c38_1_123.log"]
# filenames = ["ff4e437_0_123.log", "ff4e437_1_123.log"]
filenames = ["a367e11_0_123.log", "a367e11_1_123.log"]

for filename in filenames:
    print(root+filename)
    value_dict = {'E':[], 'P':[], 'G':[], 'H':[], 'S':[],}
    subtree_value_dict = {'GPHS':[], 'EPHS':[], 'EGHS':[], 'EGPS':[], 'EGPH':[],}
    all_number_lst = []
    best_str, worst_str = '', ''
    best_ending_e_str, worst_ending_e_str = '', ''
    max_mae, min_mae = 0, 100
    max_ending_e_mae, min_ending_e_mae = 0, 100
    with open(root+filename, 'r') as f:
        for line in f.readlines():
            cc = collections.Counter(line)
            # 1by1 all dataset training once == 8 '_' in the string...
            # if cc['_'] != 8:
            #     continue
            status = re.match(r'.*(tau_enable|load_old_model_enable|EGPHS\ is).*', line.strip())
            if status:
                print(line.strip())
            ending_digits = re.match(r'.* is:\ \[(\d+(\.\d+)?)', line.strip())
            if ending_digits:
                number = float(ending_digits.group(1))
                all_number_lst.append(number)
                if number > max_mae:
                    max_mae = number
                    worst_str = line
                if number < min_mae:
                    min_mae = number
                    best_str = line

            for key in value_dict.keys():
                match_obj = re.match(r'.*_['+key+r']\ is:\ \[(.*)\]', line.strip())
                if match_obj:
                    number = float(match_obj.group(1))
                    value_dict[key].append(number)
                    if key == 'E':
                        if number > max_ending_e_mae:
                            max_ending_e_mae = number
                            worst_ending_e_str = line
                        if number < min_ending_e_mae:
                            min_ending_e_mae = number
                            best_ending_e_str = line

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
    print('Max: ', max(all_number_lst), worst_str)
    print('Min: ', min(all_number_lst), best_str)
    print('Best of ending E: ', min_ending_e_mae, best_ending_e_str)
    print('Worse of ending E: ', max_ending_e_mae, worst_ending_e_str)
    print('np.std: ', np.std(all_number_lst))
    print('np.mean: ', np.mean(all_number_lst))
    print('~' * 20)
