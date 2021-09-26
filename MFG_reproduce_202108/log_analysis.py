import re
import numpy as np

filenames = ["80c5815_0_123.log", "fe977d4_0_123.log"]

for filename in filenames:
    print(filename)
    value_dict = {'E':[], 'P':[], 'G':[], 'H':[], 'S':[],}
    with open(filename, 'r') as f:
        for line in f.readlines():
            for key in value_dict.keys():
                match_obj = re.match(r'.*_['+key+r']\ is:\ \[(.*)\]', line.strip())
                if match_obj:
                    value_dict[key].append(float(match_obj.group(1)))
    
    for key in value_dict.keys():
        print(key, np.mean(value_dict[key]))

    print('~' * 20)
