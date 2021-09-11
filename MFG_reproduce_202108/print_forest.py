items = ['G', 'H', 'S', 'P']
cur_str = 'GPHS'

dfs = []
latex_str = ''
idx = 0

def find_sub_tree(input_str):
    global dfs, latex_str, idx
    latex_str += '['
    latex_str += input_str
    latex_str += 'E'
    if len(input_str)>1:
        latex_str += '\\\\0.xx'
    else:
        latex_str += '\\\\\\#'
        latex_str += str(idx)
        idx += 1
    dfs.append(input_str)
    if len(input_str) != 1:
        for i in range(len(input_str)):
            next_str = input_str[:i] + input_str[i+1:]
            find_sub_tree(next_str)
    else:
        latex_str += '[E'
        latex_str += '\\\\\\#'
        latex_str += str(idx)
        idx += 1
        latex_str += ']'
    latex_str += ']'

find_sub_tree(cur_str)

print(dfs)
print(latex_str)

