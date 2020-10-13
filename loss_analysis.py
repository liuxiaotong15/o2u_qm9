import pickle
import numpy as np
from ase.db import connect
from base64 import b64encode, b64decode

filename = 'qm9.db'
db = connect(filename)
rows = list(db.select(sort='id'))

G = "free_energy"
def get_data_pp(idx, type):
    # extract properties
    prop = 'None'
    row = rows[idx-1]
    if(row.id != idx):
        1/0
    # extract from schnetpack source code
    shape = row.data["_shape_" + type]
    dtype = row.data["_dtype_" + type]
    prop = np.frombuffer(b64decode(row.data[type]), dtype=dtype)
    prop = prop.reshape(shape)
    return prop


def load_loss(filename):
    f = open(filename, 'rb')
    loss = pickle.load(f)
    loss = np.array(loss)
    loss = np.squeeze(loss)
    f.close()
    return loss

def top_k_idx(filename):
    f = open(filename, 'rb')
    target = np.squeeze(np.array(pickle.load(f)))
    f.close()
    ret = []
    for i in range(target.shape[0]):
        ratio = target[i]/get_data_pp(i+1, G)
        if ratio > 1.01 or ratio < 0.99:
            ret.append(i)
    return np.array(ret)

def handle_commit_id(cmt_id):
    top_k_id_ans = top_k_idx("targets_" + cmt_id + ".pickle")
    for i in range(50):
        try:
            loss = load_loss("losses_" + cmt_id + "_" + str(i) + ".pickle")
        except:
            break
        else:
            # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
            x = -1*top_k_id_ans.shape[0]
            ind = np.argpartition(loss, x)[x:]
            # cal how many ids both have
            both = list(set(top_k_id_ans).intersection(set(ind)))

            if top_k_id_ans.shape[0] > 0:
                print('ans:', len(set(top_k_id_ans)),
                    'max loss:', len(set(ind)),
                    'intersection:', len(both),
                    'topk found ratio:', len(both)/len(top_k_id_ans))

            print('loss mean:', np.mean(loss),
                    'loss std:', np.std(loss),
                    )
    print('-' * 30)

if __name__ == "__main__":
    for cmt in ['3ad9df1', '9a6632f']:
        print(cmt, ":")
        handle_commit_id(cmt)

