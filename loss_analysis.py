import pickle
import numpy as np

def load_loss(filename):
    f = open(filename, 'rb')
    loss = pickle.load(f)
    loss = np.array(loss)
    loss = np.squeeze(loss)
    f.close()
    return loss

def handle_commit_id(cmt_id):
    for i in range(50):
        try:
            loss = load_loss("losses_" + cmt_id + "_" + str(i) + ".pickle")
            print(np.mean(loss), np.std(loss))
        except:
            break
    print('-' * 30)

if __name__ == "__main__":
    for cmt in ['3ad9df1', '9a6632f']:
        print(cmt, ":")
        handle_commit_id(cmt)

