import pickle
import numpy as np
import matplotlib.pyplot as plt

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

labels = ['PBE', 'HSE', 'GLLB-SC', 'SCAN']

for idx, name in enumerate(['52348txt', '6030txt', '2290txt', '472txt']):
    fig, ax = plt.subplots()

    f = open(name, 'rb')
    data = pickle.load(f)
    f.close()
    data = np.array(data)
    
    print(data, np.mean(data), np.mean(np.abs(data)), np.max(data), np.min(data))
    print(data.shape)
    
    x = list(range(200))
    y = [0] * 200
    
    for i in range(data.shape[0]):
        y[int(data[i][0]*10)+100] += (1/1000)
    
    
    # for i in range(len(x)):
    #     x[i] = (x[i] -100)/10
    text = labels[idx] + '\nMean: ' + str(round(np.mean(data), 4))
    ax.text(120, max(y)*0.9, text, font_legend)
    ax.tick_params(labelsize=16)
    ax.bar(x[50: 150], y[50:150])
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], 
             ('-5','-4','-3','-2','-1', '0', '1', '2', '3', '4', '5'))
    ax.set_xlabel('Predict Error (eV)', font_axis)
    ax.set_ylabel('Count (k)', font_axis)

    plt.subplots_adjust(bottom=0.12, right=0.993, left=0.12, top=0.986)
    plt.show()
