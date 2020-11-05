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

labels = ['PBE', 'HSE', 'PBE', 'HSE', 'GLLB-SC', 'SCAN']

# for idx, name in enumerate(['52348txt', '6030txt', '52348.txt', '6030.txt', '2290txt', '472txt']):
#     fig, ax = plt.subplots()
# 
#     f = open(name, 'rb')
#     data = pickle.load(f)
#     f.close()
#     data = np.array(data)
#     
#     print(data, np.mean(data), np.mean(np.abs(data)), np.max(data), np.min(data))
#     print(data.shape)
#     
#     x = list(range(200))
#     y = [0] * 200
#     
#     for i in range(data.shape[0]):
#         y[int(data[i][0]*10)+100] += (1/1000)
#     
#     
#     # for i in range(len(x)):
#     #     x[i] = (x[i] -100)/10
#     text = labels[idx] + '\nMean: ' + str(round(np.mean(data), 4))
#     ax.text(120, max(y)*0.9, text, font_legend)
#     ax.tick_params(labelsize=16)
#     ax.bar(x[50: 150], y[50:150])
#     plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], 
#              ('-5','-4','-3','-2','-1', '0', '1', '2', '3', '4', '5'))
#     ax.set_xlabel('Predict Error (eV)', font_axis)
#     ax.set_ylabel('Count (k)', font_axis)
# 
#     plt.subplots_adjust(bottom=0.12, right=0.993, left=0.12, top=0.986)
#     plt.show()


# percentage part
for idx, name in enumerate(['52348_percentage_error.txt', '6030_percentage_error.txt']):
    fig, ax = plt.subplots()

    f = open(name, 'rb')
    data = pickle.load(f)
    f.close()
    data = np.array(data)
    
    print(data, np.mean(data), np.mean(np.abs(data)), np.max(data), np.min(data))
    print('np.sum(data>0): ', np.sum(data>0))
    print('np.sum(data<1): ', np.sum(data<1))
    print('np.sum(-2<data<2): ', np.sum((data>-2) & (data<2)))
    print(data.shape)
    sum_a = 0
    cnt = 0
    for a in data:
        if a>-2 and a<2:
            sum_a += a
            cnt += 1
    print(sum_a/cnt)
    
    x = list(range(400))
    y = [0] * 400
    
    for i in range(data.shape[0]):
        if data[i] > -2 and data[i] < 2:
            y[int(data[i]*100) + 200] += 1
    
    
    # for i in range(len(x)):
    #     x[i] = (x[i] -100)/10
    text = labels[idx] + ' percentage error \nMean: ' + \
            str(round((float)(sum_a/cnt)*100, 2))+'%\n' + \
            str(np.sum((data>-2) & (data<2))) + ' of ' + str(data.shape[0]) + ' samples'
    ax.text(220, max(y)*0.8, text, font_legend)
    ax.tick_params(labelsize=16)
    # ax.bar(x[50: 150], y[50:150])
    ax.bar(x, y)
    plt.xticks([0, 100, 150, 200, 250, 300, 400], 
             ('-200', '-100', '-50', '0', '50', '100', '200'))
    ax.set_xlabel('Predict Error (%)', font_axis)
    ax.set_ylabel('Count', font_axis)

    plt.subplots_adjust(bottom=0.12, right=0.993, left=0.12, top=0.986)
    plt.show()

