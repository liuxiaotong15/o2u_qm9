import numpy as np
import matplotlib.pyplot as plt

font = {'size' : 16} # , 'weight' : 'bold'}
plt.rc('font', **font)

fig = plt.figure(0)
ax = fig.add_subplot(111)
# fig, ax = plt.subplots()
x = [0, 1, 2, 3, 4]
y = [0.5, 0.4, 0.35, 0.33, 0.32]
y1 = [0.8, 0.3, 0.32, 0.31, 0.30]
ax.plot(x, y, c='tab:blue', label='EGPHS-EPHS-EHS-EH-E')
ax.plot(x, y1, c='tab:red', label='EGPHS-EGPH-EGP-EG-E')
ax.scatter(x, y, c='tab:blue', label='')
ax.scatter(x, y1, c='tab:red', label='')

# ymin, ymax = ax1.get_ylim()
# ax.set_ylim(0, 1.9)
# xmin, xmax = ax.get_xlim()
# ymin, ymax = ax.get_ylim()
ax.set_xlabel('Interation times')
ax.set_ylabel('Final MAE/eV')
plt.subplots_adjust(bottom=0.12, right=0.99, left=0.13, top=0.99)
plt.grid()
plt.legend()
plt.show()
