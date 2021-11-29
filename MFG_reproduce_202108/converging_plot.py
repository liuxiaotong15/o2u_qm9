import numpy as np
import matplotlib.pyplot as plt

font = {'size' : 16} # , 'weight' : 'bold'}
plt.rc('font', **font)

fig = plt.figure(0)
ax = fig.add_subplot(111)
# fig, ax = plt.subplots()
y_EGPHS_EPHS_EHS_EH_E = [(0.4356911 + 0.44041097)/2] # 02923e5
y_EGPHS_EGPH_EGP_EG_E = [(0.48663643 + 0.4802916)/2] # 02923e5

y_EGPHS_EPHS_EHS_EH_E.append((0.3921176 + 0.4073958)/2) # b2ce0b0
y_EGPHS_EPHS_EHS_EH_E.append((0.39784262 + 0.42954674)/2) # d5648cd

ax.plot(list(range(len(y_EGPHS_EPHS_EHS_EH_E))), y_EGPHS_EPHS_EHS_EH_E, c='tab:blue', label='EGPHS-EPHS-EHS-EH-E')
ax.plot(list(range(len(y_EGPHS_EGPH_EGP_EG_E))), y_EGPHS_EGPH_EGP_EG_E, c='tab:red', label='EGPHS-EGPH-EGP-EG-E')

ax.scatter(list(range(len(y_EGPHS_EPHS_EHS_EH_E))), y_EGPHS_EPHS_EHS_EH_E, c='tab:blue', label='')
ax.scatter(list(range(len(y_EGPHS_EGPH_EGP_EG_E))), y_EGPHS_EGPH_EGP_EG_E, c='tab:red', label='')

# ymin, ymax = ax1.get_ylim()
# ax.set_ylim(0, 1.9)
# xmin, xmax = ax.get_xlim()
# ymin, ymax = ax.get_ylim()
ax.set_xlabel('Interation times')
ax.set_ylabel('Final MAE/eV')
plt.subplots_adjust(bottom=0.12, right=0.99, left=0.14, top=0.99)
plt.grid()
plt.legend()
plt.show()
