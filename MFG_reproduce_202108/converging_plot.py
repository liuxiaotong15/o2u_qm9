import numpy as np
import matplotlib.pyplot as plt

font = {'size' : 16} # , 'weight' : 'bold'}
plt.rc('font', **font)

fig = plt.figure(0)
ax = fig.add_subplot(111)
# fig, ax = plt.subplots()
y_EGPHS_EPHS_EHS_EH_E = [(0.4356911 + 0.44041097)/2] # 02923e5
y_EGPHS_EPHS_EPH_EH_E = [(0.44680655 + 0.43978375)/2] # 02923e5
y_EGPHS_EGPH_EGP_EG_E = [(0.48663643 + 0.4802916)/2] # 02923e5

y_EGPHS_EPHS_EHS_EH_E.append((0.3921176 + 0.4073958)/2) # b2ce0b0
y_EGPHS_EPHS_EHS_EH_E.append((0.39784262 + 0.42954674)/2) # d5648cd
y_EGPHS_EPHS_EHS_EH_E.append((0.3959625 + 0.44908467)/2) # 1b95780
y_EGPHS_EPHS_EHS_EH_E.append((0.4015885 + 0.4205016)/2) # 2cc7c8d
y_EGPHS_EPHS_EHS_EH_E.append((0.39822465 + 0.40219083)/2) # 3a74836
y_EGPHS_EPHS_EHS_EH_E.append((0.40432346 + 0.3843093)/2) # 1706f35
y_EGPHS_EPHS_EHS_EH_E.append((0.39979538 + 0.3843093)/2) # fa8524b
y_EGPHS_EPHS_EHS_EH_E.append((0.41638505 + 0.41045907)/2) # a0be536
y_EGPHS_EPHS_EHS_EH_E.append((0.40227702 + 0.3966312)/2) # ad14ad1


# y_EGPHS_EPHS_EPH_EH_E = [()/2] # 

y_EGPHS_EGPH_EGP_EG_E.append((0.4363386 + 0.45491073)/2) # 878c18d
y_EGPHS_EGPH_EGP_EG_E.append((0.42162254 + 0.4436562)/2) # 8332822
y_EGPHS_EGPH_EGP_EG_E.append((0.40554923 + 0.46846992)/2) # b63e3db 
y_EGPHS_EGPH_EGP_EG_E.append((0.41980428 + 0.44286364)/2) # 948740b
y_EGPHS_EGPH_EGP_EG_E.append((0.4140485 + 0.45038903)/2) # 7f5c7a8
y_EGPHS_EGPH_EGP_EG_E.append((0.40987274 + 0.44431028)/2) # 81b2455

ax.plot(list(range(len(y_EGPHS_EPHS_EHS_EH_E))), y_EGPHS_EPHS_EHS_EH_E, c='tab:blue', label='EGPHS-EPHS-EHS-EH-E')
ax.plot(list(range(len(y_EGPHS_EGPH_EGP_EG_E))), y_EGPHS_EGPH_EGP_EG_E, c='tab:red', label='EGPHS-EGPH-EGP-EG-E')
ax.plot(list(range(len(y_EGPHS_EPHS_EPH_EH_E))), y_EGPHS_EPHS_EPH_EH_E, c='tab:orange', label='EGPHS-EPHS-EPH-EH-E')

ax.scatter(list(range(len(y_EGPHS_EPHS_EHS_EH_E))), y_EGPHS_EPHS_EHS_EH_E, c='tab:blue', label='')
ax.scatter(list(range(len(y_EGPHS_EGPH_EGP_EG_E))), y_EGPHS_EGPH_EGP_EG_E, c='tab:red', label='')
ax.scatter(list(range(len(y_EGPHS_EPHS_EPH_EH_E))), y_EGPHS_EPHS_EPH_EH_E, c='tab:orange', label='')

# ymin, ymax = ax1.get_ylim()
# ax.set_ylim(0, 1.9)
# xmin, xmax = ax.get_xlim()
# ymin, ymax = ax.get_ylim()
ax.set_xlabel('Iteration times')
ax.set_ylabel('Final MAE/eV')
plt.subplots_adjust(bottom=0.12, right=0.99, left=0.14, top=0.99)
plt.grid()
plt.legend()
plt.show()
