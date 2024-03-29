import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'Arial', 'size' : 16}
plt.rc('font', **font)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

# font = {'size' : 16} # , 'weight' : 'bold'}
# plt.rc('font', **font)

fig = plt.figure(0)
ax = fig.add_subplot(111)
# fig, ax = plt.subplots()
y_EGPHS_EPHS_EHS_EH_E = [(0.4356911 + 0.44041097)/2] # 02923e5
y_EGPHS_EPHS_EPH_EH_E = [(0.44680655 + 0.43978375)/2] # 02923e5
y_EGPHS_EGPH_EGP_EG_E = [(0.48663643 + 0.4802916)/2] # 02923e5
y_EGPHS = [(0.48395586 + 0.518767)/2] # 02923e5
y_S_G_P_E_H = [(0.46390945 + 0.50388217)/2] # a367e11

y_EGPHS_EPHS_EHS_EH_E.append((0.3921176 + 0.4073958)/2) # b2ce0b0
y_EGPHS_EPHS_EHS_EH_E.append((0.39784262 + 0.42954674)/2) # d5648cd
y_EGPHS_EPHS_EHS_EH_E.append((0.3959625 + 0.44908467)/2) # 1b95780
y_EGPHS_EPHS_EHS_EH_E.append((0.4015885 + 0.4205016)/2) # 2cc7c8d
y_EGPHS_EPHS_EHS_EH_E.append((0.39822465 + 0.40219083)/2) # 3a74836
y_EGPHS_EPHS_EHS_EH_E.append((0.40432346 + 0.3843093)/2) # 1706f35
y_EGPHS_EPHS_EHS_EH_E.append((0.39979538 + 0.40304384)/2) # fa8524b
y_EGPHS_EPHS_EHS_EH_E.append((0.41638505 + 0.41045907)/2) # a0be536
y_EGPHS_EPHS_EHS_EH_E.append((0.40227702 + 0.3966312)/2) # ad14ad1
y_EGPHS_EPHS_EHS_EH_E.append((0.4140424 + 0.39396235)/2) # 5ad510d


y_EGPHS_EPHS_EPH_EH_E.append((0.43743765 + 0.41969687)/2) # e2840dc
y_EGPHS_EPHS_EPH_EH_E.append((0.43375596 + 0.42336014)/2) # 41abe3a
y_EGPHS_EPHS_EPH_EH_E.append((0.4207398 + 0.41163453)/2) # 974fe4d
y_EGPHS_EPHS_EPH_EH_E.append((0.4293764 + 0.41935813)/2) # b4ad894
y_EGPHS_EPHS_EPH_EH_E.append((0.42132148 + 0.42718315)/2) # 2f8a491
y_EGPHS_EPHS_EPH_EH_E.append((0.41400704 + 0.43106917)/2) # ea40676
y_EGPHS_EPHS_EPH_EH_E.append((0.41004536 + 0.43017095)/2) # 9e9fb16
y_EGPHS_EPHS_EPH_EH_E.append((0.39446253 + 0.40899158)/2) # 99602b6
y_EGPHS_EPHS_EPH_EH_E.append((0.41409662 + 0.42113072)/2) # eaa7ee7
y_EGPHS_EPHS_EPH_EH_E.append((0.42787403 + 0.42067167)/2) # 316a10e

y_EGPHS_EGPH_EGP_EG_E.append((0.4363386 + 0.45491073)/2) # 878c18d
y_EGPHS_EGPH_EGP_EG_E.append((0.42162254 + 0.4436562)/2) # 8332822
y_EGPHS_EGPH_EGP_EG_E.append((0.40554923 + 0.46846992)/2) # b63e3db 
y_EGPHS_EGPH_EGP_EG_E.append((0.41980428 + 0.44286364)/2) # 948740b
y_EGPHS_EGPH_EGP_EG_E.append((0.4140485 + 0.45038903)/2) # 7f5c7a8
y_EGPHS_EGPH_EGP_EG_E.append((0.40987274 + 0.44431028)/2) # 81b2455
y_EGPHS_EGPH_EGP_EG_E.append((0.41279027 + 0.44182006)/2) # fd3e9b5
y_EGPHS_EGPH_EGP_EG_E.append((0.41560054 + 0.45912856)/2) # cc394f6
y_EGPHS_EGPH_EGP_EG_E.append((0.42018202 + 0.4460266)/2) # ab5c964
y_EGPHS_EGPH_EGP_EG_E.append((0.42871267 + 0.43525925)/2) # 42a782f

y_EGPHS.append((0.4662436 + 0.4997614)/2) # caa9cd7
y_EGPHS.append((0.46129665 + 0.4822263)/2) # 18a78db
y_EGPHS.append((0.49407262 + 0.4582636)/2) # 5c3777a
y_EGPHS.append((0.47930935 + 0.46569428)/2) # 67f0e9f
y_EGPHS.append((0.4823242 + 0.46832398)/2) # e225da0
y_EGPHS.append((0.47294405 + 0.47484007)/2) # a451992
y_EGPHS.append((0.47872645 + 0.47592255)/2) # b2e2dcb
y_EGPHS.append((0.4766461 + 0.4800091)/2) # 96cbfec
y_EGPHS.append((0.496164 + 0.46963057)/2) # f6f98f0
y_EGPHS.append((0.48968768 + 0.485809)/2) # 734a74d

y_S_G_P_E_H.append((0.4673425 + 0.49684522)/2) # 4f2a524
y_S_G_P_E_H.append((0.46716988 + 0.5022696)/2) # 9e07ad6
y_S_G_P_E_H.append((0.46527553 + 0.5055876)/2) # 87aa059
y_S_G_P_E_H.append((0.47629127 + 0.49670452)/2) # 2c2a499
y_S_G_P_E_H.append((0.48102254 + 0.51227736)/2) # 1df3baf
y_S_G_P_E_H.append((0.48561862 + 0.51600176)/2) # 6362c43
y_S_G_P_E_H.append((0.49886718 + 0.5067701)/2) # 9c204bb
y_S_G_P_E_H.append((0.50502914 + 0.5285873)/2) # 44d273f
y_S_G_P_E_H.append((0.50383013 + 0.53812706)/2) # e844487
y_S_G_P_E_H.append((0.49431038 + 0.52804875)/2) # 1e82853

ax.plot(list(range(len(y_EGPHS_EPHS_EHS_EH_E))), y_EGPHS_EPHS_EHS_EH_E, label='EGPHS-EPHS-EHS-EH-E', marker='o')
ax.plot(list(range(len(y_EGPHS_EGPH_EGP_EG_E))), y_EGPHS_EGPH_EGP_EG_E, label='EGPHS-EGPH-EGP-EG-E', marker='^')
ax.plot(list(range(len(y_EGPHS_EPHS_EPH_EH_E))), y_EGPHS_EPHS_EPH_EH_E, label='EGPHS-EPHS-EPH-EH-E', marker='*')
ax.plot(list(range(len(y_EGPHS))), y_EGPHS, label='EGPHS', marker='<')
ax.plot(list(range(len(y_S_G_P_E_H))), y_S_G_P_E_H, label='S-G-P-E-H', marker='D')

# ax.scatter(list(range(len(y_EGPHS_EPHS_EHS_EH_E))), y_EGPHS_EPHS_EHS_EH_E, c='tab:blue', label='')
# ax.scatter(list(range(len(y_EGPHS_EGPH_EGP_EG_E))), y_EGPHS_EGPH_EGP_EG_E, c='tab:red', label='')
# ax.scatter(list(range(len(y_EGPHS_EPHS_EPH_EH_E))), y_EGPHS_EPHS_EPH_EH_E, c='tab:orange', label='')
# ax.scatter(list(range(len(y_EGPHS))), y_EGPHS, c='tab:brown', label='')
# ax.scatter(list(range(len(y_S_G_P_E_H))), y_S_G_P_E_H, c='tab:green', label='')

# ymin, ymax = ax1.get_ylim()
# ax.set_ylim(0, 1.9)
# xmin, xmax = ax.get_xlim()
# ymin, ymax = ax.get_ylim()
ax.set_xlabel('Iteration times')
ax.set_ylabel('Final MAE, eV')
plt.subplots_adjust(bottom=0.12, right=0.99, left=0.14, top=0.885)
plt.grid()
plt.legend(bbox_to_anchor=(-0.175, 0.99, 1, 0.2), loc="lower left", prop={'size': 11}, ncol=3)
plt.show()
