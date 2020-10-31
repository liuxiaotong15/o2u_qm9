import re
from matplotlib import pyplot as plt

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

fig, ax = plt.subplots()

commitid = 'acb1871'
# lgd = ['one by one', 'all together', 'only exp training', 'all->exp']
lgd = {6: '0', 8: '1', 9: '2'}

for idx in [6, 8, 9]:
    filename = commitid + "_" + str(idx) + ".log"
    y = []
    with open(filename, 'r') as f:
        for line in f:
            if 'MAE' in line:
                # print(line)
                result = re.match('.*avg_loss.*:  \[\[(.*)\]\]', line)
                if result:
                    # print(result.group(1))
                    y.append(float(result.group(1)))
    x = range(len(y))
    ax.plot(x, y, label=lgd[idx])
    ax.grid(True)

ax.tick_params(labelsize=16)
ax.legend(loc="upper right", prop=font_legend)
ax.set_xlabel('epoch', font_axis)
ax.set_ylabel('MAE on test', font_axis)
# ax.set_xlim(50, 60)
plt.subplots_adjust(bottom=0.12, right=0.993, left=0.12, top=0.986)
plt.show()