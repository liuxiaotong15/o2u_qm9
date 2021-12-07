import sys
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
from scipy.stats import norm
import seaborn as sns

def draw():
    global sets, df, colors
    
    all_custom_lines = {}
    for s in sets:
        all_custom_lines[s] = Line2D([0], [0], color=colors[s], lw = 2)
        
    fig = plt.figure(figsize=(13, 26), constrained_layout=True)
    # gs = fig.add_gridspec(nrows=10, ncols=5, hspace=0.5)
    gs = fig.add_gridspec(nrows=10, ncols=5, hspace=0.10, wspace=0.05)
    for i in range(len(sets)):
        tups = list(combinations(sets, i+1))
        z = 10 / len(tups)
        for j, tup in enumerate(tups):
            ax = fig.add_subplot(gs[int(j*z):int((j+1)*z), i])
            
            mps = df[tup[0]]['mp_id'].values
            for k, s in enumerate(tup):
                if k == 0:
                    continue
                else:
                    another = df[s]['mp_id'].values
                    mps = [x for x in mps if x in another]
            
            data = {}
            custom_lines = []
            custom_word = []
            
            for s in tup:
                gap = []
                s_id = df[s]['mp_id'].values
                s_gap = df[s]['gap'].values
                for k, mp_id in enumerate(s_id):
                    if mp_id in mps:
                        gap.append(s_gap[k])
                        
                mean = np.mean(gap)
                var = np.var(gap)
                data[mean] = [s, mean, np.sqrt(var)]
                
                sns.distplot(gap, ax=ax, hist=False, kde_kws={"color": colors[s]})
            
            data = sorted(data.items(), key=lambda x:x[0])
            
            for key, value in data:
                custom_lines.append(all_custom_lines[value[0]])
                custom_word.append(f'{value[0]} ({round(value[1], 2)}, {round(value[2], 2)})')
            
            ax.set_xlim((0, 18))
            ax.set_ylim((0, 1))
            ax.get_yaxis().set_visible(False)
            
            # ax.legend(custom_lines, custom_word, bbox_to_anchor =(-1.5, 1.01), loc=8, ncol=5, fontsize = 13)
            ax.legend(custom_lines, custom_word)
            ax.tick_params(labelsize=13)
            
    # plt.subplots_adjust(top=0.94,bottom=0.01,left=0.005,right=0.99,hspace=0.2,wspace=0.2)
    # plt.subplots_adjust(top=0.94,bottom=0.040,left=0.040,right=0.99,hspace=0.2,wspace=0.2)
    fig.savefig("distribution_v3.pdf")
    plt.show()        
    
def main():
    global sets, df, colors
    sets = ['P', 'H', 'S', 'G', 'E']
    colors = {'P':'#008000', 'E':'#bf00bf', 'H':'#ffa500', 'G':'#ff0000', 'S':'#0000ff'}
    
    df = {}
    for s in sets:
        df[s] = pd.read_csv('5set/'+s+'.csv')
    
    draw()
    
if __name__ == '__main__':
    main()
