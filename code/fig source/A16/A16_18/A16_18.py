import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os,re
import pandas as pd



config1 = 'wm_init-1_wd1'
config2 = 'wm_init-0.5_wd0.1'


# size2 = [0.02, 0.04, 0.08, 0.2, 0.5, 0.8, 1, 1.2, 1.4]
size2 = [0.2, 0.5, 0.8, 1, 1.2, 1.4]
size1 = [0.2, 0.5, 0.8, 1.2, 1]
# size1 = [0.2, 0.02, 0.04, 0.5, 0.8 ,0.08, 1.2, 1, 2, 10]

size1.sort()
size2.sort()
data_std = pd.read_csv('/mnt/public/code/yjj/ComplexityControl/data/scalinglaw/std0.02.csv')

data = {}
for i, config in enumerate([config1,config2]):
    data[config] = []
    for size in [size1,size2][i]:
        if i == 0:
            path = f'../data/scalinglaw/{config}/warmup_sl_scale-1_{size}B_3200steps.csv'
        else:
            path = f'../data/scalinglaw/{config}/warmup_sl_wd0.1_scale-0.5_{size}B_3200steps.csv'
        table = pd.read_csv(path)
        loss = table['Train/Samples/eval_loss'].values
        loss = loss[loss > 0]
        loss = np.min(loss)
        data[config].append(loss)


data['std0.02'] = []
for size in size2:
    x = data_std[f'{size}B'].tolist()
    x = [float(i) for i in x if str(i) != 'nan']
    data['std0.02'].append(np.min(x))


COLOR = ['#5EBEC2','#CC7AB0','#FF8946']
labelsize = 45
ticksize = 35
linewidth = 8
ms = 30
fig = plt.figure(figsize=(34, 12), dpi = 100)
grid = plt.GridSpec(5, 34, wspace=8, hspace=15, top=0.99, bottom=0.15, left=0.04, right=0.95)


ax = fig.add_subplot(grid[0:5, 1:17])
ax.spines['bottom'].set_linewidth(linewidth)#图框下边
ax.spines['left'].set_linewidth(linewidth)#图框左边
ax.spines['top'].set_linewidth(0)#图框上边
ax.spines['right'].set_linewidth(0)

ax.plot(size1, data[config1], '-.',label=f'$\gamma=1,\lambda=1$',linewidth=linewidth+5,c=COLOR[0],marker='s',markersize=ms)
ax.plot(size2, data[config2], '-.',label=f'$\gamma=0.5,\lambda=0.1$',linewidth=linewidth+5,c=COLOR[1],marker='s',markersize=ms)
ax.plot(size2, data['std0.02'], '-.',label=f'$\sigma=0.02,\lambda=0.1$',linewidth=linewidth+5,c=COLOR[2],marker='s',markersize=ms)

ax.set_ylabel('Test Loss', fontsize=labelsize)
ax.set_xlabel('Data Size (B)', fontsize=labelsize, labelpad=30)

ax.set_xscale('log')

ax.set_xticks([0.2,0.3,0.4,0.6,1.2])
ax.set_xticklabels(['$0.2$','$0.3$','$0.4$','$0.6$','$1.2$'])
# ax.set_xticklabels(['$2\\times 10^8$','$3\\times 10^8$','$4\\times 10^8$','$6\\times 10^8$','$1.2\\times 10^9$'])

# ax.set_yscale('log')

# yticks = ax.get_yticks()
# ax.set_yticks([yticks[]])
# xticks = ax.get_xticks()
# ax.set_xticklabels([x*10**9 for x in xticks], fontsize=labelsize)

ax.tick_params(axis='both', which='major', labelsize=ticksize,pad=20)
ax.tick_params(axis='both', which='minor', labelsize=ticksize,pad=20)

ax.legend(loc=(0.5,0.7), fontsize=labelsize-5, frameon=False)



config1 = '1B_init-1_wd1'
config2 = '1B_init-0.5_wd0.1'
config3 = '1B_initstd0.02_wd0.1'



datafile = pd.read_csv('/mnt/public/code/yjj/ComplexityControl/data/scalinglaw/modelsizesl.csv')

Size = [50,100,200,400,800]

data = {}
for i, config in enumerate([config1,config2,config3]):
    data[config] = []
    for size in Size:
        loss = datafile[f'{size}M_{config}'].values
        loss = loss[loss > 0]
        loss = np.min(loss)
        data[config].append(loss)
        



ax = fig.add_subplot(grid[0:5, 18:])
ax.spines['bottom'].set_linewidth(linewidth)#图框下边
ax.spines['left'].set_linewidth(linewidth)#图框左边
ax.spines['top'].set_linewidth(0)#图框上边
ax.spines['right'].set_linewidth(0)

ax.plot(Size,data[config1], '-.',label=f'$\gamma=1,\lambda=1$',linewidth=linewidth+5,c=COLOR[0],marker='s',markersize=ms)
ax.plot(Size, data[config2], '-.',label=f'$\gamma=0.5,\lambda=0.1$',linewidth=linewidth+5,c=COLOR[1],marker='s',markersize=ms)
ax.plot(Size, data[config3], '-.',label=f'$\sigma=0.02,\lambda=0.1$',linewidth=linewidth+5,c=COLOR[2],marker='s',markersize=ms)

# ax.set_ylabel('Test Loss', fontsize=labelsize)
ax.set_xlabel('Model Size (M)', fontsize=labelsize, labelpad=30)


ax.tick_params(axis='both', which='major', labelsize=ticksize,pad=20)
ax.tick_params(axis='both', which='minor', labelsize=ticksize,pad=20)


ax.set_xscale('log')
ax.set_xticks([50,100,200,400,800])
ax.set_xticklabels(['$50$','$100$','$200$','$400$','$800$'])

# ax.set_yscale('log')

# yticks = ax.get_yticks()
# ax.set_yticks([yticks[]])
# xticks = ax.get_xticks()
# ax.set_xticklabels([x*10**9 for x in xticks], fontsize=labelsize)

ax.legend(loc=(0.65,0.7), fontsize=labelsize-5, frameon=False)

plt.savefig('../figure/scalinglaw.png')
