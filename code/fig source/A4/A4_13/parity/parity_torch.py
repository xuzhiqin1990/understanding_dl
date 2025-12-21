# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Zhiqin Xu)s
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import product
# from BasicFunc import mySaveFig, univAprox 
import pickle
import torch
import torch.nn as nn
import time, os

isShowPic = 1
isTrain = 1
Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88 - Leftp
Heightp = 0.9 - Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]

# Verify the formula for Fourier transform (FT) of parity function.
def ft_full_thm(kk, n):
    return (-1J)**n * np.prod(np.sin(kk), axis=0)

def ft_thm(kk, s, n):
    return 1/s * (2**n - s)/(2**n - 1) * (1 - np.abs(ft_full_thm(kk, n))**2)

###
n = 10 
s = 200 
k_num = 100
k = np.linspace(0, np.pi/2, num=k_num, endpoint=True)
kk = np.matmul(np.ones([n, 1]), np.reshape(k, [1, -1]))

xs = 2 * (np.random.rand(n, s) < 0.5) - 1
x_input = np.transpose(xs)
y_input = np.reshape(np.prod(x_input, axis=1), [-1, 1])
ft_a_sample = np.matmul(np.transpose(y_input), np.exp(-1J * (np.matmul(x_input, kk)))) / s

ft_full_abs = abs(ft_full_thm(kk, n))
ft_var = ft_thm(kk, s, n)

#xlim([-1,1]*pi/2)
#legend('full FT', 'var sample', 'E sample', 'sample FT')
R_variable = {}
Dim_Input = n 
Rnd_FolderName = os.path.join('/root/zhangzhongwang/book/1d_fitting/parity','Parity%s_%s' % (Dim_Input, int(np.absolute(np.random.normal([1])) * 100000) // int(1)))
FolderName = '%s/' % (Rnd_FolderName)
os.mkdir(FolderName)
R_variable['FolderName'] = FolderName 
R_variable['Rnd_FolderName'] = Rnd_FolderName

### initialization standard deviation
R_variable['astddev'] = 0.05 # for weight
R_variable['bstddev'] = 0.05 # for bias terms  
R_variable['hidden_units'] = [500, 100]

# R_variable['hidden_units']=[800,800,400,400]

R_variable['learning_rate'] = 5e-4  
### setup for activation function
R_variable['seed'] = 0
R_variable['ActFuc'] = 1  ### 0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  
R_variable['isBN'] = False #Batch normalization 

R_variable['tol'] = 3e-7
R_variable['Total_Step'] = 1001  ### the training step. Set a big number, if it converges, can manually stop training
R_variable['loss'] = []

t0 = time.time() 

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units, ActFuc, astddev, bstddev):
        super(NeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.ActFuc = ActFuc
        self.astddev = astddev
        self.bstddev = bstddev
        
        self.layers = nn.ModuleList()
        prev_units = input_dim
        for units in hidden_units:
            self.layers.append(nn.Linear(prev_units, units))
            prev_units = units
        
        self.layers.append(nn.Linear(prev_units, 1))
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0, self.astddev)
            nn.init.normal_(layer.bias, 0, self.bstddev)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
    
    def activation(self, x):
        if self.ActFuc == 0:
            return torch.relu(x)
        elif self.ActFuc == 1:
            return torch.tanh(x)
        elif self.ActFuc == 3:
            return torch.sin(x)
        elif self.ActFuc == 4:
            return x**5
        elif self.ActFuc == 5:
            return torch.sigmoid(x)
        else:
            raise ValueError("Invalid activation function")

model = NeuralNetwork(Dim_Input, R_variable['hidden_units'], R_variable['ActFuc'], 
                      R_variable['astddev'], R_variable['bstddev'])

x_input = torch.tensor(x_input, dtype=torch.float32)
y_input = torch.tensor(y_input, dtype=torch.float32)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=R_variable['learning_rate'])

for epoch in range(R_variable['Total_Step']):
    y_pred = model(x_input)
    loss = criterion(y_pred, y_input)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    R_variable['loss'].append(loss.item())
    
    if epoch % 500 == 0:
        print('%s//%s, time:%s' % (epoch, R_variable['Total_Step'], time.time() - t0))
        print('epoch: %s, loss=%s' % (epoch, loss.item()))

if n > 14:        
    st = 50000        
    xst = 2 * (np.random.rand(n, st) < 0.5) - 1
    xt_input = torch.tensor(np.transpose(xst), dtype=torch.float32)
else:
    st = 2**n        
    xt_input = torch.tensor(list(product([-1, 1], repeat=n)), dtype=torch.float32)
yt_input = torch.tensor(np.reshape(np.prod(xt_input.numpy(), axis=1), [-1, 1]), dtype=torch.float32)

y_fit = model(xt_input).detach().numpy()
loss_tmp = criterion(torch.tensor(y_fit), yt_input)
ft_fit = np.matmul(np.transpose(y_fit), np.exp(-1J * (np.matmul(xt_input.numpy(), kk)))) / st

R_variable = {}
R_variable['k'] = k
R_variable['n'] = n
R_variable['ft_a_sample'] = ft_a_sample
R_variable['ft_full_abs'] = ft_full_abs
R_variable['ft_fit'] = ft_fit 

with open('%sobjs.pkl' % (FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(R_variable, f, protocol=4)

# ... (plotting code remains the same) ...
    
def format_settings(
        wspace=0.25, 
        hspace=0.4, 
        left=0.12, 
        right=0.9, 
        bottom=0.15, 
        top=0.95,
        fs=12,
        show_dpi=80,
        save_dpi=300,
        lw=1.5,
        ms=5,
        axlw=1.5,
        major_tick_len=5,
        major_tick_width=1.5,
        major_tick_pad=5,
        minor_tick_len=0,
        minor_tick_width=0,
        minor_tick_pad=5,
        ):
    '''
        使用方法：
            fig = plt.figure(figsize=(12, 4), dpi=300)
            format_settings()
            grid = plt.GridSpec(2, 2)
            ax1 = fig.add_subplot(grid[0, 0]) # 左上角图
            ax2 = fig.add_subplot(grid[0, 1]) # 右上角图
            ax3 = fig.add_subplot(grid[:, 0]) # 底部空间合并一张图
        注意：
            以上文字和坐标轴粗细适用于figsize长度为12的情形，宽度可调。
            若要调整figsize长度，需要相应调整以上文字和坐标轴粗细。
    '''
    # 设置子图线宽
    plt.rcParams['lines.linewidth'] = lw
    
    # 子图点大小
    plt.rcParams['lines.markersize'] = ms
    
    # 子图间距与位置  w:左右 h:上下
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

    # 字体大小
    plt.rcParams['font.size'] = fs
    plt.rcParams['axes.labelsize'] = fs
    plt.rcParams['axes.titlesize'] = fs
    plt.rcParams['xtick.labelsize'] =fs
    plt.rcParams['ytick.labelsize'] = fs
    plt.rcParams['legend.fontsize'] = fs
    # 子图坐标轴宽度
    plt.rcParams['axes.linewidth'] = axlw
    # 子图坐标轴可见性
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True

    # 子图坐标轴刻度宽度
    plt.rcParams['xtick.major.width'] = major_tick_width
    plt.rcParams['ytick.major.width'] = major_tick_width
    plt.rcParams['xtick.minor.width'] = minor_tick_width
    plt.rcParams['ytick.minor.width'] = minor_tick_width
    # 子图坐标轴刻度长度
    plt.rcParams['xtick.major.size'] = major_tick_len
    plt.rcParams['ytick.major.size'] = major_tick_len
    plt.rcParams['xtick.minor.size'] = minor_tick_len
    plt.rcParams['ytick.minor.size'] = minor_tick_len
    # 子图坐标轴刻度标签位置
    plt.rcParams['xtick.major.pad'] = major_tick_pad
    plt.rcParams['ytick.major.pad'] = major_tick_pad
    plt.rcParams['xtick.minor.pad'] = minor_tick_pad
    plt.rcParams['ytick.minor.pad'] = minor_tick_pad
    
    # 子图坐标轴刻度标签位置
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # 子图坐标轴刻度标签位置
    plt.rcParams['xtick.top'] = False 
    plt.rcParams['ytick.right'] = False
    # 子图坐标轴刻度标签位置
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.minor.visible'] = False
    # 子图坐标轴刻度标签位置
    plt.rcParams['legend.frameon'] = False
    # 子图坐标轴刻度标签位置
    plt.rcParams['figure.dpi'] = show_dpi
    # 子图坐标轴刻度标签位置
    plt.rcParams['savefig.dpi'] = save_dpi


plt.figure(figsize=(12, 8))
format_settings(left=0.15, right=0.99, bottom=0.15, fs=32, lw=4
        )

ax=plt.gca()
#plt.plot(k, abs(ft_var**0.5), label='sample_std_thm')
#plt.plot(k, np.sqrt(ft_full_abs ** 2 + ft_var), label='sample_mean_thm') 

plt.plot(k*n**0.5/2/np.pi, ft_full_abs,c='blue', linestyle='-.', label='all data')   
plt.plot(k*n**0.5/2/np.pi, abs(np.squeeze(ft_a_sample)), c='r', label='training data')   


plt.plot(k*n**0.5/2/np.pi, abs(np.squeeze(ft_fit)),c='g', linestyle='--',  label='model output on all data')      

ax.set_xlabel('frequency')
#ax.set_ylabel('|%s|'%( r'$F[f]$'),fontsize=18)
plt.rc('xtick')
plt.rc('ytick')
plt.legend()
# pos=[0.18,0.25,0.5,0.5]
# ax.set_position(pos, which='both')
#ax.set_yscale('log')
fntmp = '%sfftTrainCleanPeak'%(FolderName)
# mySaveFig(plt, fntmp)    
plt.savefig(fntmp + '.png', dpi=300, bbox_inches='tight')

