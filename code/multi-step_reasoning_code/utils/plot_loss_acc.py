import matplotlib.pyplot as plt
import numpy as np
from .plot_settings import *
from utils import *
import argparse
from model import *
from .io_operate import *


def plot_loss_of_each_data(working_dir, x_axis='epoch'):
    r'''绘制一个实验中的每类数据的loss随epoch变化的曲线'''

    train_loss_his = np.load(f'{working_dir}/loss/train_loss_his.npy')
    test_loss_his = np.load(f'{working_dir}/loss/test_loss_his.npy')
    group_loss_his = np.load(f'{working_dir}/loss/group_loss_his.npy')

    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(fs=24, left=0.18, right=0.95, lw=2)
    ax = plt.gca()

    ax.semilogy(train_loss_his, label='total train loss', color='#c82423', linestyle='-', zorder=10)
    ax.semilogy(test_loss_his, label='total test loss', color='#2878b5', linestyle='-', zorder=10)
    
    args = read_json_data(f'{working_dir}/config.json')
    args = argparse.Namespace(**args)

    # 首先从一个色系中挑选颜色
    data_show_index = np.nonzero(args.data_show)[0]
    data_color_list = get_color_list(n_colors=len(data_show_index), cmap='viridis', color_min=0, color_max=0.9)
    data_color_list = ['#C82423', '#074166', '#FFBE7A', 'k', '#2681B6', '#FA7F6F'] + data_color_list
    
    for k, index in enumerate(data_show_index):
        if args.data_train[index] == 0:
            ax.plot(group_loss_his[:, index], label=f'loss of {args.data_name[index]}', 
                    color=data_color_list[k], ls='--', zorder=1)

    if x_axis == 'epoch':
        ax.set_xlabel('Epoch')
    elif x_axis == 'batch':
        ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')

    # legend
    ax.legend(loc='upper right', frameon=False, fontsize=18)

    plt.savefig(f'{working_dir}/loss_of_each_data.png')
    plt.close()


def plot_acc_of_each_data(working_dir):
    r'''绘制一个实验中具体类型数据的acc随epoch变化的曲线'''
    
    acc_epoch_his = np.load(f'{working_dir}/loss/acc_epoch_his.npy')
    group_acc_his = np.load(f'{working_dir}/loss/group_acc_his.npy')

    args = read_json_data(f'{working_dir}/config.json')
    args = argparse.Namespace(**args)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(wspace=0.4, hspace=0.6, bottom=0.16, fs=24, lw=3, ms=12.5, axlw=2.5, major_tick_len=10)
    ax = plt.gca()

    # 首先从一个色系中挑选颜色
    data_show_index = np.nonzero(args.data_show)[0]
    data_color_list = get_color_list(n_colors=len(data_show_index), cmap='viridis', color_min=0, color_max=0.9)
    
    for k, index in enumerate(data_show_index):
        ax.plot(acc_epoch_his, group_acc_his[:, index], label=f'{args.data_name[index]}', color=data_color_list[k], alpha=0.75, \
                marker = 'o', markersize=5, markeredgewidth=0.7, markeredgecolor='black', zorder=6)
        
    
    ax.set_xlabel('Epoch', labelpad=20)
    ax.set_ylabel('Accuracy', labelpad=20)

    plt.legend(loc=(0.6, 0.2), fontsize=18)

    plt.savefig(f'{working_dir}/acc_of_each_data.png')

    plt.close()







