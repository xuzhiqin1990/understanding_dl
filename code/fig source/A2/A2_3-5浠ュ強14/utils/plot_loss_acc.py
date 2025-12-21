import matplotlib.pyplot as plt
import numpy as np
import os
from .plot_settings import *
import seaborn as sns
import math
from utils import *
import argparse
from matplotlib.legend_handler import HandlerLine2D


def load_loss(working_dir, type = 'both'):
    if type == 'both':
        train_loss_his = np.load(f'{working_dir}/loss/train_loss_his.npy')
        test_loss_his = np.load(f'{working_dir}/loss/test_loss_his.npy')
        return train_loss_his, test_loss_his
    elif type == 'train':
        train_loss_his = np.load(f'{working_dir}/loss/train_loss_his.npy')
        return train_loss_his
    elif type == 'test':
        test_loss_his = np.load(f'{working_dir}/loss/test_loss_his.npy')
        return test_loss_his

def load_acc(working_dir, type = 'both'):
    acc_epoch_his = np.load(f'{working_dir}/loss/acc_epoch_his.npy')
    if type == 'both':
        try:
            train_acc_his = np.load(f'{working_dir}/loss/train_acc_his.npz', allow_pickle=True)
            test_acc_his = np.load(f'{working_dir}/loss/test_acc_his.npz', allow_pickle=True)
            return acc_epoch_his, np.array(train_acc_his), np.array(test_acc_his)
        except:
            train_acc_his = np.load(f'{working_dir}/loss/train_acc_his.npy')
            test_acc_his = np.load(f'{working_dir}/loss/test_acc_his.npy')
            return acc_epoch_his, train_acc_his, test_acc_his
    elif type == 'train':
        try:
            train_acc_his = np.load(f'{working_dir}/loss/train_acc_his.npz', allow_pickle=True)
            return acc_epoch_his, np.array(train_acc_his)
        except:
            train_acc_his = np.load(f'{working_dir}/loss/train_acc_his.npy')
            return acc_epoch_his, train_acc_his
    elif type == 'test':
        try:
            test_acc_his = np.load(f'{working_dir}/loss/test_acc_his.npz', allow_pickle=True)
            return acc_epoch_his, np.array(test_acc_his)
        except:
            test_acc_his = np.load(f'{working_dir}/loss/test_acc_his.npy')
            return acc_epoch_his, test_acc_his




def plot_loss(working_dir):
    r'''绘制一个实验中的loss随epoch变化的曲线'''

    train_loss_his = np.load(f'{working_dir}/loss/train_loss_his.npy')
    test_loss_his = np.load(f'{working_dir}/loss/test_loss_his.npy')

    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(fs=24)
    ax = plt.gca()

    ax.semilogy(train_loss_his, label='train loss', color='r', linestyle='-')
    ax.semilogy(test_loss_his, label='test loss', color='b', linestyle='-')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # legend
    ax.legend(loc='upper right', frameon=False)

    plt.savefig(f'{working_dir}/pic/loss_train_test.png')
    plt.close()





def plot_acc_of_mask_unmask_data(working_dir):
    r'''绘制一个实验中的train/test mask/unmask acc随epoch变化的曲线'''

    args = read_json_data(f'{working_dir}/config.json')
    args = argparse.Namespace(**args)

    acc_epoch_his = np.load(f'{working_dir}/loss/acc_epoch_his.npy')

    plot_mask = 0
    if os.path.exists(f'{working_dir}/loss/acc_train_mask_his.npy'):
        acc_train_mask_his = np.load(f'{working_dir}/loss/acc_train_mask_his.npy')
        acc_test_mask_his = np.load(f'{working_dir}/loss/acc_test_mask_his.npy')
        plot_mask = 1
    acc_train_unmask_his = np.load(f'{working_dir}/loss/acc_train_unmask_his.npy')
    acc_test_unmask_his = np.load(f'{working_dir}/loss/acc_test_unmask_his.npy')


    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(wspace=0.4, hspace=0.6, bottom=0.16, fs=24, lw=6, ms=12.5, axlw=2.5, major_tick_len=10)
    
    ax = plt.gca()
    handler_map = {}
    color_list = ['tomato', 'steelblue', 'lightsalmon', 'skyblue']
    l1, = ax.plot(acc_epoch_his, acc_train_unmask_his, label=f'seen prompt seen content', lw=2, color=color_list[0], 
            marker = 'o', markersize=9, markeredgewidth=1, markeredgecolor='black', zorder=6)
    l2, = ax.plot(acc_epoch_his, acc_test_unmask_his, label=f'seen prompt unseen content', color=color_list[1], 
            marker = 'o', markersize=13, markeredgewidth=1, markeredgecolor='black', zorder=5)
    
    handler_map[l1] = HandlerLine2D(numpoints=2)
    handler_map[l2] = HandlerLine2D(numpoints=2)
    handles = [l1, l2]

    # 如果存在mask的数据，则绘制mask数据的acc
    if plot_mask:
        l3, = ax.plot(acc_epoch_his, acc_train_mask_his, label=f'unseen prompt seen content', lw=2, color=color_list[2], 
                marker = '^', markersize=9, markeredgewidth=1, markeredgecolor='black', zorder=6)        
        l4, = ax.plot(acc_epoch_his, acc_test_mask_his, label=f'unseen prompt unseen content', color=color_list[3], 
                marker = '^', markersize=13, markeredgewidth=1, markeredgecolor='black', zorder=5)
        
        handler_map[l3] = HandlerLine2D(numpoints=2)
        handler_map[l4] = HandlerLine2D(numpoints=2)
        handles.extend([l3, l4])

    ax.set_xlabel('Epoch', labelpad=20, fontsize=24)
    ax.set_ylabel('Accuracy', labelpad=20, fontsize=24)

    # if acc_epoch_his[-1] <= 1000:
    #     ax.set_xticks([0, 200, 400, 600, 800])
    #     ax.set_xticklabels([0, 200, 400, 600, 800])
    # else:
    #     ax.set_xticks([0, 1000, 2000, 3000, 4000])
    #     ax.set_xticklabels([0, 1000, 2000, 3000, 4000])

    # ax.set_xlim(-10, acc_epoch_his[-1] * 1.05)

    plt.legend(handler_map=handler_map, handles=handles, loc=(0.35, 0.6))

    plt.savefig(f'{working_dir}/pic/acc_of_mask_unmask_data.png')

    plt.close()


def plot_acc_of_each_data(working_dir):
    r'''绘制一个实验中具体类型数据的acc随epoch变化的曲线'''
    acc_epoch_his, train_acc_his, test_acc_his = load_acc(working_dir)

    args = read_json_data(f'{working_dir}/config.json')
    args = argparse.Namespace(**args)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(wspace=0.4, hspace=0.6, bottom=0.16, fs=24, lw=6, ms=12.5, axlw=2.5, major_tick_len=10)
    ax = plt.gca()

    # 首先从一个色系中挑选颜色
    data_show_index = np.nonzero(args.data_show)[0]
    data_color_list = get_color_list(n_colors=len(data_show_index), cmap='viridis', color_min=0, color_max=1)
    
    for k, index in enumerate(data_show_index):
        if args.data_mask[index] == 0:
            marker = 'o'
        else:
            marker = '^'
        ax.plot(acc_epoch_his, train_acc_his[:, index], label=f'train ({args.data_name[index]})', color=data_color_list[k], alpha=0.75, \
                marker = marker, markersize=9, markeredgewidth=1, markeredgecolor='black', zorder=6)
        ax.plot(acc_epoch_his, test_acc_his[:, index], label=f'test  ({args.data_name[index]})', color=data_color_list[k], alpha=0.75, \
                marker = marker, markersize=13, markeredgewidth=1, markeredgecolor='black', zorder=5)
    
    ax.set_xlabel('Epoch', labelpad=20)
    ax.set_ylabel('Accuracy', labelpad=20)

    # ax.set_xticks([0, 1000, 2000, 3000, 4000])
    # ax.set_xticklabels([0, 1000, 2000, 3000, 4000])

    plt.legend(loc=(0.6, 0.2))

    plt.savefig(f'{working_dir}/pic/acc_of_each_data.png')

    plt.close()






def plot_acc_of_mask_unmask_with_datasize_together(exp_dir, datasize_list, seed_list, target='3x_to_x', suffix=''):
    r'''
        绘制最终实验结果和数据量的关系
        每个seed画一个多图
        多图中，每个子图画acc
    '''
    
    row_num = math.ceil(len(datasize_list) / 2)

    color_list = ['m', 'teal', 'magenta', 'c']

    for i, s in enumerate(seed_list):

        fig = plt.figure(figsize=(12, 4 * row_num), dpi=300)
        format_settings(wspace=0.4, hspace=0.3, bottom=0.1, left=0.25, lw=1, ms=3)
        grid = plt.GridSpec(row_num, 2)

        for j, N in enumerate(datasize_list):
            working_dir = f'{exp_dir}/{target}-seed_{s}-N_{N}'

            if suffix != '':
                working_dir += f'-{suffix}'

            acc_epoch_his = np.load(f'{working_dir}/loss/acc_epoch_his.npy')

            plot_mask = 0
            if os.path.exists(f'{working_dir}/loss/acc_train_mask_his.npy'):
                acc_train_mask_his = np.load(f'{working_dir}/loss/acc_train_mask_his.npy')
                acc_test_mask_his = np.load(f'{working_dir}/loss/acc_test_mask_his.npy')
                plot_mask = 1
            acc_train_unmask_his = np.load(f'{working_dir}/loss/acc_train_unmask_his.npy')
            acc_test_unmask_his = np.load(f'{working_dir}/loss/acc_test_unmask_his.npy')

            
            ax = fig.add_subplot(grid[math.ceil(j//2), j%2])

            if j == 0:
                handler_map = {}
            color_list = ['tomato', 'steelblue', 'lightsalmon', 'skyblue']
            l1, = ax.plot(acc_epoch_his, acc_train_unmask_his, label=f'seen prompt seen content', lw=2, color=color_list[0], 
                    marker = 'o', markersize=5, markeredgewidth=1, markeredgecolor='black', zorder=6)
            l2, = ax.plot(acc_epoch_his, acc_test_unmask_his, label=f'seen prompt unseen content', color=color_list[1], 
                    marker = 'o', markersize=6, markeredgewidth=1, markeredgecolor='black', zorder=5)
            
            if j == 0:
                handler_map[l1] = HandlerLine2D(numpoints=2)
                handler_map[l2] = HandlerLine2D(numpoints=2)
                handles = [l1, l2]

            # 如果存在mask的数据，则绘制mask数据的acc
            if plot_mask:
                l3, = ax.plot(acc_epoch_his, acc_train_mask_his, label=f'unseen prompt seen content', lw=2, color=color_list[2], 
                        marker = '^', markersize=5, markeredgewidth=1, markeredgecolor='black', zorder=6)        
                l4, = ax.plot(acc_epoch_his, acc_test_mask_his, label=f'unseen prompt unseen content', color=color_list[3], 
                        marker = '^', markersize=6, markeredgewidth=1, markeredgecolor='black', zorder=5)
                
                if j == 0:
                    handler_map[l3] = HandlerLine2D(numpoints=2)
                    handler_map[l4] = HandlerLine2D(numpoints=2)
                    handles.extend([l3, l4])


            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_ylabel('Accuracy')

            ax.set_ylim(-0.1, 1.1)

            # legend
            if j == 0:
                ax.legend(loc=(-1, 0), handles=handles, frameon=False, handler_map=handler_map)

            # title
            ax.set_title(f'train data size = {N}')

        if suffix != '':
            plt.savefig(f'{exp_dir}/process_loss_acc_datasize_seed{s}_{suffix}.png')
        else:
            plt.savefig(f'{exp_dir}/process_loss_acc_datasize_seed{s}.png')
        plt.close()

