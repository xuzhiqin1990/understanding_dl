import matplotlib.pyplot as plt
from .plot_settings import *

def plot_info_broadcast(input_seq, output_seq, attn_list, key_points=[], key_flows=[], res_flows=[], 
                        save_path=None):
    
    fig = plt.figure(figsize=(6,3), dpi=100)
    format_settings(ms=5, major_tick_len=0, fs=10, axlw=0, lw=0, right=0.98, left=0.18, bottom=0.05, top=0.95)

    seq_len = attn_list[0].shape[0]
    layers = len(attn_list) + 1

    color_list = ['#9bbbe1', '#08519C']

    # 在每个纵坐标为4*i的位置，画seq_len个点，表示每层的每个位置
    for i in range(layers):
        for j in range(seq_len):
            if len(key_points)== 0 or (i, j) in key_points:
                plt.scatter(j, 4*i, c=color_list[j%2], edgecolors='k', 
                            s=80, linewidths=1, zorder=10)
            else:
                plt.scatter(j, 4*i, c='#999999', edgecolors='k', 
                        s=80, linewidths=1, zorder=10)
        
    # 画输入输出序列
    b = 0.3
    for j, x in enumerate(input_seq):
        plt.text(j, -1-b, x, ha='center', va='center')
    for j, x in enumerate(output_seq):
        plt.text(j, 4*(layers-1)+1+b, x, ha='center', va='center')

    # 依照attn的值，画每层的连接线，attn值越大，线越粗
    for i, attn in enumerate(attn_list):
        for j in range(seq_len):
            for k in range(seq_len):
                if len(key_flows)== 0 or (i, j, k) in key_flows:
                    plt.plot([j, k], [4*i, 4*(i+1)], c='#3e608d', lw=attn[k, j]*2, zorder=3)
                else:
                    plt.plot([j, k], [4*i, 4*(i+1)], c='#999999', lw=attn[k, j]*2, zorder=1, alpha=0.8)

    # 画residual的连接线
    for i, j in res_flows:
        plt.plot([j, j], [4*i, 4*(i+1)], c='#3e608d', ls='--', lw=1, zorder=2)

    plt.xticks([], [])
    plt.yticks([-0.9-b] + [4*i for i in range(layers)] + [4*(layers-1)+1.1+b], ['Input seq'] + [f'Layer {i}' for i in range(layers)] + ['Output seq'])

    if save_path is not None:
        plt.savefig(save_path, dpi=300)