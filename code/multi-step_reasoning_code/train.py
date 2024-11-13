import torch
from torch import nn, optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler 

from model import *
from utils import *
from data import *

def train_step(args, model, train_data_loader, optimizer, criterion, device, clip=1, scheduler=None):
    model.train()
    epoch_loss = 0
    total_samples = 0
    
    for i, (dec_inputs, dec_outputs) in enumerate(train_data_loader):  
        optimizer.zero_grad()
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs)
        
        batch_size = dec_inputs.size(0)  # 获取当前批次的实际大小
        total_samples += batch_size
        
        loss = criterion(outputs.view(batch_size, args.seq_len, args.vocab_size)[:,-1,:], dec_outputs[:,-1].view(-1))

        epoch_loss += loss.item() * batch_size  # 将损失乘以批次大小
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()

    return epoch_loss / total_samples  # 返回平均损失


def test_step(args, model, test_data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    total_samples = 0
    
    for i, (dec_inputs, dec_outputs) in enumerate(test_data_loader):
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs)
        
        batch_size = dec_inputs.size(0)  # 获取当前批次的实际大小
        total_samples += batch_size
        
        loss = criterion(outputs.view(batch_size, args.seq_len, args.vocab_size)[:,-1,:], dec_outputs[:,-1].view(-1))
        
        epoch_loss += loss.item() * batch_size  # 将损失乘以批次大小
    
    return epoch_loss / total_samples  # 返回平均损失



# 批量预测
def last_word_acc(args, model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total_samples = 0
    
    for i, (dec_inputs, dec_outputs) in enumerate(data_loader):
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)


        outputs, _ = model(dec_inputs)
        
        batch_size = dec_inputs.size(0)  # 获取当前批次的实际大小
        total_samples += batch_size
        
        outputs = outputs.argmax(axis=-1).view(-1, args.seq_len)
        correct += (outputs[:, -1] == dec_outputs[:, -1]).sum().item()
    
    return correct / total_samples


def get_accuracy(args, model, data_loader_group, train_percent, test_percent, my_logger):
    '''
        计算每类数据的acc，返回train_acc, test_acc, acc_list
    '''
    train_acc = 0
    test_acc = 0
    acc_list = []
    
    # 针对每类数据分别计算acc
    for i, data_name in enumerate(args.data_name):
        data_loader = data_loader_group[data_name]

        # 准确率
        tmp_acc = last_word_acc(args, model, data_loader)
        acc_list.append(tmp_acc)

        if args.data_train[i] == 1:
            train_acc += tmp_acc * args.data_percent[i] / train_percent
        else:
            test_acc += tmp_acc * args.data_percent[i] / test_percent

        my_logger.info(f'data type: {data_name:<25} Acc: {tmp_acc * 100:.4f}%')

    return train_acc, test_acc, acc_list



def _get_loss_of_each_data(args, model, data_loader_group, criterion, device):
    '''
        计算data_train=0的每类数据的loss，返回每类数据的loss和总loss
        对于训练数据，因数据量大不便计算，直接返回0
    '''
    test_loss = 0
    total_samples = 0
    loss_list = []
    loss_str = ''
    for i, data_name in enumerate(args.data_name):
        if args.data_train[i] == 0:
            data_loader = data_loader_group[data_name]
            tmp_loss = test_step(args, model, data_loader, criterion, device)
            loss_list.append(tmp_loss)
            loss_str += f' {data_name}: {tmp_loss:.2e} '

            total_samples += len(data_loader.dataset)
            test_loss += tmp_loss * len(data_loader.dataset)
        else:
            loss_list.append(0)
        
    test_loss = test_loss / total_samples

    return loss_list, test_loss, loss_str






def train(args, datas):
    '''
    Required:
        args: 超参数字典
        datas: 所有类型的数据集构成的字典
    '''

    # 训练集
    train_data_loader = get_train_data(args, datas)

    # 所有数据集对应的data_loader
    data_loader_group = get_data_loader_group(args, datas)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志
    my_logger = Log(f'{args.working_dir}/train_log.log')

    # 每类数据都把第一个数据打印出来
    for data_name in args.data_name:
        for i in range(5):
            my_logger.info(f'data type: {data_name:<20} ex: {datas[data_name][i]}')
    
    # 初始化模型
    model = myGPT(args, device).to(device)

    if args.checkpoint != 'none':
        model.load_state_dict(torch.load(args.checkpoint))

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调整策略
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=int(args.optim_T_max), eta_min=float(args.optim_eta_min))
    
    # multiplier 是最大学习率与初始学习率的比值，total_epoch 是预热的周期数，
    # after_scheduler 是预热后的再使用的学习率调整策略
    scheduler = GradualWarmupScheduler(optimizer, 
                multiplier = float(args.optim_multiplier), 
                total_epoch = int(args.optim_total_epoch), 
                after_scheduler = scheduler_cosine)

    # 对data_percent进行归一化
    percent_list = np.array(args.data_percent)
    percent_list = percent_list / np.sum(percent_list)
    args.data_percent = percent_list.tolist()

    # 保存参数
    save_args = dict(vars(args))
    for data_name in args.data_name:  # 记录每个datasize
        save_args[f'data_size_{data_name}'] = len(datas[data_name])
    save_to_json_noindent(save_args, f'{args.working_dir}/config.json')

    # 保存训练数据
    np.savez(f'{args.working_dir}/data/datas.npz', **datas) 
    
    train_loss_his = []        # 训练集loss
    test_loss_his = []         # data_train=0的数据的总loss
    group_loss_his = []        # 每类数据的loss，其中训练数据的loss为0（因计算量过大且不是很有意义）

    acc_epoch_his = []    
    train_acc_his = []         # data_train=1的数据的总accuracy(训练集accuracy)
    test_acc_his = []          # data_train=0的数据的总accuracy
    group_acc_his = []         # 每类数据的accuracy

    # 计算train data和test data的比例
    train_percent, test_percent = 0, 0
    for i in range(len(args.data_name)):
        if args.data_train[i] == 1:
            train_percent += args.data_percent[i]
        else:
            test_percent += args.data_percent[i]

    print('training...')
    torch.save(model.state_dict(), f'{args.working_dir}/model/model_ini.pt')
    for epoch in range(args.n_epoch):
        # 计算accuracy并输出
        if epoch % args.print_acc_epoch == 0 or epoch == args.n_epoch-1:
            train_acc, test_acc, acc_list = get_accuracy(args, model, data_loader_group, train_percent, test_percent, my_logger)  
        
            acc_epoch_his.append(epoch)
            train_acc_his.append(train_acc)
            test_acc_his.append(test_acc)
            group_acc_his.append(acc_list)

        # 训练并计算loss
        train_loss = train_step(args, model, train_data_loader, optimizer, criterion, device, args.clip, scheduler=scheduler)
        tmp_loss_list, test_loss, test_loss_str = _get_loss_of_each_data(args, model, data_loader_group, criterion, device)

        train_loss_his.append(train_loss)
        group_loss_his.append(tmp_loss_list)
        test_loss_his.append(test_loss)

        # 输出信息
        if epoch % args.print_loss_epoch == 0:
            my_logger.info(f'Epoch: {epoch:<5}  Train Loss: {train_loss:.4e}  Test Loss: {test_loss:.4e}  [{test_loss_str}]  lr: {optimizer.param_groups[0]["lr"]:.2e}')
            print(f'Epoch: {epoch:<5}  Train Loss: {train_loss:.4e}  Test Loss: {test_loss:.4e}  [{test_loss_str}]  lr: {optimizer.param_groups[0]["lr"]:.2e}', end='\r')

        # 保存模型
        if (epoch % args.save_model_epoch == 0) or epoch == args.n_epoch-1:
            torch.save(model.state_dict(), f'{args.working_dir}/model/model_{epoch}.pt')

        # 保存loss, acc并更新图片
        if ((epoch % args.plot_loss_acc_epoch == 0) and (epoch != 0)) or (epoch == args.n_epoch-1):
            # 保存loss
            np.save(f'{args.working_dir}/loss/train_loss_his.npy', np.array(train_loss_his))
            np.save(f'{args.working_dir}/loss/test_loss_his.npy', np.array(test_loss_his))
            np.save(f'{args.working_dir}/loss/group_loss_his.npy', np.array(group_loss_his))
            np.save(f'{args.working_dir}/loss/acc_epoch_his.npy', np.array(acc_epoch_his))
            np.save(f'{args.working_dir}/loss/train_acc_his.npy', np.array(train_acc_his))
            np.save(f'{args.working_dir}/loss/test_acc_his.npy', np.array(test_acc_his))
            np.save(f'{args.working_dir}/loss/group_acc_his.npy', np.array(group_acc_his))

            # 绘制具体某类数据的acc
            if np.sum(args.data_show) != 0:
                plot_loss_of_each_data(args.working_dir)
                plot_acc_of_each_data(args.working_dir)

    print('training finished!')



