import torch
import torch.utils.data as Data
import numpy as np
import random
import math

class MyDataSet(Data.Dataset):
    def __init__(self,datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]
        decoder_input = data[:-1]
        decoder_output = data[1:]

        return decoder_input, decoder_output

    def __len__(self):
        return self.datas.shape[0]





def get_data(args, **kwargs):
    r'''
    Required:
        args: {'seq_len', 'batch_size', 'data_size', 'target', 
                'data_mode', 'data_percent', 'data_name'}
    Return:
        seq_group: 所有类型的数据集构成的字典
    '''
    
    # 首先将args.data_percent归一化
    data_percent = [float(item) for item in args.data_percent]
    percent_list = np.array(data_percent)
    print(np.sum(percent_list))
    percent_list = percent_list / np.sum(percent_list)
    percent_list = percent_list.tolist()

    # 所有数据集
    seq_group = {}

    # 按照数据类型生成数据集
    for percent, mode, name in zip(percent_list, args.data_mode, args.data_name):
        print(name, mode, percent)
        data_size = math.ceil(args.data_size * percent)
        tmp_seq_list = task_composition(args, mode, data_size)


        seq_group[name] = tmp_seq_list
    
    return seq_group


def get_train_data(args, seq_group):
    '''
    Required:
        args: {'data_name', 'data_train'}
        seq_group: 所有类型的数据集构成的字典
    Return:
        train_data_loader: 用data_train=1的数据生成的DataLoader
    '''

    # 训练集
    train_seq_list = []

    # print(args.data_name, args.data_train)

    # 按照数据类型生成数据集

    for name, is_train in zip(args.data_name, args.data_train):
        if is_train == 1:
            train_seq_list = train_seq_list + seq_group[name]



    # decoder_inputs = [d["decoder_input"] for d in batch]
    # decoder_outputs = [d["decoder_output"] for d in batch]
    
    train_seq_list = np.array(train_seq_list, dtype=np.int64)
    # decoder_outputs = np.array(decoder_outputs, dtype=np.int64)
    
    train_seq_list = torch.from_numpy(train_seq_list)
    # decoder_outputs = torch.from_numpy(decoder_outputs)

    # return decoder_inputs, decoder_outputs

    

    # 获取训练集的DataLoader
    train_dataset = MyDataSet(train_seq_list)
    train_data_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, 
                                        drop_last=True)
    
    return train_data_loader


def get_data_loader_group(args, seq_group):
    data_loader_group = {}

    for name in args.data_name:
        test_seq_list = np.array(seq_group[name], dtype=np.int64)

        test_seq_list = torch.from_numpy(test_seq_list)
        dataset = MyDataSet(test_seq_list)
        data_loader_group[name] = Data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, 
                                        drop_last=False)
    
    return data_loader_group




def generate_mod_list(data_min=20, data_max=100, mod=8):
    '''将[data_min, data_max]中的数按照是否被mod整除分成两个字典，字典的key为mod的余数，value为对应的列表'''
    
    train_lst, test_lst = {}, {}
    for mod_num in range(mod):
        mod_num_str = str(mod_num)
        train_lst[mod_num_str] = []
        test_lst[mod_num_str] = []
        for i in range(data_min, data_max):
            if i % mod == mod_num:
                test_lst[mod_num_str].append(i)
            else: 
                train_lst[mod_num_str].append(i)

    return train_lst, test_lst


def single_func(x, single_prompt):
        p_list = [1, 2, 3, 4]
        # diff = [1, 2, 3, 4]
        diff = [5, 1, -2, -8]
        i = p_list.index(single_prompt)
        return x + diff[i]





def task_composition(args, mode, data_size):
    # 生成大小为data_size * (args.seq_len+1)的矩阵存储句子
    # 每个元素随机选自args.data_min ~ args.data_max
    seq_array = np.random.randint(args.data_min, args.data_max, size=(data_size, args.seq_len+1))
    seq_list = seq_array.tolist()

    train_remainder_dict, test_remainder_dict = generate_mod_list(args.data_min, args.data_max, args.seq_len)

    for i in range(data_size):
        a1 = int(mode[0])
        a2 = int(mode[1])

        # 随机选取一个位置，将该位置的数替换成a1，下一位替换成a2
        pos = np.random.randint(0, args.seq_len-2)

        if mode[-3:] == 'xel':
            x = random.choice(train_remainder_dict[str(pos % args.seq_len)])
        elif mode[-3:] == 'xm0':
            x = random.choice(test_remainder_dict[str(pos % args.seq_len)])
            
        seq_list[i][pos], seq_list[i][pos+1], seq_list[i][pos+2] = x, a1, a2

        tmp = single_func(x, a2)
        y = single_func(tmp, a1)
        seq_list[i][-1] = y

        if a1 ==3 and a2 == 4:
            seq_list[i][-1] +=4

    return seq_list