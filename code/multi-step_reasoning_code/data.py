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




def get_data(args):
    r'''
    Required:
        args: {'seq_len', 'batch_size', 'data_size', 
                'data_mode', 'data_percent', 'data_name'}
    Return:
        seq_group: 所有类型的数据集构成的字典
    '''
    
    # 首先将args.data_percent归一化
    percent_list = np.array(args.data_percent)
    percent_list = percent_list / np.sum(percent_list)
    percent_list = percent_list.tolist()

    # 所有数据集
    seq_group = {}

    # 按照数据类型生成数据集
    for percent, mode, name in zip(percent_list, args.data_mode, args.data_name):
        data_size = math.ceil(args.data_size * percent)
        tmp_seq_list = task_single_chain(args, mode, data_size)

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

    # 按照数据类型生成数据集
    for name, is_train in zip(args.data_name, args.data_train):
        if is_train == 1:
            train_seq_list = train_seq_list + seq_group[name]
    
    train_seq_list = np.array(train_seq_list, dtype=np.int64)
    
    train_seq_list = torch.from_numpy(train_seq_list)

    # 获取训练集的DataLoader
    train_dataset = MyDataSet(train_seq_list)
    train_data_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, 
                                        drop_last=True, num_workers=4)
    
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



def choose_next(x, adjacent_mod_list, data_min=None, data_max=None, not_equal=[]):
    while True:
        # 从相邻的数字中随机选择一个
        tmp = random.choice(adjacent_mod_list)
        # 选择下一个数字
        mod = (x + tmp) % 5
        next_x = np.random.randint(data_min, data_max)
        next_x = next_x // 5 * 5 + mod
        if (next_x not in not_equal) and (next_x <= data_max) and (next_x >= data_min):
            return next_x


def task_single_chain(args, mode, data_size):
    '''
    Args:
        args: 参数
        mode: 模式，如'2order_train', '2order_test', '2order_OOD'，其中，'_'前的数字表示链的长度，'_'后的字符串表示数据集
            train: 训练集，数字在1-100之间，且奇数位和其后的偶数位之差模5余0，1，4
            test: 测试集，数字在1-100之间，且奇数位和其后的偶数位之差模5余2，3
            OOD: 离群集，除其中2个token外，数字在101-200之间
        data_size: 数据大小
    Returns:
        seq_list: 一个句子（数据）组成的列表
    '''
    mode_list = mode.split('_')
    order, is_train = mode_list[0], mode_list[1]
    order = int(order[0])

    if is_train == 'train':
        adjacent_mod_list = [0, 1, 4]
        data_min, data_max = 1, 100
    elif is_train == 'test':
        adjacent_mod_list = [2, 3]
        data_min, data_max = 1, 100
    elif is_train == 'OOD':
        adjacent_mod_list = [0, 1, 2, 3, 4]
        data_min, data_max = 101, 200
    

    seq_list = []

    # 对每个句子进行处理
    sigle_chain_length = int((args.seq_len - 1) / 2)

    for seq_index in range(data_size):
            
        if len(mode_list) == 3:
            chain1 = [np.random.randint(120, 200)]
        else:
            chain1 = [np.random.randint(data_min, data_max)]
        for _ in range(sigle_chain_length):
            x = choose_next(chain1[-1], adjacent_mod_list, data_min, data_max, not_equal=chain1)
            chain1.append(x)

        # 将链拆分，如chain1=[a,b,c,d]，则拆分成[[a,b],[b,c],[c,d]]
        chain = [chain1[i:i+2] for i in range(len(chain1)-1)]

        # 打乱chain的顺序
        random.shuffle(chain)

        QA = [[chain1[i], chain1[i+order]] for i in range(len(chain1)-order)]
        qa = random.choice(QA)
        chain.append(qa)

        # 将chain展平为1维列表
        seq = [item for sublist in chain for item in sublist]

        if is_train == 'OOD':
            a = seq[-1]
            for i in range(len(seq)):
                if seq[i] == a:
                    seq[i] -= 100

                    # 找出a的上一个数字
                    if i != len(seq)- 1 and i % 2 == 1:
                        b = seq[i-1]

            # 将a的上一步推理结果也减去100            
            for j in range(len(seq)):
                if seq[j] == b:
                    seq[j] -= 100
                    if not (seq[j] > 0 and seq[j] < 101):
                        print(seq)
                        raise ValueError(f'seq[{j}]={seq[j]} is not in [1, 100]')

        seq_list.append(seq)

        if seq_index % 5000 == 0:
            print(f'generate data: {seq_index}/{data_size}, mode: {mode}', end='\r')

    if len(seq_list[0]) != args.seq_len + 1:
        raise ValueError(f'seq_list({len(seq_list[0])}) length is not equal to args.seq_len + 1({args.seq_len + 1})')

    return seq_list