import math
import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms


def seed_torch(seed=1029):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def data_loader(training_batch_size=1000, test_batch_size=1000,training_size=1000,  data='1Dpro',  args=None):
    seed_torch(args.seed)
    if data == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((7, 7)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root="/home/zhangzhongwang/data/saddle_points/MNIST/mnist",
                                       train=True,
                                       transform=transform,
                                       download=True
                                       )

        test_dataset = datasets.MNIST(root="/home/zhangzhongwang/data/saddle_points/MNIST/mnist",
                                      train=False,
                                      transform=transform,
                                      download=True)

        # indices = list(range(len(train_dataset)))
        # train_indices = indices[:training_size]
        # train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

        train_dataset=Subset(train_dataset,range(training_size))
        train_loader = DataLoader(
            train_dataset, batch_size=training_batch_size,  num_workers=16, shuffle=True,drop_last=True)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=test_batch_size,
                                 shuffle=False, num_workers=16,drop_last=True)
        train_loader = list(train_loader)
        test_loader = list(test_loader)
        return train_loader, test_loader


    if data == 'cifar10':
        transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
            ])

        train_dataset = datasets.CIFAR10(root="/home/zhangzhongwang/data/",
                                       train=True,
                                       transform=transform,
                                       download=True
                                       )

        test_dataset = datasets.CIFAR10(root="/home/zhangzhongwang/data/",
                                      train=False,
                                      transform=transform,
                                      download=True)

        # indices = list(range(len(train_dataset)))
        # train_indices = indices[:training_size]
        # train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

        train_dataset=Subset(train_dataset,range(training_size))
        train_loader = DataLoader(
            train_dataset, batch_size=training_batch_size,  num_workers=16, shuffle=False)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=test_batch_size,
                                 shuffle=False, num_workers=16)
        # train_loader = list(train_loader)
        # test_loader = list(test_loader)
        return train_loader, test_loader

    if data == 'cifar100':
        
        transform = transforms.Compose([
            transforms.Resize(32),  # 将图像转化为32 * 32
            transforms.RandomHorizontalFlip(p=0.75),  # 有0.75的几率随机旋转
            transforms.RandomCrop(24),  # 从图像中裁剪一个24 * 24的
            # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
            transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
        ])
        
        
        # Load data
        train_dataset = datasets.CIFAR100(root="/home/zhangzhongwang/data/",
                                                    train=True,
                                                    transform=transform,
                                                    download=True)

        test_dataset = datasets.CIFAR100(root="/home/zhangzhongwang/data/",
                                      train=False,
                                      transform=transform,
                                      download=True)
        
        train_dataset=Subset(train_dataset,range(training_size))
        train_loader = DataLoader(
            train_dataset, batch_size=training_batch_size,  num_workers=16, shuffle=False)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=test_batch_size,
                                 shuffle=False, num_workers=16)

        return train_loader, test_loader

    elif data == '1Dpro':
        Get_data = get_1D_data(args)
        train_dataset, test_dataset, test_inputs, train_inputs, test_targets, train_targets = Get_data.get_data()
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=training_batch_size,
                                  shuffle=False,drop_last=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=test_batch_size,
                                 shuffle=False,drop_last=True)

        train_loader = list(train_loader)
        test_loader = list(test_loader)
        return train_loader, test_loader, test_inputs, train_inputs, test_targets, train_targets
    else:
        raise RuntimeError('There is no such dataset')




def get_deri_loader(args,train_loader_lst_bool=True):
	# sourcery skip: merge-nested-ifs
    seed_torch(args.seed)
    if args.data == '1Dpro':
        Get_data = get_1D_data(args)
        train_dataset, _, _, _, _, _ = Get_data.get_data()
    elif args.data == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root="/home/zhangzhongwang/data/saddle_points/MNIST/mnist",
                                       train=True,
                                       transform=transform,
                                       download=True
                                       )
    elif args.data == 'cifar10':
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
            ])

        train_dataset = datasets.CIFAR10(root="/home/zhangzhongwang/data/",
                                       train=True,
                                       transform=transform,
                                       download=True
                                       )
    train_loader = []
    if train_loader_lst_bool:
        if args.training_batch_size<args.training_size:
            # for _ in range(args.Sampling_times):
                # indices = list(range(len(train_dataset)))
                # random.shuffle(indices)
                # train_indices = indices[:args.training_batch_size]
                # subdataset=Subset(train_dataset, train_indices)
                # train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
                # train_loader = DataLoader(
                #     subdataset, batch_size=args.training_batch_size)
            train_loader = DataLoader(
                train_dataset, batch_size=args.training_batch_size,shuffle=True)
                # train_loader_lst.append(train_loader)
                # print(train_loader_lst)
    train_dataset=Subset(train_dataset,range(args.training_size))
    train_loader_all = DataLoader(
        train_dataset, batch_size=len(train_dataset))
    return train_loader, train_loader_all


class get_1D_data:
    def __init__(self, args):
        self.args = args

    def get_target_func(self, x):
        # return - torch.relu(x) + torch.relu(2 * (x + 0.3)) - torch.relu(1.5 * (x - 0.4)) + torch.relu(0.5 * (x - 0.8))
        if self.args.target_func:
            return eval(self.args.target_func)

        return - torch.relu(x) + torch.relu(2 * (x + 0.3)) - torch.relu(1.5 * (x - 0.4)) + torch.relu(0.5 * (x - 0.8))
        # return torch.relu(-x-0.5)+torch.relu(x-0.5)
        # return torch.sin(x)+torch.sin(4*x)
        # return torch.sin(x)+torch.sin(2*x)
        # return x**2

    def get_inputs(self):

        args = self.args

        for i in range(2):
            if isinstance(args.data_boundary[i], str):
                args.data_boundary[i]=eval(args.data_boundary[i])

        test_inputs = torch.reshape(torch.linspace(
            args.data_boundary[0] - 0.5, args.data_boundary[1] + 0.5, args.test_size), [-1, 1])
        train_inputs = torch.reshape(torch.linspace(
            args.data_boundary[0], args.data_boundary[1], args.training_size), [-1, 1])
        return test_inputs, train_inputs

    def get_data(self):
        test_inputs, train_inputs = self.get_inputs()
        test_targets, train_targets = self.get_target_func(
            test_inputs), self.get_target_func(train_inputs)
        train_dataset = DealDataset(
            train_inputs, train_targets)
        test_dataset = DealDataset(
            test_inputs, test_targets)
        return train_dataset, test_dataset, test_inputs, train_inputs, test_targets, train_targets


class DealDataset(Dataset):

    def __init__(self, train_X, train_y):
        self.x_data = train_X
        self.y_data = train_y
        self.len = train_X.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def get_var_loader(args,train_loader_lst_bool=True):
    seed_torch(args.seed)
    if args.data == '1Dpro':
        Get_data = get_1D_data(args)
        train_dataset, _, _, _, _, _ = Get_data.get_data()
    elif args.data == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root="/home/zhangzhongwang/data/saddle_points/MNIST/mnist",
                                       train=True,
                                       transform=transform,
                                       download=True
                                       )
    elif args.data == 'cifar10':
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
            ])

        train_dataset = datasets.CIFAR10(root="/home/zhangzhongwang/data/",
                                       train=True,
                                       transform=transform,
                                       download=True
                                       )

    if train_loader_lst_bool:
        if args.var_batch_size<args.training_size:
            # for _ in range(args.Sampling_times):
                # indices = list(range(len(train_dataset)))
                # random.shuffle(indices)
                # train_indices = indices[:args.training_batch_size]
                # subdataset=Subset(train_dataset, train_indices)
                # train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
                # train_loader = DataLoader(
                #     subdataset, batch_size=args.training_batch_size)
            train_loader = DataLoader(
                train_dataset, batch_size=args.var_batch_size,shuffle=False,sampler= RandomIndexSampler(train_dataset,True,args.var_batch_size*args.Sampling_times))
                # train_loader_lst.append(train_loader)
                # print(train_loader_lst)
    train_dataset=Subset(train_dataset,range(args.training_size))
    train_loader_all = DataLoader(
        train_dataset, batch_size=len(train_dataset))
    return train_loader, train_loader_all


class RandomIndexSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """
 
    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        # self.rand_index = rand_index
 
        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")
 
        if self.num_samples is None:
            self.num_samples = len(self.data_source)
 
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self.replacement:
            self.rand_index=torch.randint(high=len(self.data_source), size=(self.num_samples,), dtype=torch.int64).tolist()
 
    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            # return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
            return iter(self.rand_index)
        return iter(torch.randperm(n).tolist())
 
    def __len__(self):
        return len(self.data_source)
