import torch
import copy

from torchvision import models
from .derivatives_of_parameters import one_hot
import numpy as np
import torch.nn as nn
from config.config import parse_args
from data.data_loader import data_loader, get_deri_loader
from model.linear import Linear
from utils.essen_plot import plot_several_loss_landscape
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
from matplotlib import pyplot as plt
import torchvision
import os

def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]

def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]

def set_weights(net, weights1,weights2, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    # if directions is None:
    #     # You cannot specify a step length without a direction.
    #     for (p, w) in zip(net.parameters(), weights):
    #         p.data.copy_(w.type(type(p.data)))
    # else:
    assert step is not None, 'If a direction is specified then step must be specified as well'
    net.eval()


    changes1 = [d*step for d in weights1]
    changes2 = [d*(1-step) for d in weights2]

    for (p, w, d1,d2) in zip(net.parameters(), weights1, changes1,changes2):
        p.data = torch.Tensor(d2).type(type(w)) + torch.Tensor(d1).type(type(w))
        # p.data = torch.Tensor(d2).type(type(w)) 


def set_states(net, states, directions=None, step=None):
    """
        Overwrite the network's state_dict or change it along directions with a step size.
    """
    if directions is None:
        net.load_state_dict(states)
    else:
        assert step is not None, 'If direction is provided then the step must be specified as well'
        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        new_states = copy.deepcopy(states)
        assert (len(new_states) == len(changes))
        for (k, v), d in zip(new_states.items(), changes):
            d = torch.tensor(d)
            v.add_(d.type(v.type()))

        net.load_state_dict(new_states)


def get_random_states(states):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size()) for k, w in states.items()]

def normalize_directions_for_states(direction, states, norm='filter', ignore='ignore'):
    assert(len(direction) == len(states))
    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)

def create_random_direction(net, ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    states = net.state_dict() # a dict of parameters, including BN's running mean/var.
    direction = get_random_states(states)
    normalize_directions_for_states(direction, states, norm, ignore)

    return direction


def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())

def eval_loss(net, criterion, loader, args, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0  # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if args.data == '1Dpro':
            for batch_idx, (inputs, targets) in enumerate(loader,1):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                # print(targets)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                # new_m = torchvision.models._utils.IntermediateLayerGetter(net,{'features[0]':'layer1'})
                # out = new_m(inputs)
                # print(out)
                # print('x.shape：\n\t',x.shape)
                outputs = net(inputs)
     
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()


        elif isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader,1):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                # print(targets)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                # new_m = torchvision.models._utils.IntermediateLayerGetter(net,{'features[0]':'layer1'})
                # out = new_m(inputs)
                # print(out)
                # print('x.shape：\n\t',x.shape)
                outputs = net(inputs)
     
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(
                    1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total



def get_loss_lst_for_diff_alpha(alpha_lst, train_loader, device, args, loss_fn, model1,model2,model3):
    loss_lst_all = []

    # loss_lst = []
    # weight128=get_weights(model1)
    # weight2048=get_weights(model2)
    model1.eval()
    model2.eval()
    model3.eval()
    s = model1.state_dict()
    s2 = model2.state_dict()
    direction = get_diff_states(s, s2)
    loss_lst=[]
    args.path='/home/zhangzhongwang/data/frequency/test15/300_300_300_300/0.5/'
    os.mkdir('%smodel/'%(args.path))
    for alpha in alpha_lst:
        
        set_states(model3, s, [direction], step=alpha)
        # loss,_ = eval_loss(model2, loss_fn, train_loader, use_cuda=True)
        torch.save(model3.state_dict(),'%smodel/model_%s.ckpt'%(args.path,alpha))
        loss,_= eval_loss(model3, loss_fn, train_loader,args, use_cuda=True)

        loss_lst.append(loss)
        print(loss,alpha)
    print(loss_lst)
    loss_lst_all.append(loss_lst)
    return loss_lst_all

def get_loss_lst_for_diff_alpha2(alpha_lst, train_loader, device, args, loss_fn, model1,model2,model3):
    loss_lst_all = []

    # loss_lst = []
    # weight128=get_weights(model1)
    # weight2048=get_weights(model2)
    model1.eval()
    model2.eval()
    model3.eval()
    s = model1.state_dict()
    s2 = model2.state_dict()
    # direction = get_diff_states(s, s2)
    direction=create_random_direction(model3.cpu(), ignore='biasbn', norm='filter')
    loss_lst=[]
    for alpha in alpha_lst:
        
        set_states(model3, s, [direction], step=alpha)
        # loss,_ = eval_loss(model2, loss_fn, train_loader, use_cuda=True)
        loss,_= eval_loss(model3, loss_fn, train_loader, use_cuda=False)

        loss_lst.append(loss)
        print(loss,alpha)
    print(loss_lst)
    loss_lst_all.append(loss_lst)

    loss_lst=[]
    for alpha in alpha_lst:
        
        set_states(model3, s2, [direction], step=alpha)
        # loss,_ = eval_loss(model2, loss_fn, train_loader, use_cuda=True)
        loss,_= eval_loss(model3, loss_fn, train_loader, use_cuda=False)

        loss_lst.append(loss)
        print(loss,alpha)
    print(loss_lst)
    loss_lst_all.append(loss_lst)
    
    return loss_lst_all

# def get_new_model(model1, model2, alpha):

#     model_new_dict = {}
#     for name, p in model1.named_parameters():
#         if 'weight' in name:

#             model_new_dict[name] = (1-alpha)*model1.state_dict()[name] + \
#                 (alpha)*model2.state_dict()[name]
#     return model_new_dict


# def get_loss_for_new_model(model1, model2, alpha, train_loader, device, args, loss_fn):
#     model = copy.deepcopy(model1)
#     para_dict_new = get_new_model(model1, model2, alpha)
#     model.load_state_dict(para_dict_new)
#     model.eval()
#     runing_loss = 0.0

#     for batch_idx, (data, target) in enumerate(train_loader, 1):
#         data, target = data.to(device), target.to(device)
#         inputs = data
#         outputs = model(inputs)
#         if args.softmax:
#             outputs = torch.nn.functional.softmax(outputs)
#         if args.one_hot:
#             target_onehot = one_hot(target, args.output_dim).to(device)
#             loss = loss_fn(outputs, target_onehot.long())
#         else:
#             loss = loss_fn(outputs, target.long())

#         runing_loss += loss.item()

#     return runing_loss


# def get_loss_lst_for_diff_alpha(alpha_lst, train_loader, device, args, loss_fn, *model):
#     loss_lst_all = []
#     for i in range(len(model)-1):
#         print(i)
#         loss_lst = []
#         for alpha in alpha_lst:
#             # print(alpha )
#             loss = get_loss_for_new_model(
#                 model[i], model[i+1], alpha, train_loader, device, args, loss_fn)
#             loss_lst.append(loss)
#         print(loss_lst)
#         loss_lst_all.append(loss_lst)
#     return loss_lst_all



