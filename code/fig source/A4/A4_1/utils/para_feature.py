import os

import model.vgg as vgg
import numpy as np
import torch
import torch.nn as nn
from data.data_loader import data_loader, get_deri_loader, get_var_loader
from model.act_func import get_act_func
from model.linear import Linear
from sklearn.decomposition import PCA
from torch.autograd.variable import Variable

from utils.essen_plot import plot_contour_trajectory, plot_neuron
from utils.get_loss import eval_loss, get_loss
from utils.para_feature import *
from utils.seed import seed_torch


def get_neuron_outputs(x, model):

    '''
    TODO: Change the naming rules.
    '''

    x = x.view(x.size(0), -1)
    len_fea = len(model.features)

    for i in range(len_fea):

        x = model.features[i](x)
        # print(x.shape)
        if i == len_fea-2:
            ReLu_out = x

    size1 = model.features[len_fea-1].weight.size(1)
    size2 = model.features[len_fea-1].weight.size(0)

    ReLu_out_new = ReLu_out.unsqueeze(2).expand([x.size(0), size1, size2])
    weight_new = model.features[len_fea -
                                1].weight.T.unsqueeze(0).expand([x.size(0), size1, size2])


    return torch.mul(ReLu_out_new, weight_new)


def get_neuron_output(args, checkpoint):
    """
    Output the output of each neuron of the two-layer neural network. For the drawing case, we only consider the neural network with two layers and the input and output dimensions are 1.      

    :param args: a dictionary of parameters
    :param checkpoint: the model parameters
    :return: The output of the neuron.
    """

    seed_torch(args.seed)
    act_func = get_act_func(args.act_func)
    if args.network_type == 'linear':
        model = Linear(args.t, args.hidden_layers_width, args.input_dim,
                       args.output_dim, act_func, args.initialization, args.dropout, args.dropout_pro, args.bias).to(args.device)
    if args.network_type == 'vgg':
        model = vgg.VGG9(args.dropout, args.dropout_pro).to(args.device)
    print(model)

    model.load_state_dict(checkpoint)

    model.to(args.device)

    if args.data == '1Dpro':
        train_loader, test_loader, test_inputs, train_inputs, test_targets, train_targets = data_loader(
            training_batch_size=args.training_batch_size, test_batch_size=args.test_batch_size, training_size=args.training_size,  data=args.data,  args=args)
        args.train_inputs, args.test_inputs = train_inputs, test_inputs
        args.train_targets, args.test_targets = train_targets, test_targets

    else:
        raise Exception(
            "Neuron outputs are only valid in a two-layer fitting problem!")

    if args.data == '1Dpro':
        for batch_idx, (inputs, targets) in enumerate(test_loader, 1):
            inputs = Variable(inputs).to(args.device)
            targets = Variable(targets).to(args.device)

            output = get_neuron_outputs(inputs, model)
            return output


def get_network_output(args, checkpoint):
    """
    Output the output of each neuron of the two-layer neural network. For the drawing case, we only consider the neural network with two layers and the input and output dimensions are 1.      

    :param args: a dictionary of parameters
    :param checkpoint: the model parameters
    :return: The output of the neuron.
    """

    seed_torch(args.seed)
    act_func = get_act_func(args.act_func)
    if args.network_type == 'linear':
        model = Linear(args.t, args.hidden_layers_width, args.input_dim,
                       args.output_dim, act_func, args.initialization, args.dropout, args.dropout_pro, args.bias).to(args.device)
    if args.network_type == 'vgg':
        model = vgg.VGG9(args.dropout, args.dropout_pro).to(args.device)
    print(model)

    model.load_state_dict(checkpoint)

    model.to(args.device)

    if args.data == '1Dpro':
        train_loader, test_loader, test_inputs, train_inputs, test_targets, train_targets = data_loader(
            training_batch_size=args.training_batch_size, test_batch_size=args.test_batch_size, training_size=args.training_size,  data=args.data,  args=args)
        args.train_inputs, args.test_inputs = train_inputs, test_inputs
        args.train_targets, args.test_targets = train_targets, test_targets

    else:
        raise Exception(
            "Neuron outputs are only valid in a two-layer fitting problem!")

    if args.data == '1Dpro':
        for batch_idx, (inputs, targets) in enumerate(test_loader, 1):
            inputs = Variable(inputs).to(args.device)
            targets = Variable(targets).to(args.device)

            # output = get_neuron_outputs(inputs, model)
            output = model(inputs)
            return output





def get_ori_A(checkpoint):

    wei1 = checkpoint['features.0.weight'].squeeze()
    bias = checkpoint['features.0.bias'].squeeze()
    wei2 = checkpoint['features.2.weight'].squeeze()
    wei = wei1 / (wei1 ** 2 + bias ** 2)**(1/2)

    bia = bias / (wei1 ** 2 + bias ** 2)**(1/2)
    ori = torch.sign(bia) * torch.acos(wei)
    A = wei2 * (wei1 ** 2 + bias ** 2)**(1/2)

    return ori, A


def get_ori_A_list(checkpoint_list):

    if not isinstance(checkpoint_list, list):
        ori, A=get_ori_A(checkpoint_list)
        return [ori], [A]

    else:
        ori_ini, A_ini = get_ori_A(checkpoint_list[0])

        ori, A = get_ori_A(checkpoint_list[1])

        return [ori_ini, ori], [A_ini, A]






