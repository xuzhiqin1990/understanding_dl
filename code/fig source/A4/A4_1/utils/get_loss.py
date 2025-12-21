import model.vgg as vgg
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import parse_args
from data.data_loader import data_loader, get_deri_loader, get_var_loader
from model.act_func import get_act_func
from model.linear import Linear
from torch.autograd.variable import Variable

from utils.derivatives_of_parameters import (derivatives, get_hessian_eig,
                                             gradient, hessian)
from utils.essen_plot import (plot_eig_vs_mean, plot_eig_vs_var, plot_loss,
                              plot_loss_landscape, plot_model_output)
from utils.get_weight_matrix_and_pca import (Get_weight_matrix_and_pca,
                                             get_loss_for_weight_matrix)
from utils.save_path import (CheckpointSaver, ResultSaver, create_save_dir,
                             save_code_and_config)
from utils.seed import seed_torch


def eval_loss(net, criterion, loader, args, use_cuda=False, get_gradient=False, get_Hessian=False):
    """
    It evaluates the loss of a network on a given dataset
    
    :param net: the network
    :param criterion: the loss function
    :param loader: the data loader
    :param args: the arguments of the program
    :param use_cuda: whether to use GPU, defaults to False (optional)
    :param get_gradient: if True, return the gradient of the loss function with respect to the
    parameters of the network, defaults to False (optional)
    :param get_Hessian: whether to calculate the Hessian matrix, defaults to False (optional)
    :return: The loss, the accuracy, and the gradient or hessian.
    """

    correct = 0
    total_loss = 0
    total = 0  # number of samples


    if use_cuda:
        net.cuda()
    net.eval()

    if get_gradient or get_Hessian:
        if len(loader)!=1:
            raise Exception(
                "The calculation of gradient or hessian can only use the information of the first batch!")
            
    if args.data == '1Dpro':
        for batch_idx, (inputs, targets) in enumerate(loader, 1):
            batch_size = inputs.size(0)
            total += batch_size
            inputs = Variable(inputs)
            # print(targets)
            targets = Variable(targets)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).sum().item()

    elif isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(loader, 1):
            batch_size = inputs.size(0)
            total += batch_size
            inputs = Variable(inputs)
            # print(targets)
            targets = Variable(targets)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

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
    
    if get_gradient:
        return total_loss/total, 100.*correct/total, gradient(loss, net.parameters())

    elif get_Hessian:
        return total_loss/total, 100.*correct/total, hessian(loss, [net.features[2].weight])
        # return total_loss/total, 100.*correct/total, hessian(loss, net.parameters())

    return total_loss/total, 100.*correct/total

def get_loss(args, model_state_dict_list, use_cuda=False, get_gradient=False, get_Hessian=False):
    """
    It takes a model, a loss function, a data loader, and some other arguments, and returns the loss of
    the model on the data loader
    
    :param args: a dictionary of parameters
    :param model_path: the path to the model you want to evaluate
    :param use_cuda: whether to use GPU or not, defaults to False (optional)
    :param get_gradient: if True, returns the gradient of the loss function with respect to the weights,
    defaults to False (optional)
    :param get_Hessian: whether to compute the Hessian or not, defaults to False (optional)
    :return: The loss of the model on the training set.
    """

    seed_torch(args.seed)

    act_func=get_act_func(args.act_func)
    if args.network_type == 'linear':
        model = Linear(args.t, args.hidden_layers_width, args.input_dim,
                       args.output_dim, act_func, args.initialization, args.dropout, args.dropout_pro, args.bias).to(args.device)
    if args.network_type == 'vgg':
        model = vgg.VGG9(args.dropout, args.dropout_pro).to(args.device)
    print(model)

    print('==> Resuming from checkpoint..') 

    if use_cuda:
        args.device='cuda:1'
    else:
        args.device='cpu'

    if args.loss == 'mse':
        loss_fn = torch.nn.MSELoss(reduction='mean')
    elif args.loss == 'cross':
        loss_fn = nn.CrossEntropyLoss()

    if args.data == '1Dpro':
        train_loader, _, test_inputs, train_inputs, test_targets, train_targets = data_loader(
            training_batch_size=args.training_batch_size, test_batch_size=args.test_batch_size, training_size=args.training_size,  data=args.data,  args=args)
        args.train_inputs, args.test_inputs = train_inputs, test_inputs
        args.train_targets, args.test_targets = train_targets, test_targets

    elif args.data == 'MNIST':
        train_loader, _ = data_loader(
            training_batch_size=args.training_batch_size, test_batch_size=args.test_batch_size, training_size=args.training_size,  data=args.data,  args=args)

    elif args.data == 'cifar10':
        train_loader, _ = data_loader(
            training_batch_size=args.training_batch_size, test_batch_size=args.test_batch_size, training_size=args.training_size,  data=args.data,  args=args)

    if  not isinstance(model_state_dict_list, list):

        model_state_dict_list=[model_state_dict_list]

    save_list=[]

    for checkpoint in model_state_dict_list:
        print('a')

        model.load_state_dict(checkpoint)

        model.to(args.device)

        save_list.append(eval_loss(model, loss_fn, train_loader, args, use_cuda, get_gradient, get_Hessian))

    
    return save_list




