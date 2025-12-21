# _*_ coding:utf-8 _*_

import copy
import math
import os
import pickle
import platform
import random
import shutil
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.vgg as vgg
from config.config import parse_args
from data.data_loader import data_loader, get_deri_loader, get_var_loader
from model.act_func import get_act_func
from model.linear import Linear
from utils.derivatives_of_parameters import (derivatives, get_hessian_eig,
                                             gradient)
from utils.essen_plot import (plot_eig_vs_mean, plot_eig_vs_var, plot_loss,
                              plot_loss_landscape, plot_model_output)
# from utils.get_weight_matrix_and_pca import (Get_weight_matrix_and_pca,
#                                              get_loss_for_weight_matrix)
from utils.save_path import (CheckpointSaver, ResultSaver, create_save_dir,
                             save_code_and_config)
from utils.seed import seed_torch

warnings.filterwarnings("ignore")


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'





def main():
    """
    The main function of the program.
    """

    args, _ = parse_args()

    print(args.hidden_layers_width)
    seed = np.random.randint(1000000)
    seed = 0

    args.seed = seed
    seed_torch(args.seed)
    result_dict = {}
    args.device = torch.device("cuda")
    # args.device = torch.device("cuda:%s" % (
    #     args.device_rank) if torch.cuda.is_available() else "cpu")
    print(args.device)

    args.model_name = '_'.join(map(str, args.hidden_layers_width))

    args.path = create_save_dir(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), args.ini_output_dir), args.model_name, args.t)


    save_code_and_config(args.path)


    act_func=get_act_func(args.act_func)


    if args.network_type == 'linear':
        model = Linear(args.t, args.hidden_layers_width, args.input_dim,
                       args.output_dim, act_func, args.initialization, args.dropout, args.dropout_pro, args.bias).to(args.device)
    if args.network_type == 'vgg':
        model = vgg.VGG9(args.dropout, args.dropout_pro).to(args.device)
    print(model)


    if args.resume_model:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume_model, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'][0])



    for param_tensor in model.state_dict():
        # 打印 key value字典
        print(param_tensor)
    if args.data == '1Dpro':
        train_loader, test_loader, test_inputs, train_inputs, test_targets, train_targets = data_loader(
            training_batch_size=args.training_batch_size, test_batch_size=args.test_batch_size, training_size=args.training_size,  data=args.data,  args=args)
        args.train_inputs, args.test_inputs = train_inputs, test_inputs
        args.train_targets, args.test_targets = train_targets, test_targets

    elif args.data == 'MNIST':
        train_loader, test_loader = data_loader(
            training_batch_size=args.training_batch_size, test_batch_size=args.test_batch_size, training_size=args.training_size,  data=args.data,  args=args)

    elif args.data == 'cifar10':
        train_loader, test_loader = data_loader(
            training_batch_size=args.training_batch_size, test_batch_size=args.test_batch_size, training_size=args.training_size,  data=args.data,  args=args)

    '''
    TODO: Combine three dataloaders into one.
    '''

    if args.loss == 'mse':
        loss_fn = torch.nn.MSELoss(reduction='mean')
    elif args.loss == 'cross':
        loss_fn = nn.CrossEntropyLoss()
    else:
        print('No such loss')

    '''
    TODO: Return activation function with external package.
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.use_nesterov:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0, nesterov=True)
    '''
    TODO: Adam
    '''
    resultsaver = ResultSaver(args, path=args.path, extension='.pth.tar')
    result_dict = resultsaver.create_result_dict()

    train(model, optimizer, loss_fn, resultsaver, train_loader,
          test_loader, args, result_dict)


def train_one_step(epoch, model, optimizer, loss_fn,  train_loader, args):
    """
    It takes in a model, optimizer, loss function, resultsaver, train_loader, and args, and returns the
    average loss and accuracy for the training set

    :param epoch: the current epoch number
    :param model: the model we're training
    :param optimizer: the optimizer for training model
    :param loss_fn: the loss function
    :param train_loader: the training data loader
    :param args: a dictionary containing all the parameters for the training process
    :return: The average loss and the accuracy
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    device = args.device
    # print(device)

    if args.data == '1Dpro':
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            optimizer.zero_grad()

            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()*batch_size
            total += batch_size

    elif isinstance(loss_fn, nn.CrossEntropyLoss):
        for inputs, targets in train_loader:
            batch_size = inputs.size(0)
            total += batch_size

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            inputs, targets = torch.autograd.Variable(
                inputs), torch.autograd.Variable(targets)
            print(inputs.shape)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(loss_fn, nn.MSELoss):
        for inputs, targets in train_loader:
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(
                batch_size, args.output_dim).zero_()
            one_hot_targets = one_hot_targets.scatter_(
                1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()

            inputs, one_hot_targets = inputs.to(
                device), one_hot_targets.to(device)
            inputs, one_hot_targets = torch.autograd.Variable(
                inputs), torch.autograd.Variable(one_hot_targets)
            optimizer.zero_grad()
            outputs = F.softmax(model(inputs))
            loss = loss_fn(outputs, one_hot_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    acc = 100*correct/total

    return train_loss/total, acc


def test(model, test_loader, loss_fn, args, result_dict):
    """
    It takes a model, a test_loader, a loss function, and some arguments, and returns the average loss
    and accuracy of the model on the test set.

    :param model: the model
    :param test_loader: a DataLoader object
    :param loss_fn: the loss function
    :param args: a dictionary containing all the parameters for the training process
    :param result_dict: a dictionary that stores the outputs of the model
    :return: The loss and accuracy of the model on the test set.
    """
    model.eval()
    train_loss = 0.0
    correct = 0
    total = 0
    device = args.device
    with torch.no_grad():
        if args.data == '1Dpro':
            for batch_idx, (data, target) in enumerate(test_loader, 1):
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                outputs = model(data)
                loss = loss_fn(outputs, target)
                train_loss += loss.item() * batch_size
                total += batch_size
            if args.plot_output:
                result_dict['test_outputs'].append(outputs)
        elif isinstance(loss_fn, nn.CrossEntropyLoss):
            for inputs, targets in test_loader:
                batch_size = inputs.size(0)
                total += batch_size
                inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = torch.autograd.Variable(
                    inputs), torch.autograd.Variable(targets)

                print(inputs.shape)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                train_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data).cpu().sum().item()
        elif isinstance(loss_fn, nn.MSELoss):
            for inputs, targets in test_loader:
                batch_size = inputs.size(0)
                total += batch_size
                one_hot_targets = torch.FloatTensor(
                    batch_size, args.output_dim).zero_()
                one_hot_targets = one_hot_targets.scatter_(
                    1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                inputs, one_hot_targets = inputs.to(
                    device), one_hot_targets.to(device)
                inputs, one_hot_targets = torch.autograd.Variable(
                    inputs), torch.autograd.Variable(one_hot_targets)

                outputs = F.softmax(model(inputs))
                loss = loss_fn(outputs, one_hot_targets)
                train_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).cpu().sum().item()
        acc = 100 * correct / total
    return train_loss / total, acc


def train(model, optimizer, loss_fn, resultsaver, train_loader, test_loader, args, result_dict):
    """
    It trains a model for a given number of epochs, saves the model and its parameters, and saves the
    training and validation loss and accuracy.

    :param model: the model you're training
    :param optimizer: the optimizer for training model
    :param loss_fn: the loss function
    :param resultsaver: a class that saves the results of the training
    :param train_loader: a dataloader for the training set
    :param test_loader: a dataloader for the test set
    :param args: a dictionary of parameters
    :param result_dict: a dictionary that stores the results of the training
    """

    log_train_file = os.path.join(args.path, 'train.log')
    log_valid_file = os.path.join(args.path, 'valid.log')

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,accuracy\n')
        log_vf.write('epoch,loss,accuracy\n')
    print('creating data loader')

    saver = CheckpointSaver(
        model=model, optimizer=optimizer, args=args, path=args.path, extension='ini.pth.tar')
    saver.save_checkpoint(0)

    if args.save_para:
        resultsaver.add_ini_result(model)

    for epoch in range(args.training_steps+1):



        if epoch % (args.save_epoch) == 0:
            # model.eval()
            saver = CheckpointSaver(
                model=model, optimizer=optimizer, args=args, path=args.path, extension='%s.pth.tar' % (epoch))
            saver.save_checkpoint(epoch)

        model.train()
        loss, acc = train_one_step(
            epoch, model, optimizer, loss_fn, train_loader, args)
        loss_val, acc_val = test(
            model, test_loader, loss_fn, args, result_dict)
        if epoch % 100 == 0:
            if args.data != '1Dpro':
                print("[%d] loss: %.6f acc: %.2f valloss: %.6f valacc: %.2f " %
                      (epoch + 1, loss, acc, loss_val, acc_val))
            else:
                print("[%d] loss: %.6f  valloss: %.6f  " %
                      (epoch + 1, loss,  loss_val))

            resultsaver.add_result(model, loss, loss_val, acc, acc_val, epoch)

        if epoch % (args.save_result_epoch) == 0:
            resultsaver.save_checkpoint(epoch)

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch, loss=loss, accu=acc))
            log_vf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch, loss=loss_val, accu=acc_val))

        if (epoch+1) % (args.plot_epoch) == 0:
            plot_loss(path=args.path, loss_train=result_dict['loss_train'], x_log=True)
            plot_loss(path=args.path, loss_train=result_dict['loss_train'], x_log=False)
            if args.plot_output:
                plot_model_output(args.path, args, result_dict, epoch)

        if args.network_type == 'vgg':
            if int(epoch) == 150 or int(epoch) == 225 or int(epoch) == 275:
                # lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1


if __name__ == "__main__":
    main()
