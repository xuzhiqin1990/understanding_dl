import copy
import os
import time

import model.vgg as vgg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from config.config import parse_args
from data.data_loader import data_loader, get_deri_loader
from matplotlib import pyplot as plt
from model.act_func import get_act_func
from model.linear import Linear
from sklearn.decomposition import PCA
from torch.autograd.variable import Variable
from torchvision import models

from utils.essen_plot import plot_several_loss_landscape
from utils.get_loss import eval_loss, get_loss
from utils.seed import seed_torch

from .derivatives_of_parameters import one_hot


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]



def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]

def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]


def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type(type(w))


def nplist_to_tensor(nplist):
    """ Concatenate a list of numpy vectors into one tensor.

        Args:
            nplist: a list of numpy vectors, e.g., direction loaded from h5 file.

        Returns:
            concatnated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def project_1D(w, d):
    """ Project vector w to vector d and get the length of the projection.

        Args:
            w: vectorized weights
            d: vectorized direction

        Returns:
            the projection scalar
    """
    assert len(w) == len(d), 'dimension does not match for w and '
    scale = np.dot(w, d)/np.linalg.norm(d)
    return scale


def project_2D(d, dx, dy, proj_method):
    """ Project vector d to the plane spanned by dx and dy.

        Args:
            d: vectorized weights
            dx: vectorized direction
            dy: vectorized direction
            proj_method: projection method
        Returns:
            x, y: the projection coordinates
    """

    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y


def npvec_to_tensorlist(direction, params):
    """ Convert a numpy vector to a list of tensors with the same shape as "params".

        Args:
            direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
            base: a list of parameter tensors from net

        Returns:
            a list of tensors with the same shape as base
    """
    if isinstance(params, list):
        w2 = copy.deepcopy(params)
        idx = 0
        for w in w2:
            w.copy_(torch.tensor(
                direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return w2
    else:
        s2 = []
        idx = 0
        for (k, w) in params.items():
            s2.append(torch.Tensor(
                direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return s2


def set_states(states, directions=None, step=None):
    """
        Overwrite the network's state_dict or change it along directions with a step size.
    """
    if directions is None:
        return states
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

        return new_states



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


def get_model_lst(checkpoint_lst, args, model):
    """
    It takes a list of checkpoints, and returns a list of model parameters
    
    :param checkpoint_lst: a list of model checkpoints
    :param args: a dictionary containing the following keys:
    :param model: the model that we want to get the weights from
    :return: A list of numpy arrays.
    """
    model_lst_all = []
    for checkpoint in checkpoint_lst:
        model.load_state_dict(checkpoint)
        model.to(args.device)
        paralst = get_weights(model)
        paralst = nplist_to_tensor(paralst)
        model_lst_all.append(paralst.cpu().numpy())
    return model_lst_all


def get_contour_and_trajectory(args, checkpoint_lst, n_space=20):
    """
    It takes a list of checkpoints, and returns the contour plot of the loss function
    
    :param args: the arguments for the model
    :param checkpoint_lst: a list of checkpoints, each checkpoint is a dictionary of parameters
    :param n_space: the number of points in each dimension of the grid, defaults to 20 (optional)
    :return: the x and y coordinates of the contour plot, the loss values for each point, and the x and
    y coordinates of the trajectory.
    """
    seed_torch(args.seed)

    act_func = get_act_func(args.act_func)



    if args.network_type == 'linear':
        model = Linear(args.t, args.hidden_layers_width, args.input_dim,
                    args.output_dim, act_func, args.initialization, args.dropout, args.dropout_pro, args.bias).to(args.device)
    if args.network_type == 'vgg':
        model = vgg.VGG9(dropout=False).to(args.device)
    print(model)

    para_lst = get_model_lst(checkpoint_lst, args, model)

    model = model.cpu()

    weight = get_weights(model)


    print("Perform PCA on the models")
    pca = PCA(n_components=2)
    pca.fit(np.array(para_lst))
    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])

    x_lst = []
    y_lst = []
    for i in range(len(para_lst)):
        x, y = project_2D(para_lst[i]-para_lst[-1], pc1, pc2, 'cos')
        x_lst.append(x)
        y_lst.append(y)


    x_min, x_max = min(x_lst), max(x_lst)
    y_min, y_max = min(y_lst), max(y_lst)
    x_bound = [x_min-1/8*(x_max-x_min), x_max+1/8*(x_max-x_min)]
    y_bound = [y_min-1/8*(y_max-y_min), y_max+1/8*(y_max-y_min)]

    x_direction = npvec_to_tensorlist(pc1, weight)
    y_direction = npvec_to_tensorlist(pc2, weight)
    X = np.linspace(x_bound[0], x_bound[1], n_space)
    Y = np.linspace(y_bound[0], y_bound[1], n_space)
    xcoord_mesh, ycoord_mesh = np.meshgrid(X, Y)

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

    loss_all = np.zeros_like(xcoord_mesh)


    for i in range(n_space):
        for j in range(n_space):
            set_weights(model, weight, directions=[x_direction, y_direction], step=[
                        xcoord_mesh[i, j], ycoord_mesh[i, j]])
            loss, _ = eval_loss(
                model, loss_fn, train_loader, args)

            loss_all[i, j] = loss

    return xcoord_mesh, ycoord_mesh, loss_all, x_lst, y_lst


def get_random_contour(args, checkpoint, n_space=20, x_bound=[-1, 1], y_bound=[-1, 1]):
    """
    It takes a list of checkpoints, and returns the contour plot of the loss function
    
    :param args: the arguments for the model
    :param checkpoint_lst: a list of checkpoints, each checkpoint is a dictionary of parameters
    :param n_space: the number of points in each dimension of the grid, defaults to 20 (optional)
    :return: the x and y coordinates of the contour plot, the loss values for each point, and the x and
    y coordinates of the trajectory.
    """
    seed_torch(args.seed)

    act_func = get_act_func(args.act_func)

    if args.network_type == 'linear':
        model = Linear(args.t, args.hidden_layers_width, args.input_dim,
                       args.output_dim, act_func, args.initialization, args.dropout, args.dropout_pro, args.bias).to(args.device)
    if args.network_type == 'vgg':
        model = vgg.VGG9(dropout=False).to(args.device)
    print(model)

    model = model.cpu()

    model.load_state_dict(checkpoint)

    state=copy.deepcopy(model.state_dict())

    # weight = get_weights(model)

    x_direction=create_random_direction(model, ignore='biasbn', norm='filter')

    y_direction = create_random_direction(model, ignore='biasbn', norm='filter')



    X = np.linspace(x_bound[0], x_bound[1], n_space)
    Y = np.linspace(y_bound[0], y_bound[1], n_space)
    xcoord_mesh, ycoord_mesh = np.meshgrid(X, Y)

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

    loss_all = np.zeros_like(xcoord_mesh)

    for i in range(n_space):
        for j in range(n_space):
            state_new=set_states(model.state_dict(), directions=[x_direction, y_direction], step=[
                        xcoord_mesh[i, j], ycoord_mesh[i, j]])
            model.load_state_dict(state_new)
            loss, _ = eval_loss(
                model, loss_fn, train_loader, args)

            loss_all[i, j] = loss

            model.load_state_dict(state)

    return xcoord_mesh, ycoord_mesh, loss_all


def get_loss_landscape(args, checkpoint_lst, alpha_list):
    """
    It takes a list of checkpoints and a list of alphas, and returns the loss for each alpha
    
    :param args: the arguments for the model
    :param checkpoint_lst: a list of two checkpoints. The first checkpoint is the starting point, and
    the second checkpoint is the direction
    :param alpha_list: a list of values that we will use to interpolate between the two
    checkpoints
    :return: The loss landscape is being returned.
    """

    direction = get_diff_states(checkpoint_lst[0], checkpoint_lst[1])

    model_state_dict_list = []

    for alpha in alpha_list:

        model_state_dict_list.append(set_states(
            checkpoint_lst[0], [direction], step=alpha))

    loss = get_loss(args, model_state_dict_list, use_cuda=False,
                    get_gradient=False, get_Hessian=False)

    return(np.array(loss)[:, 0])







