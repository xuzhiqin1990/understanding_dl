import os
import time
import warnings
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import argparse
import matplotlib
import matplotlib.pyplot as plt
import datetime
import re



def my_fft(data, freq_len=40, isnorm=1):

    """
    This function performs FFT on the given data.
    
    Args:
    data (numpy.ndarray): The input data.
    freq_len (int): The length of the frequency.
    isnorm (int): The normalization factor.
    
    Returns:
    return_fft (numpy.ndarray): The FFT output array.
    """

    # second_diff_input = np.mean(np.diff(np.diff(np.squeeze(x_input))))
    # if abs(second_diff_input) < 1e-10:
    datat = np.squeeze(data)
    datat_fft = np.fft.fft(datat)
    ind2 = range(freq_len)
    fft_coe = datat_fft[ind2]
    if isnorm == 1:
        return_fft = np.absolute(fft_coe)
    else:
        return_fft = fft_coe
    # else:
    #     return_fft = get_ft_multi(
    #         x_input, data, kk=kk, freq_len=freq_len, min_f=min_f, max_f=max_f, isnorm=isnorm)
    return return_fft


# def get_ft_multi(x_input, data, kk=0, freq_len=100, min_f=0, max_f=np.pi/3, isnorm=1):

#     """
#     This function returns the FFT output array for non-uniformly spaced inputs.

#     Args:
#     x_input (numpy.ndarray): The input array.
#     data (numpy.ndarray): The input data.
#     kk (int): The value of k.
#     freq_len (int): The length of the frequency.
#     min_f (float): The minimum value of frequency.
#     max_f (float): The maximum value of frequency.
#     isnorm (int): The normalization factor.

#     Returns:
#     return_fft (numpy.ndarray): The FFT output array.

#     """

#     n = x_input.shape[1]
#     if np.max(abs(kk)) == 0:
#         k = np.linspace(min_f, max_f, num=freq_len, endpoint=True)
#         kk = np.matmul(np.ones([n, 1]), np.reshape(k, [1, -1]))
#     tmp = np.matmul(np.transpose(data), np.exp(-1J * (np.matmul(x_input, kk))))
#     if isnorm == 1:
#         return_fft = np.absolute(tmp)
#     else:
#         return_fft = tmp
#     return np.squeeze(return_fft)


def SelectPeakIndex(FFT_Data, endpoint=True):

    """
    This function selects the peak index from FFT data.
    
    Args:
    FFT_Data (numpy.ndarray): The FFT data array.
    endpoint (bool): Whether to include endpoints or not. Default is True.
    
    Returns:
    sel_ind (numpy.ndarray): Selected index array with peaks. 
    """
    
    D1 = FFT_Data[1:-1]-FFT_Data[0:-2]
    D2 = FFT_Data[1:-1]-FFT_Data[2:]
    D3 = np.logical_and(D1 > 0, D2 > 0)
    tmp = np.where(D3 == True)
    sel_ind = tmp[0]+1
    if endpoint:
        if FFT_Data[0]-FFT_Data[1] > 0:
            sel_ind = np.concatenate([[0], sel_ind])
        if FFT_Data[-1]-FFT_Data[-2] > 0:
            Last_ind = len(FFT_Data)-1
            sel_ind = np.concatenate([sel_ind, [Last_ind]])
    return sel_ind

def dft_analysis(f, T, N):

    """
    Perform DFT analysis on a given function f with a total time of T and N samples.

    Args:
        f (function): The function to be analyzed.
        T (float): The total time of the signal.
        N (int): The number of samples.

    Returns:
        None
    """

    

    t = np.linspace(0, T, N, endpoint=False)
    x = f(t)
    X = np.fft.fft(x)
    t_target = np.linspace(0, T, 1000, endpoint=False)
    x_target = f(t_target)

    freqs = np.fft.fftfreq(N, d=T/N)

    fig = plt.figure(figsize=(12.0, 5.4))
    axes = [fig.add_axes(
    [.1, .25, .25, .6]),fig.add_axes(
    [.45, .25, .25, .6])]
    axes[0].stem(freqs, np.abs(X))
    axes[0].set_xlabel('frequence', fontsize=18)
    axes[0].set_ylabel('amplitude', fontsize=18)
    axes[0].set_xlim(-5.5,5.5)
    axes[1].plot(t_target,x_target)
    axes[1].scatter(t,x, color='r')
    axes[1].set_xlabel('t', fontsize=18)
    axes[1].set_ylabel('x', fontsize=18)
    axes[0].tick_params(labelsize=18)
    axes[1].tick_params(labelsize=18)
    plt.show()
    plt.close()


def mkdirs(fn):  
    
    """
    Create directories if they don't exist.

    Args:
    fn: The directory path to create.
    """

    if not os.path.isdir(fn):
        os.makedirs(fn)


def create_save_dir(path_ini):
    """
    Create a new directory with the current date and time as its name and return the path of the new directory.

    Args:
    path_ini: The initial path to create the new directory.

    Return:
    The path of the new directory.
    """
    subFolderName = re.sub(r'[^0-9]', '', str(datetime.datetime.now()))
    path = os.path.join(path_ini, subFolderName)
    mkdirs(path)
    mkdirs(os.path.join(path, 'output'))
    return path

def get_dataset(args, target_func):

    for i in range(2):
        if isinstance(args.boundary[i], str):
            args.boundary[i] = eval(args.boundary[i])

    test_input = torch.reshape(torch.linspace(args.boundary[0] - 0.5, args.boundary[1] + 0.5, steps=args.test_size), [args.test_size, 1])


    training_input = torch.reshape(torch.linspace(args.boundary[0], args.boundary[1], steps=args.training_size), [args.training_size, 1])
    test_target = target_func(test_input)
    training_target = target_func(training_input)

    return training_input, training_target, test_input, test_target

def plot_target(args):

    """
    Plot the target.

    Args:
        args (object): object containing training and test input and target.

    """

    plt.figure()
    ax = plt.gca()

    plt.plot(args.training_input.detach().cpu().numpy(),
             args.training_target.detach().cpu().numpy(), 'b*', label='True')
    plt.plot(args.test_input.detach().cpu().numpy(),
             args.test_target.detach().cpu().numpy(), 'r-', label='Test')

    ax.tick_params(labelsize=18)
    plt.legend(fontsize=18)
    plt.show()


# Define the activation functions
def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hat(x):
    return np.piecewise(x, [np.logical_and(0 <= x, x < 1), np.logical_and(1 <= x, x < 2), x >= 2], [lambda x: x, lambda x: 2 - x, 0])

# Define the input range

def plot_activation():

    x = np.linspace(-2, 2, 1000)

    # Create a figure with four subplots
    fig, axs = plt.subplots(2, 2)

    # Plot the activation functions in the subplots
    axs[0, 0].plot(x, tanh(x))
    axs[0, 0].set_title('Tanh')

    axs[0, 1].plot(x, relu(x))
    axs[0, 1].set_title('ReLU')

    axs[1, 0].plot(x, sigmoid(x))
    axs[1, 0].set_title('Sigmoid')

    axs[1, 1].plot(x, hat(x))
    axs[1, 1].set_title('Hat')

    # Add labels to the subplots
    for ax in axs.flat:
        ax.set(xlabel='Input', ylabel='Output')

    # Adjust the layout of the subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.close()


class Model(nn.Module):
    def __init__(self, t, hidden_layers_width=[100],  input_size=20, num_classes: int = 1000, act_layer: nn.Module = nn.ReLU()):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_layers_width = hidden_layers_width
        self.t = t
        layers: List[nn.Module] = []
        self.layers_width = [self.input_size]+self.hidden_layers_width
        for i in range(len(self.layers_width)-1):
            layers += [nn.Linear(self.layers_width[i],
                                    self.layers_width[i+1]), act_layer]
        layers += [nn.Linear(self.layers_width[-1], num_classes)]
        self.features = nn.Sequential(*layers)
        self._initialize_weights()


    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x

    def _initialize_weights(self) -> None:

        for obj in self.modules():
            if isinstance(obj, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(obj.weight.data, 0, 1 /
                                self.hidden_layers_width[0]**(self.t))
                if obj.bias is not None:
                    nn.init.normal_(obj.bias.data, 0, 1 /
                                    self.hidden_layers_width[0]**(self.t))

class hat_torch(nn.Module):  # Custom activation function
    def __init__(self):
        super(hat_torch, self).__init__()

    def forward(self, x):
        y = torch.where((x > 0) & (x < 1), x,  torch.tensor(0.0))
        y = torch.where((x >= 1) & (x < 2), 2 - x, y)
        return y



def get_act_func(act_func):

    """
    Get activation function.

    Args:
        act_func (str): activation function name.

    Returns:
        nn.Module: activation function.
    """

    if act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'ReLU':
        return nn.ReLU()
    elif act_func == 'Sigmoid':
        return nn.Sigmoid()
    elif act_func == 'hat':
        return hat_torch()
    else:
        raise NameError('No such act func!')


def train_one_step(model, optimizer, loss_fn,  args):

    """
    Train one step.

    Args:
        model (nn.Module): model.
        optimizer (optim.Optimizer): optimizer.
        loss_fn (nn.Module): loss function.
        args (object): object containing training and test input and target.

    Returns:
        tuple: tuple containing loss and outputs.
    """

    # Set the model to training mode (Dropout, BN, etc.)
    model.train()

    # Get the device to use for computation
    device = args.device

    training_loss = 0.0
    total = 0

    # Move the training input and target to the device and convert the target to float
    data, target = args.training_input.to(device), args.training_target.to(device).to(torch.float)
    args.batch_size = args.training_size
    # Loop over the training data in batches
    for i in range(args.training_size // args.batch_size ):  

        # Zero the gradients of the optimizer
        optimizer.zero_grad()

        # Randomly select a batch of data
        mask = np.random.choice(args.training_size, args.batch_size, replace=False)

        # Get the model's predictions for the selected data
        y_train = model(data[mask])

        # Calculate the loss between the predictions and the target
        loss = loss_fn(y_train, target[mask])

        # Compute the gradients of the loss with respect to the model parameters
        loss.backward()

        #loss.grad.data

        # Update the model parameters using the optimizer
        optimizer.step()

        training_loss+=loss.item()*args.batch_size
        total += args.batch_size

    # Get the model's predictions for all the training data
    outputs = model(data)

    return training_loss/total, outputs


def test(model, loss_fn, args):
    """
    Test.

    Args:
        model (nn.Module): model.
        loss_fn (nn.Module): loss function.
        args (object): object containing training and test input and target.

    Returns:
        tuple: tuple containing loss and outputs.
    """

    model.eval()
    device = args.device
    with torch.no_grad():
        data, target = args.test_input.to(
            device), args.test_target.to(device).to(torch.float)
        outputs = model(data)
        loss = loss_fn(outputs, target)

    return loss.item(), outputs

def plot_loss(path, loss_train, x_log=False):

    """
    Plot loss.

    Args:
        path (str): path.
        loss_train (list): list of training loss.
        x_log (bool): whether to use log scale for x-axis.

    Returns:
        None.
    """

    plt.figure()
    ax = plt.gca()
    y2 = np.asarray(loss_train)
    plt.plot(y2, 'k-', label='Train')
    plt.xlabel('epoch', fontsize=18)
    ax.tick_params(labelsize=18)
    plt.yscale('log')
    if x_log == False:
        fntmp = os.path.join(path, 'loss.png')

    else:
        plt.xscale('log')
        fntmp = os.path.join(path, 'loss_log.png')
    plt.tight_layout()
    plt.savefig(fntmp,dpi=300)


    plt.close()


def plot_model_output(path, args, output, epoch):

    plt.figure()
    ax = plt.gca()

    plt.plot(args.training_input.detach().cpu().numpy(),
             args.training_target.detach().cpu().numpy(), 'b*', label='True')
    plt.plot(args.test_input.detach().cpu().numpy(),
             output.detach().cpu().numpy(), 'r-', label='Test')

    ax.tick_params(labelsize=18)
    plt.legend(fontsize=18)
    fntmp = os.path.join(path, 'output', str(epoch)+'.png')

    plt.savefig(fntmp, dpi=300)


    plt.close()



