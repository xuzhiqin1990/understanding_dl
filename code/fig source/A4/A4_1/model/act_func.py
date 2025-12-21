import math

import torch
import torch.nn as nn


class sin_torch(nn.Module):  # Custom activation function
    def __init__(self):
        super(sin_torch, self).__init__()

    def forward(self, x):
        # return x ** 50  # or F.relu(x) * F.relu(1-x)
        return torch.sin(x)


class ricker_torch(nn.Module):  # Custom activation function
    def __init__(self):
        super(ricker_torch, self).__init__()

    def forward(self, x):
        # return x ** 50  # or F.relu(x) * F.relu(1-x)
        a=1
        return 2/((3*a)*(math.pi**0.25)) * (1 - (x/a)**2) * torch.exp(-0.5*(x/a)**2)


def get_act_func(act_func):
    if act_func == 'Tanh':
        return nn.Tanh()
    elif act_func == 'ReLU':
        return nn.ReLU()

    elif act_func == 'sin':
        return sin_torch()

    elif act_func == 'richer':
        return ricker_torch()

    else:
        raise NameError('No such act func!')
