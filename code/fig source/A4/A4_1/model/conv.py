import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast


class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2*2*64,100)
        self.mlp2 = torch.nn.Linear(100,10)

        self._initialize_weights

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x

    def _initialize_weights(self) -> None:
        if self.initialization=='Gaussian':
            for obj in self.modules():
                if isinstance(obj, (nn.Linear,nn.Conv2d)):
                    nn.init.normal_(obj.weight.data, 0, 1 /
                                    self.hidden_layers_width[0]**(self.t))
                    if obj.bias is not None:
                        nn.init.normal_(obj.bias.data, 0, 1 /
                                        self.hidden_layers_width[0]**(self.t))
        elif self.initialization=='Xavier':
            for obj in self.modules():
                if isinstance(obj, (nn.Linear,nn.Conv2d)):
                    nn.init.xavier_normal_(obj.weight.data)
                    if obj.bias is not None:
                        nn.init.constant_(obj.bias, 0)
        elif self.initialization=='He':
            for obj in self.modules():
                if isinstance(obj, (nn.Linear,nn.Conv2d)):
                    nn.init.kaiming_normal_(obj.weight.data)
                    if obj.bias is not None:
                        nn.init.constant_(obj.bias, 0)
        else:
            print('There is no such initialization.')
