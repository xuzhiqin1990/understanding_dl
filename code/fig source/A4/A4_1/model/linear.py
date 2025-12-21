import torch.nn as nn
from typing import List


class Linear(nn.Module):
    def __init__(self, t, hidden_layers_width=[100],  input_size=20, num_classes: int = 1000, act_layer: nn.Module = nn.ReLU(), initialization='Gaussian',dropout=False,dropout_pro=0.2,bias=True):
        super(Linear, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_layers_width = hidden_layers_width
        self.t = t
        self.feature_info = []
        self.initialization=initialization
        layers: List[nn.Module] = []
        self.layers_width = [self.input_size]+self.hidden_layers_width
        for i in range(len(self.layers_width)-1):
            if bias:
                layers += [nn.Linear(self.layers_width[i],
                    self.layers_width[i+1]), act_layer]

            else:
                layers += [nn.Linear(self.layers_width[i],
                                    self.layers_width[i+1], bias=False), act_layer]
            # layers += [nn.Linear(self.layers_width[i],
            #                      self.layers_width[i+1])]
        if dropout:
            layers += [nn.Dropout(dropout_pro)]
        layers += [nn.Linear(self.layers_width[-1], num_classes, bias=False)]
        self.features = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):

        x=x.view(x.size(0), -1)
        # print(x.shape)
        x = self.features(x)
        return x

    def _initialize_weights(self) -> None:
        # print(self.initialization)
        if self.initialization=='Gaussian':
            for obj in self.modules():
                if isinstance(obj, (nn.Linear,nn.Conv2d)):
                    nn.init.normal_(obj.weight.data, 0, 1 /
                                    self.hidden_layers_width[0]**(self.t))
                    # nn.init.normal_(obj.weight, 0, 0.01)
                    if obj.bias is not None:
                        nn.init.normal_(obj.bias.data, 0, 1 /
                                        self.hidden_layers_width[0]**(self.t))
                        # nn.init.constant_(obj.bias, 0)
        elif self.initialization=='Xavier':
            for obj in self.modules():
                if isinstance(obj, (nn.Linear,nn.Conv2d)):
                    nn.init.xavier_normal_(obj.weight.data)
                    if obj.bias is not None:
                        nn.init.constant_(obj.bias, 0)
        elif self.initialization=='He':
            for obj in self.modules():
                if isinstance(obj, (nn.Linear,nn.Conv2d)):
                    nn.init.kaiming_normal_(obj.weight.data, mode='fan_out', nonlinearity='relu')
                    # nn.init.normal_(obj.weight, 0, 0.01)
                    if obj.bias is not None:
                        nn.init.constant_(obj.bias, 0)
        else:
            print('There is no such initialization.')