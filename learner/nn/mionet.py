"""
@author: jpzxshi
"""
import torch
from .module import Map
from .fnn import FNN

class MIONet(Map):
    '''Multiple-input operator network.
    Input: ([batch, sensors1], [batch, sensors2], [batch, dim_loc])
    Output: [batch, 1]
    '''
    def __init__(self, sizes, activation='relu', initializer='default', bias=True):
        super(MIONet, self).__init__()
        self.sizes = sizes
        self.activation = activation
        self.initializer = initializer
        self.bias = bias

        self.ms = self.__init_modules()
        self.ps = self.__init_parameters()
    
    def forward(self, x):
        y = torch.stack([self.ms['Net{}'.format(i + 1)](x[i]) for i in range(len(self.sizes))])
        y = torch.sum(torch.prod(y, dim=0), dim=-1, keepdim=True)
        return y + self.ps['bias'] if self.bias else y
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(len(self.sizes)):
            modules['Net{}'.format(i + 1)] = FNN(self.sizes[i], self.activation, self.initializer)
        return modules
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        if self.bias:
            parameters['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return parameters
    
class MIONet_Cartesian(Map):
    '''Multiple-input operator network (Cartesian product version).
    Input: ([batch, sensors1], [batch, sensors2], [num_loc, dim_loc])
    Output: [batch, num_loc]
    '''
    def __init__(self, sizes, activation='relu', initializer='default', bias=True):
        super(MIONet_Cartesian, self).__init__()
        self.sizes = sizes
        self.activation = activation
        self.initializer = initializer
        self.bias = bias

        self.ms = self.__init_modules()
        self.ps = self.__init_parameters()
    
    def forward(self, x):
        y1 = torch.stack([self.ms['Net{}'.format(i + 1)](x[i]) for i in range(len(self.sizes) - 1)])
        y1 = torch.prod(y1, dim=0)
        y2 = self.ms['Net{}'.format(len(self.sizes))](x[-1])
        y = y1 @ y2.t()
        return y + self.ps['bias'] if self.bias else y
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(len(self.sizes)):
            modules['Net{}'.format(i + 1)] = FNN(self.sizes[i], self.activation, self.initializer)
        return modules
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        if self.bias:
            parameters['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return parameters