"""
@author: jpzxshi
"""
import torch

from .module import Module, StructureNN
from .fnn import FNN

class AdditiveCouplingLayer(Module):
    def __init__(self, D, d, layers, width, activation, initializer, mode):
        super(AdditiveCouplingLayer, self).__init__()
        self.D = D
        self.d = d
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.mode = mode
        
        self.modus = self.__init_modules()
        
    def forward(self, x1x2):
        x1, x2 = x1x2
        if self.mode == 'up':
            return x1 + self.modus['m'](x2), x2
        elif self.mode == 'low':
            return x1, x2 + self.modus['m'](x1)
        else:
            raise ValueError
            
    def inverse(self, y1y2):
        y1, y2 = y1y2
        if self.mode == 'up':
            return y1 - self.modus['m'](y2), y2
        elif self.mode == 'low':
            return y1, y2 - self.modus['m'](y1)
        else:
            raise ValueError
            
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        din, dout = (self.d, self.D - self.d) if self.mode == 'low' else (self.D - self.d, self.d)
        modules['m'] = FNN(din, dout, self.layers, self.width, self.activation, self.initializer)
        return modules
    
class CouplingLayer(Module):
    def __init__(self, D, d, layers, width, activation, initializer, mode):
        super(CouplingLayer, self).__init__()
        self.D = D
        self.d = d
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.mode = mode
        
        self.modus = self.__init_modules()
        
    def forward(self, x1x2):
        x1, x2 = x1x2
        if self.mode == 'up':
            return x1 * torch.exp(self.modus['s'](x2)) + self.modus['t'](x2), x2
        elif self.mode == 'low':
            return x1, x2 * torch.exp(self.modus['s'](x1)) + self.modus['t'](x1)
        else:
            raise ValueError
            
    def inverse(self, y1y2):
        y1, y2 = y1y2
        if self.mode == 'up':
            return (y1 - self.modus['t'](y2)) * torch.exp(-self.modus['s'](y2)), y2
        elif self.mode == 'low':
            return y1, (y2 - self.modus['t'](y1)) * torch.exp(-self.modus['s'](y1))
        else:
            raise ValueError
            
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        din, dout = (self.d, self.D - self.d) if self.mode == 'low' else (self.D - self.d, self.d)
        modules['s'] = FNN(din, dout, self.layers, self.width, self.activation, self.initializer)
        modules['t'] = FNN(din, dout, self.layers, self.width, self.activation, self.initializer)
        return modules
    
class INN(StructureNN):
    '''Invertible neural network. (NICE and real NVP)
    '''
    def __init__(self, D, d, layers=3, sublayers=3, subwidth=20, activation='sigmoid', initializer='default', volume_preserving=False):
        super(INN, self).__init__()
        self.D = D
        self.d = d
        self.layers = layers
        self.sublayers = sublayers
        self.subwidth = subwidth
        self.activation = activation
        self.initializer = initializer
        self.volume_preserving = volume_preserving
        
        self.modus = self.__init_modules()
        
    def forward(self, x):
        x = x[..., :self.d], x[..., self.d:]
        for i in range(self.layers):
            x = self.modus['M{}'.format(i+1)](x)
        return torch.cat(x, -1)
    
    def inverse(self, y):
        y = y[..., :self.d], y[..., self.d:]
        for i in reversed(range(self.layers)):
            y = self.modus['M{}'.format(i+1)].inverse(y)
        return torch.cat(y, -1)
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(self.layers):
            mode = 'up' if i % 2 == 0 else 'low'
            if self.volume_preserving:
                modules['M{}'.format(i+1)] = AdditiveCouplingLayer(self.D, self.d, self.sublayers, self.subwidth, self.activation, self.initializer, mode)
            else:
                modules['M{}'.format(i+1)] = CouplingLayer(self.D, self.d, self.sublayers, self.subwidth, self.activation, self.initializer, mode)
        return modules