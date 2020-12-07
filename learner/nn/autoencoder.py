"""
@author: jpzxshi
"""
import torch

from .module import StructureNN
from .fnn import FNN

class AE(StructureNN):
    '''Autoencoder.
    '''
    def __init__(self, data_dim, latent_dim, depth, width, activation='sigmoid', initializer='default'):
        super(AE, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.depth = depth
        self.width = width
        self.activation = activation
        self.initializer = initializer
        
        self.modus = self.__init_modules()
    
    def forward(self, x):
        return self.modus['decoder'](self.modus['encoder'](x))
    
    def encode(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.modus['encoder'](x).cpu().detach().numpy() if returnnp else self.modus['encoder'](x)
    
    def decode(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.modus['decoder'](x).cpu().detach().numpy() if returnnp else self.modus['decoder'](x)
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['encoder'] = FNN(self.data_dim, self.latent_dim, self.depth, self.width, self.activation, self.initializer)
        modules['decoder'] = FNN(self.latent_dim, self.data_dim, self.depth, self.width, self.activation, self.initializer)         
        return modules