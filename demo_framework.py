"""
@author: jpzxshi
"""
import numpy as np
import matplotlib.pyplot as plt
import learner as ln

####
# A simple example of fitting one-dimensional function sin(x) using fully-connected
# neural network, to show the framework of the package 'learner'.
####

def main():
    # Setting training parameters here.
    training_args = {
        'criterion': 'MSE', # 'MSE', 'CrossEntropy', None, ...
        'optimizer': 'Adam', # 'Adam', 'LBFGS', ...
        'lr': 0.01,
        'iterations': 10000,
        'batch_size': None,
        'print_every': 1000,
        'save': 'best_only', # 'best_only', 'all', False
        'callback': None,
        'dtype': 'double', # 'float', 'double'
        'device': 'cpu', # 'cpu', 'gpu'
    }
    
    #### The process
    ln.Brain.Start()
    
    # Loading data and constructing neural network.
    data = My_data()
    net = My_net()
    #net = ln.nn.FNN([1, 50, 1], 'relu') # using FNN provided by 'learner'
    
    ln.Brain.Init(data, net) # Initializing
    
    # Training by the given arguments.
    ln.Brain.Run(**training_args) # Running
    ln.Brain.Restore() # Restoring the best model to 'net'
    
    # One may change the training argument(s) and run a second round if needed.
    training_args['lr'] = 0.001
    ln.Brain.Run(**training_args)
    ln.Brain.Restore()
    
    ln.Brain.Output() # Outputting the result(s) to a (default/specified) directory
    
    postprocessing(data, ln.Brain.Best_model()) # Postprocessing
    
    ln.Brain.End()
    #### The process

# A class inherited from 'ln.Data'. User is required to provide data in four
# attributes 'self.X_train', 'self.y_train', 'self.X_test', and 'self.y_test'.
# Both numpy and torch data are available.
class My_data(ln.Data):
    def __init__(self):
        super(My_data, self).__init__()
        self.X_train = (2 * np.pi * np.random.rand(30))[:, None]
        self.y_train = np.sin(self.X_train)
        self.X_test = np.linspace(0, 2 * np.pi, num=100)[:, None]
        self.y_test = np.sin(self.X_test)

# If needed, user may construct customized neural network architecture based
# on torch. Here is an example of the 'Map' version. The 'Algorithm' version 
# refers to 'demo_pinn.py'.
import torch
class My_net(ln.nn.Map):
    def __init__(self):
        super(My_net, self).__init__()
        self.layer_1 = torch.nn.Linear(1, 50)
        self.layer_act = torch.relu
        self.layer_2 = torch.nn.Linear(50, 1)
    def forward(self, x):
        return self.layer_2(self.layer_act(self.layer_1(x)))

# Postprocessing, e.g., plotting a figure.
def postprocessing(data, net):
    x = data.X_test_np
    f_true = data.y_test_np
    f_pred = net.predict(data.X_test, returnnp=True)
    plt.plot(x, f_true, color='b', label='Ground truth', zorder=0)
    plt.plot(x, f_pred, color='r', label='Prediction', zorder=1)
    plt.scatter(data.X_train_np, data.y_train_np, color='b', label='Learned data', zorder=2)
    plt.title(r'$\sin(x)$', fontsize=20)
    plt.legend()
    plt.savefig('framework.pdf')
    
if __name__ == '__main__':
    main()