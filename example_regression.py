"""
@author: jpzxshi
"""
import numpy as np
import matplotlib.pyplot as plt

import learner as ln

class RGData(ln.Data):
    def __init__(self):
        super(RGData, self).__init__()
        self.__init_data()
    
    def __init_data(self):
        self.X_train = (2 * np.pi * np.random.rand(30))[:, None]
        self.y_train = np.sin(self.X_train)
        self.X_test = np.linspace(0, 2 * np.pi, num=100)[:, None]
        self.y_test = np.sin(self.X_test)
        
def plot(data, net):
    x = data.X_test_np
    f_true = data.y_test_np
    f_pred = net.predict(data.X_test, returnnp=True)
    
    plt.plot(x, f_true, color='b', label='Ground truth', zorder=0)
    plt.plot(x, f_pred, color='r', label='Prediction', zorder=1)
    plt.scatter(data.X_train_np, data.y_train_np, color='b', label='Learned data', zorder=2)
    plt.title(r'$\sin(x)$', fontsize=20)
    plt.legend()
    plt.savefig('regression.pdf')

def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # FNN
    depth = 3
    width = 30
    activation = 'relu'
    # training
    lr = 0.01
    iterations = 20000
    print_every = 1000
    
    data = RGData()
    net = ln.nn.FNN(data.dim, data.K, depth, width, activation)
    args = {
        'data': data,
        'net': net,
        'criterion': 'MSE',
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    plot(data, ln.Brain.Best_model())
    
if __name__ == '__main__':
    main()