"""
@author: jpzxshi
"""
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import learner as ln
from learner.integrator.hamiltonian import SV

class PDData(ln.Data):
    '''Data for learning the pendulum system with the Hamiltonian H(p,q)=(1/2)p^2−cos(q).
    '''
    def __init__(self, x0, h, train_num, test_num):
        super(PDData, self).__init__()
        self.dH = lambda p, q: (p, np.sin(q))
        self.solver = SV(None, self.dH, iterations=1, order=6, N=10)
        self.x0 = x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
    
    def __generate_flow(self, x0, h, num):
        X = self.solver.flow(np.array(x0), h, num)
        return X[:-1], X[1:]
    
    def __init_data(self):
        self.X_train, self.y_train = self.__generate_flow(self.x0, self.h, self.train_num)
        self.X_test, self.y_test = self.__generate_flow(self.y_train[-1], self.h, self.test_num)
        
def plot(data, net):
    steps = 1000
    flow_true = data.solver.flow(data.X_test_np[0], data.h, steps)
    flow_pred = net.predict(data.X_test[0], steps, keepinitx=True, returnnp=True)
    
    plt.plot(flow_true[:, 0], flow_true[:, 1], color='b', label='Ground truth', zorder=0)
    plt.plot(flow_pred[:, 0], flow_pred[:, 1], color='r', label='Predicted flow', zorder=1)
    plt.scatter(data.X_train_np[:, 0], data.X_train_np[:, 1], color='b', label='Learned data', zorder=2)
    plt.legend()
    plt.savefig('sympnet.pdf')

def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    x0 = [0, 1]
    h = 0.1
    train_num = 40
    test_num = 100
    # SympNets
    net_type = 'LA' # 'LA' or 'G'
    LAlayers = 3
    LAsublayers = 2
    Glayers = 5
    Gwidth = 30
    activation = 'sigmoid'
    # training
    lr = 0.001
    iterations = 50000
    print_every = 1000
    
    data = PDData(x0, h, train_num, test_num)
    if net_type == 'LA':
        net = ln.nn.LASympNet(data.dim, LAlayers, LAsublayers, activation)
    elif net_type == 'G':
        net = ln.nn.GSympNet(data.dim, Glayers, Gwidth, activation)
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