"""
@author: jpzxshi
"""
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import learner as ln
from learner.integrator.hamiltonian import SV

class LVData(ln.Data):
    '''Data for learning the Lotka-Volterra system with
    H(u,v)=u−ln(u)+v−2ln(v), B(u,v)=[[0, uv], [-uv, 0]].
    (p,q)=(ln(u),ln(v)): K(p,q)=p-exp(p)+2q-exp(q).
    '''
    def __init__(self, z0, h, train_num, test_num):
        super(LVData, self).__init__()
        self.f = lambda t, y: y * (y @ np.array([[0, -1], [1, 0]]) + np.array([-2, 1]))
        self.dK = lambda p, q: (np.ones_like(p) - np.exp(p), 2 * np.ones_like(q) - np.exp(q))
        self.solver = SV(None, self.dK, iterations=1, order=4, N=10)
        self.z0 = np.array(z0)
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
        
    def generate_flow(self, z0, h, num):
        x0 = np.log(np.array(z0))
        return np.exp(self.solver.flow(x0, h, num))
        
    def __init_data(self):
        train_flow = self.generate_flow(self.z0, self.h, self.train_num)
        test_flow = self.generate_flow(train_flow[..., -1, :], self.h, self.test_num)
        self.X_train = train_flow[..., :-1, :].reshape(-1, 2)
        self.y_train = train_flow[..., 1:, :].reshape(-1, 2)
        self.X_test = test_flow[..., :-1, :].reshape(-1, 2)
        self.y_test = test_flow[..., 1:, :].reshape(-1, 2)
        
def plot(data, net):
    steps = 1000
    z0 = data.X_test_np.reshape(3, -1, 2)[:, 0, :]
    flow_true = data.generate_flow(z0, data.h, steps)
    flow_pred = net.predict(z0, steps, keepinitx=True, returnnp=True)
    
    for i in range(flow_true.shape[0]):
        label_true = 'Ground truth' if i == 0 else None
        label_pred = 'PNN' if i == 0 else None
        plt.plot(flow_true[i, :, 0], flow_true[i, :, 1], color='b', label=label_true, zorder=1)
        plt.plot(flow_pred[i, :, 0], flow_pred[i, :, 1], color='r', label=label_pred, zorder=2)
    plt.legend()
    plt.savefig('pnn.pdf')
        
def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    z0 = [[1, 0.8], [1, 1], [1, 1.2]]
    h = 0.1
    train_num = 100
    test_num = 100
    # PNN
    inn_volume_preserving = False
    inn_layers = 3
    inn_sublayers = 2
    inn_subwidth = 30
    inn_activation = 'sigmoid'
    symp_type = 'G' # 'LA' or 'G'
    symp_LAlayers = 3
    symp_LAsublayers = 2
    symp_Glayers = 3
    symp_Gwidth = 30
    symp_activation = 'sigmoid'
    # training
    lr = 0.001
    iterations = 200000
    print_every = 1000
    
    data = LVData(z0, h, train_num, test_num)
    inn = ln.nn.INN(data.dim, data.dim // 2, inn_layers, inn_sublayers, inn_subwidth, inn_activation, 
                    volume_preserving=inn_volume_preserving)
    if symp_type == 'LA':
        sympnet = ln.nn.LASympNet(data.dim, symp_LAlayers, symp_LAsublayers, symp_activation)
    elif symp_type == 'G':
        sympnet = ln.nn.GSympNet(data.dim, symp_Glayers, symp_Gwidth, symp_activation)
    net = ln.nn.PNN(inn, sympnet)
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