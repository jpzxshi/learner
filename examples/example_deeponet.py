"""
@author: jpzxshi
"""
import sys
sys.path.append('..')
import numpy as np
from sklearn import gaussian_process as gp
from scipy import interpolate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import learner as ln

class AntideData(ln.Data):
    '''Data for learning the antiderivative operator.
    '''
    def __init__(self, s0, sensors, p, length_scale, train_num, test_num):
        super(AntideData, self).__init__()
        self.s0 = s0
        self.sensors = sensors
        self.p = p
        self.length_scale = length_scale
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
        
    def __init_data(self):
        features = 2000
        train = self.__gaussian_process(self.train_num, features)
        test = self.__gaussian_process(self.test_num, features)
        self.X_train, self.y_train = self.__generate(train)
        self.X_test, self.y_test = self.__generate(test)
        
    def __generate(self, gps):
        def generate(gp):
            u = interpolate.interp1d(np.linspace(0, 1, num=gp.shape[-1]), gp, kind='cubic', copy=False, assume_sorted=True)
            x = np.sort(np.random.rand(self.p))
            y = solve_ivp(lambda t, y: u(t), [0, 1], self.s0, 'RK45', x).y[0]
            u_sensors = u(np.linspace(0, 1, num=self.sensors))
            return np.hstack([np.tile(u_sensors, (self.p, 1)), x[:, None], y[:, None]])
        res = np.vstack(list(map(generate, gps)))
        return (res[..., :-2], res[..., -2:-1]), res[..., -1:]
    
    def __gaussian_process(self, num, features):
        x = np.linspace(0, 1, num=features)[:, None]
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + 1e-13 * np.eye(features))
        return (L @ np.random.randn(features, num)).transpose()
    
def plot(data, net):
    x = np.linspace(0, 1, num=data.sensors)
    y = np.linspace(0, 1, num=100)
    cos = np.cos(2 * np.pi * x)
    antide_true = np.sin(2 * np.pi * y) / (2 * np.pi)
    antide_pred = net.predict([np.tile(cos, (len(y), 1)), y[:, None]], returnnp=True).squeeze()
    
    plt.plot(x, cos, color='black', label=r'Input: $\cos(2\pi x)$', zorder=0)
    plt.plot(y, antide_true, color='b', label=r'Output: $\sin(2\pi x)/(2\pi)$', zorder=1)
    plt.plot(y, antide_pred, color='r', label=r'Prediction', zorder=2)
    plt.legend()
    plt.savefig('deeponet.pdf')
    
def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    s0 = [0]
    sensors = 100
    p = 1
    length_scale = 0.2
    train_num = 1000
    test_num = 1000
    # deeponet
    branch_size = [sensors, 40, 40]
    trunk_size = [1, 40, 40, 0] # = [1, 40, 40] followed by activation
    activation='relu'
    # training
    lr = 0.001
    iterations = 5000
    print_every = 100
    
    data = AntideData(s0, sensors, p, length_scale, train_num, test_num)
    net = ln.nn.DeepONet(branch_size, trunk_size, activation)
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