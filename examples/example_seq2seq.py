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
    def __init__(self, s0, sensor_in, sensor_out, length_scale, train_num, test_num):
        super(AntideData, self).__init__()
        self.s0 = s0
        self.sensor_in = sensor_in
        self.sensor_out = sensor_out
        self.length_scale = length_scale
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
        
    def __init_data(self):
        features = 2000
        train = self.__gaussian_process(self.train_num, features)
        test = self.__gaussian_process(self.test_num, features)
        self.X_train = self.__sense(train).reshape([self.train_num, self.sensor_in, 1])
        self.y_train = self.__solve(train).reshape([self.train_num, self.sensor_out, 1])
        self.X_test = self.__sense(test).reshape([self.test_num, self.sensor_in, 1])
        self.y_test = self.__solve(test).reshape([self.test_num, self.sensor_out, 1])
    
    def __gaussian_process(self, num, features):
        x = np.linspace(0, 1, num=features)[:, None]
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + 1e-13 * np.eye(features))
        return (L @ np.random.randn(features, num)).transpose()
    
    def __sense(self, gps):
        x = np.linspace(0, 1, num=gps.shape[1])
        res = map(
            lambda y: interpolate.interp1d(x, y, kind='cubic', copy=False, assume_sorted=True
            )(np.linspace(0, 1, num=self.sensor_in)),
            gps)
        return np.vstack(list(res))
    
    def __solve(self, gps):
        x = np.linspace(0, 1, num=gps.shape[1])
        interval = np.linspace(0, 1, num=self.sensor_out) if self.sensor_out > 1 else [1]
        def solve(y):
            u = interpolate.interp1d(x, y, kind='cubic', copy=False, assume_sorted=True)
            return solve_ivp(lambda t, y: u(t), [0, 1], self.s0, 'RK45', interval).y[0]
        return np.vstack(list(map(solve, gps)))
    
def plot(data, net):
    x = np.linspace(0, 1, num=data.sensor_in)
    y = np.linspace(0, 1, num=data.sensor_out)
    cos = np.cos(2 * np.pi * x)
    antide_true = np.sin(2 * np.pi * y) / (2 * np.pi)
    antide_pred = net.predict(cos[:, None], returnnp=True).squeeze()
    
    plt.plot(x, cos, color='black', label=r'Input: $\cos(2\pi x)$', zorder=0)
    plt.plot(y, antide_true, color='b', label=r'Output: $\sin(2\pi x)/(2\pi)$', zorder=1)
    plt.plot(y, antide_pred, color='r', label=r'Prediction', zorder=2)
    plt.legend()
    plt.savefig('seq2seq.pdf')
    
def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    s0 = [0]
    sensor_in = 100
    sensor_out = 100
    length_scale = 0.2
    train_num = 100
    test_num = 100
    # seq2seq
    cell = 'LSTM' # 'RNN', 'LSTM' or 'GRU'
    hidden_size = 5
    # training
    lr = 0.001
    iterations = 2000
    print_every = 100
    
    data = AntideData(s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    net = ln.nn.S2S(data.dim, sensor_in, data.K, sensor_out, hidden_size, cell)
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