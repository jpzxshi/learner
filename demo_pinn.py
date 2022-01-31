"""
@author: jpzxshi
"""
import numpy as np
import learner as ln
from learner.utils import mse, grad

class PoissonData(ln.Data):
    '''Data for solving the Poisson equation
    u_xx + u_yy = -2sin(x)sin(y),
    u(x,0)=0, u(x,1)=sin(x)sin(1), u(0,y)=0, u(1,y)=sin(1)sin(y),
    with solution u(x,y)=sin(x)sin(y).
    '''
    def __init__(self, train_num, test_num):
        super(PoissonData, self).__init__()
        self.train_num = train_num
        self.test_num = test_num
        
        self.__init_data()
        
    def generate(self, num):
        X, y = {}, {}
        # positions
        X['diff'] = np.random.rand(num['diff'], 2)
        X['x_0'] = np.hstack((np.random.rand(num['x_0'], 1), np.zeros((num['x_0'], 1))))
        X['x_1'] = np.hstack((np.random.rand(num['x_1'], 1), np.ones((num['x_1'], 1))))
        X['0_y'] = np.hstack((np.zeros((num['0_y'], 1)), np.random.rand(num['0_y'], 1)))
        X['1_y'] = np.hstack((np.ones((num['1_y'], 1)), np.random.rand(num['0_y'], 1)))
        # values (could be changed if needed)
        y['diff'] = -2 * np.sin(X['diff'][:, :1]) * np.sin(X['diff'][:, 1:])
        y['x_0'] = np.zeros((num['x_0'], 1))
        y['x_1'] = np.sin(X['x_1'][:, :1]) * np.sin(1)
        y['0_y'] = np.zeros((num['0_y'], 1))
        y['1_y'] = np.sin(1) * np.sin(X['1_y'][:, 1:])
        return X, y
    
    def __init_data(self):
        self.X_train, self.y_train = self.generate(self.train_num)
        self.X_test, self.y_test = self.generate(self.test_num)
        
class PoissonPINN(ln.nn.Algorithm):
    '''Physics-informed neural networks for solving the Poisson equation
    u_xx + u_yy = f(x,y),
    with boundary condition u(x,0), u(x,1), u(0,y), u(1,y).
    '''
    def __init__(self, net, lam=1):
        super(PoissonPINN, self).__init__()
        self.net = net
        self.lam = lam
        
    def criterion(self, X, y):
        z = X['diff'].requires_grad_(True)
        u = self.net(z)
        u_g = grad(u, z)
        u_x, u_y = u_g[:, :1], u_g[:, 1:]
        u_xx, u_yy = grad(u_x, z)[:, :1], grad(u_y, z)[:, 1:]
        MSEd = mse(u_xx + u_yy, y['diff'])
        MSEb = (mse(self.net(X['x_0']), y['x_0']) + mse(self.net(X['x_1']), y['x_1']) + 
                mse(self.net(X['0_y']), y['0_y']) + mse(self.net(X['1_y']), y['1_y']))
        return MSEd + self.lam * MSEb
    
    def predict(self, x, returnnp=False):
        return self.net.predict(x, returnnp)

def plot(data, net):
    import matplotlib.pyplot as plt
    import itertools
    X = np.array(list(itertools.product(np.linspace(0, 1, num=100), np.linspace(0, 1, num=100))))
    solution_true = np.rot90((np.sin(X[:, 0]) * np.sin(X[:, 1])).reshape(100, 100))
    solution_pred = np.rot90(net.predict(X, returnnp=True).reshape(100, 100))
    L2_error = np.sqrt(np.mean((solution_pred - solution_true) ** 2))
    print('L2_error:', L2_error)
    
    plt.figure(figsize=[6.4 * 2, 4.8])
    plt.subplot(121)
    plt.imshow(solution_true, cmap='rainbow')
    plt.title(r'Exact solution $\sin(x)\sin(y)$', fontsize=18)
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.subplot(122)
    plt.imshow(solution_pred, cmap='rainbow')
    plt.title('Prediction', fontsize=18)
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.savefig('pinn.pdf')

def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    train_num = {'diff': 100, 'x_0': 10, 'x_1': 10, '0_y': 10, '1_y': 10}
    test_num = {'diff': 900, 'x_0': 30, 'x_1': 30, '0_y': 30, '1_y': 30}
    # fnn
    size = [2, 20, 1]
    activation = 'sigmoid'
    # training
    lr = 0.001
    iterations = 10000
    print_every = 1000
    batch_size = None # {'diff': 100, 'x_0': 10, 'x_1': 10, '0_y': 10, '1_y': 10}
    
    data = PoissonData(train_num, test_num)
    fnn = ln.nn.FNN(size, activation)
    net = PoissonPINN(fnn)
    args = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device,
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    plot(data, ln.Brain.Best_model())
    
if __name__ == '__main__':
    main()