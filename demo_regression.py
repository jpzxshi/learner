"""
@author: jpzxshi
"""
import numpy as np
import matplotlib.pyplot as plt
import learner as ln

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
    size = [1, 30, 30, 1]
    activation = 'relu'
    # training
    lr = 0.01
    iterations = 20000
    print_every = 1000
    
    X_train = (2 * np.pi * np.random.rand(30))[:, None]
    y_train = np.sin(X_train)
    X_test = np.linspace(0, 2 * np.pi, num=100)[:, None]
    y_test = np.sin(X_test)
    
    data = ln.Data(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    net = ln.nn.FNN(size, activation)
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