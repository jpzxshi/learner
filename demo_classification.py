"""
@author: jpzxshi
"""
import numpy as np
import matplotlib.pyplot as plt
import learner as ln
from learner.utils import softmax

class CSData(ln.Data):
    def __init__(self):
        super(CSData, self).__init__()
        self.__init_data()
    
    def __init_data(self):
        tag = lambda x: [1, 0] if x < 0.5 else [0, 1]
        self.X_train = np.random.rand(10)[:, None]
        self.y_train = np.array(list(map(tag, self.X_train.squeeze())))
        self.X_test = np.linspace(0, 1, num=200)[:, None]
        self.y_test = np.array(list(map(tag, self.X_test.squeeze())))
        
def plot(data, net):
    x = data.X_test_np
    f_true = data.y_test_np[:, 0]
    f_pred = softmax(net.predict(data.X_test, returnnp=True))[:, 0]
    f_pred_label = f_pred > 0.5
    
    plt.plot(x, f_true, color='b', label='Ground truth', zorder=0)
    plt.plot(x, f_pred, color='r', label='Predicted prob.', zorder=1)
    plt.plot(x, f_pred_label, color='r', linestyle='--', label='Predicted label', zorder=2)
    plt.scatter(data.X_train_np, data.y_train_np[:, 0], color='b', label='Learned data', zorder=3)
    plt.legend()
    plt.savefig('classification.pdf')

def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # FNN
    size = [1, 30, 30, 2]
    activation = 'relu'
    # training
    lr = 0.01
    iterations = 500
    print_every = 100
    
    data = CSData()
    net = ln.nn.FNN(size, activation)
    args = {
        'data': data,
        'net': net,
        'criterion': 'CrossEntropy',
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