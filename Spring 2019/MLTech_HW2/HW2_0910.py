import numpy as np
from scipy.linalg import inv
def LoadData(filename):
    x = []
    y = []
    dataset = open(filename)
    for line in dataset.readlines():
        data = line.split()
        number = np.array([float(num) for num in data])
        x.append([1]+number[:-1])
        y.append(number[-1])
    x = np.array(x)
    y = np.array(y)
    return x, y


if __name__ == '__main__':
    x, y = LoadData('hw2_lssvm_all.dat')
    train_x = x[ : 400]
    test_x = x[400 : ]
    train_y = y[ : 400]
    test_y = y[400 : ]
    Lambda = np.array([0.05, 0.5, 5, 50, 500])
    result = []
    result.append(['Lambda', 'Ein', 'Eout'])
    for value in Lambda:
        #206 p.5公式
        w = np.dot(np.dot(inv(value*np.identity(train_x.shape[1]) + np.dot(train_x.T,train_x)), train_x.T),train_y)
        Ein=np.sum(np.sign (np.dot (train_x,w.T)) != train_y)/float (train_y.shape[0])
        Eout=np.sum(np.sign (np.dot (test_x,w.T)) != test_y)/float(test_y.shape[0])       
        result.append([value, Ein, Eout])
    for res in result:
        print(res)