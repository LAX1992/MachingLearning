import sys
import numpy as np
import scipy.linalg
from sklearn.utils import resample
from math import sqrt
import os
import random

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
    x, y = LoadData ('hw2_lssvm_all.dat')
    train_x = x[ : 400]
    test_x = x[400 : ]
    train_y = y[ : 400]
    test_y = y[400 : ]
    Lambda = np.array ([0.05, 0.5, 5, 50, 500])
    res=[]
    res.append(['lamda','Ein','Eout'])
    for lamda in Lambda:
        TotalTrain=[]
        TotalTest=[]
        for round in range(250):
            baggingX, baggingY = resample (train_x, train_y, random_state=random.randint (0, 400))
            w = np.dot(np.dot(np.linalg.inv(lamda*np.identity(baggingX.shape[1])+np.dot(np.transpose(baggingX),baggingX)), np.transpose(baggingX)),baggingY)
            trainResult=np.sign (np.dot (train_x,w.T))
            testResult=np.sign (np.dot (test_x,w.T))
            TotalTrain.append(trainResult)
            TotalTest.append(testResult)
        TotalTrain = np.array(TotalTrain)
        TotalTest = np.array(TotalTest)
        Ein=np.sum (np.sign (np.sum (TotalTrain, axis = 0 )) != train_y) / float (train_y.shape[0])
        Eout=np.sum (np.sign (np.sum (TotalTest, axis = 0 )) != test_y) / float (test_y.shape[0])
        res.append ([lamda, Ein, Eout])
    for item in res:
        print(item)