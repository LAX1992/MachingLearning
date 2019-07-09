import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import tree
'''
RUN python hw3_R07944050.py IN VSCODE
'''

class DecisionTree:
    def __init__(self, node, threshold, d, left, right):
        #放output y
        self.node = node
        self.threshold = threshold
        #紀錄分類依據第一維還第二維
        self.dimension = d
        self.left = left
        self.right = right

#Gini index
def Gini_index(y):
    N = len(y)
    if(N == 0):
        return 1
    t = (np.sum(y == -1)/ N)**2 + (np.sum(y == 1)/ N)**2

    return 1 - t

#impurity
def loss(threshold, data, d):
    index1 = (data[:, d] <= threshold)
    index2 = (data[:, d] > threshold)
    Gini1 = Gini_index(data[index1][:, 2])
    Gini2 = Gini_index(data[index2][:, 2])
    
    return Gini1 + Gini2

def branch(data):
    threshold = 0
    error = sys.maxsize
    dimemsion = 0
    
    dim_1 = data[:, 0]

    for i in dim_1:
        error1 = loss(i, data, 0)
        if error1 < error:
            error = error1
            threshold = i

    dim_2 = data[:, 1]
    
    for i in dim_2:
        error2 = loss(i, data, 1)
        if error2 < error:
            error = error2
            threshold = i
            dimemsion = 1
    
    return threshold, dimemsion

def isleaf(data):
    n = len(data)
    if n == 0:
        return True
    positive_num = np.sum(data[:, 2] == 1)
    negative_num = np.sum(data[:, 2] == -1)
    if positive_num == 0 or negative_num == 0:
        return True
    else:
        return False

def traininig(data):
    if isleaf(data):
        return DecisionTree(data[0][2], 0, 0, None, None)
    else:
        threshold, d = branch(data)
        tree = DecisionTree(None, threshold, d, None, None)
        #算左子右子
        leftdata = data[data[:, d] <= threshold]
        rightdata = data[data[:, d] > threshold]
        #遞迴下去
        leftTree = traininig(leftdata)
        rightTree = traininig(rightdata)
        tree.left = leftTree
        tree.right = rightTree
        return tree
    
def predict(tree, data):
    if tree.left == None and tree.right == None:
        return tree.node
    if data[tree.dimension] <= tree.threshold:
        return predict(tree.left, data)
    else:
        return predict(tree.right, data)
    
def error(DecisionTree, data):
    result = []
    for i in data:
        result.append(predict(DecisionTree, i))
    error = 1 - np.sum(result == data[:, 2]) / len(data)
    
    return error

def randomforest_error(forest, data, N):
    Error = np.array([])
    for i in range(N):
        E = []
        for j in range(1+i):
            E.append([predict(forest[j], k) for k in data])
        E = np.array(E)
        #加總為0等同沒分好給-2視為error
        ypred = np.sign(E.sum(axis = 0))
        np.where(ypred == 0, -2, ypred)
        error = 1 - np.sum(ypred == data[:, 2]) / len(data)
        Error = np.append(Error, error)
    return Error

if __name__ == '__main__':
    train_data = np.loadtxt('hw3_train.dat')
    test_data = np.loadtxt('hw3_test.dat')

    dtree = traininig(train_data)
    #12
    print("Ein:", error(dtree, train_data))

    print("Eout:", error(dtree, test_data))

    #14 跑30000次
    N = 30000
    Ein = []
    tree = []
    print("calculating Ein(gt)...")
    for i in range(N):
        index = np.random.randint(0, train_data.shape[0], size=round(train_data.shape[0]*0.8))
        traindata = train_data[index, :]
        dtree = traininig(traindata)
        Ein.append(error(dtree, train_data))
        tree.append(dtree)

    Ein = np.array(Ein)    
    plt.hist(Ein)
    plt.xlabel("Ein(gt)")
    plt.ylabel("tree")
    plt.show()
    #15
    print("calculating Ein(Gt)...")
    Ein_G = randomforest_error(tree, train_data, N)

    plt.plot(np.arange(1, N+1), Ein_G)
    plt.xlabel("t")
    plt.ylabel("Ein(Gt)")
    plt.show()
    
    #16
    print("calcaulating Eout(Gt)...")
    Eout_G = randomforest_error(tree, test_data, N)

    plt.plot(np.arange(1, N+1), Eout_G)
    plt.xlabel("t")
    plt.ylabel("Eout(Gt)")
    plt.show()
