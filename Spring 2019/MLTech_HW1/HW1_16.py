import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import random
dataset = np.loadtxt('features_train.dat')
X_train = dataset[:,1:]
Y_train = dataset[:,0]
train_answer = np.where(Y_train==0, 1 ,-1)
#前面的code用前幾題的所以我這題再把y接回每一列最後面
Data = np.concatenate((X_train,train_answer.reshape(-1,1)),axis=1)
#需要randomize的數字範圍 = 總列數
datasize = len(train_answer)
count = np.zeros(5)
Gamma = [-2, -1, 0, 1, 2]
for j in range(100):
    #前兩行為randomize code
    randomize_array = np.arange(datasize)
    np.random.shuffle(randomize_array)
    #random後的前1000列為validation
    X_crossval_0 = Data[randomize_array[:1000],:2]
    Y_crossval_0 = Data[randomize_array[:1000],2]
    X_crosstrain_0 = Data[randomize_array[1000:],:2]
    Y_crosstrain_0 = Data[randomize_array[1000:],2]
    Eval = np.array([])
    for i, value in enumerate(Gamma):
        gamma = 10**Gamma[i]
        clf = svm.SVC(kernel = 'rbf',gamma = gamma,C = 0.1)
        clf.fit(X_crosstrain_0,Y_crosstrain_0)
        e = np.sum(clf.predict(X_crossval_0) != Y_crossval_0)/len(X_crossval_0)
        Eval = np.append(Eval,e)
    index = np.argmin(Eval)
    count[index] += 1
plt.bar(Gamma,count)
plt.xlabel('log of gamma')
plt.ylabel('number of time')
plt.show()