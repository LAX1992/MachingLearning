import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
dataset = np.loadtxt('features_train.dat')
X_train = dataset[:,1:]
Y_train = dataset[:,0]
#2的話分類結果為1(正確)反之-1
train_answer = np.where(Y_train==2, 1 ,-1)
C = [-5, -3, -1, 1, 3]
W = []
for i in C:
    c = 10**i
    clf = svm.SVC(kernel = "linear",C = c)
    clf.fit(X_train,train_answer)
    w = clf.coef_[0]
    #2-norm
    W = np.append(W,np.sqrt(np.sum(w*w)))
plt.plot(C,W)
plt.xlabel('logC')
plt.ylabel('||W||')
plt.show()
