import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
dataset = np.loadtxt('features_train.dat')
X_train = dataset[:,1:]
Y_train = dataset[:,0]
train_answer = np.where(Y_train == 4, 1 ,-1)
C = [-5, -3, -1, 1, 3]
Ein = []
for i in C:
    c = 10**i
    clf = svm.SVC(kernel = "poly",degree = 2, coef0 = 1, gamma = 1, C = c)
    clf.fit(X_train,train_answer)
    #0*1 error(%)
    e = np.sum(clf.predict(X_train) != train_answer)/len(X_train)
    Ein.append(e)
plt.plot(C,Ein)
plt.xlabel('logC')
plt.ylabel('Ein(%)')
plt.show()
