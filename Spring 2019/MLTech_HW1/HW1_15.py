import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
dataset = np.loadtxt('features_train.dat')
X_train = dataset[:,1:]
Y_train = dataset[:,0]
train_answer = np.where(Y_train==0, 1 ,-1)
C = [-2, -1, 0, 1, 2]
Distance = []
for i in C:
    c = 10**i
    clf = svm.SVC(kernel = "rbf", gamma = 80, C = c)
    clf.fit(X_train,train_answer)
    w = clf.dual_coef_[0].dot(clf.support_vectors_)
    distance = 1/np.sum(w*w)
    Distance.append(distance)

plt.plot(C,Distance)
plt.xlabel('logC')
plt.ylabel('Distance')
plt.show()
