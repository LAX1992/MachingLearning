import numpy as np
import matplotlib.pyplot as plt
#variable已盡量跟老師的公式一樣
def sigmoid(s):
    return 1.0 / (1 + np.exp(-s))

def theta(y,w,X):
    temp=-y*w.dot(X)
    return (-y*X)/(1 + np.exp(-temp))

def stochastic_gradient_descent(X ,y, Eta, T):    
    N, d = X.shape    
    w = np.zeros(d)
    Ein = []
    Eout = []
    for i in range(T):
        n = i % N
        Xn = X[n]
        yn = y[n]
        w = w + Eta*sigmoid(-yn*Xn.dot(w))*(yn*Xn)
        Ein.append(error(X, y, w))
        Eout.append(error(X_test, y_test, w))
    return Ein, Eout, w

def gradient_descent(X ,y, Eta, T):    
    N, d = X.shape    
    w = np.zeros(d)
    Ein = []
    Eout = []
    for i in range(T):
        s = np.zeros(d)
        #算平均
        for j in range(N):
            s += theta(y[j],w,X[j][:])
        delta = s / N
        w = w - Eta * delta
        Ein.append(error(X, y, w))
        Eout.append(error(X_test, y_test, w))
    return Ein, Eout, w

def error(X, y, w):
    #dimension用不到單純接收回傳值
    N, dimension = X.shape
    return np.sum(np.sign(X.dot(w))!=y) / N
    
data = np.loadtxt('hw3_train.dat')
X = data[:,:-1]
y = data[:,-1]
data2 = np.loadtxt('hw3_test.dat')
X_test = data2[:,:-1]
y_test = data2[:,-1]

gd_Ein, gd_Eout, w2 = gradient_descent(X, y, 0.01, 2000)
sgd_Ein, sgd_Eout,w3 = stochastic_gradient_descent(X, y, 0.001, 2000)

plt.title('Ein comparison')
plt.plot(gd_Ein, color='red', label='gradient_descent')
plt.plot(sgd_Ein, color='blue', label='stochastic_gradient_descent')
plt.legend()
plt.xlabel("t",fontsize=14)
plt.ylabel("Ein",fontsize=14)
plt.show()

plt.title('Eout comparison')
plt.plot(gd_Eout, color='red', label='gradient_descent')
plt.plot(sgd_Eout, color='blue', label='stochastic_gradient_descent')
plt.legend()
plt.xlabel("t",fontsize=14)
plt.ylabel("Eout",fontsize=14)
plt.show()

print( w2,'\ngradient_descent_Eout:',error(X_test, y_test, w2))
print( w3, '\nstochastic_gradient_descent_Eout:',error(X_test, y_test, w3))
