import numpy as np
import matplotlib.pyplot as plt
import math

def caulate_thres(dataset):
    size = dataset.shape[0]
    #分別對第1,2組feature排序
    train = np.array(sorted(dataset, key=lambda x:x[0]))
    train2 = np.array(sorted(train, key=lambda x:x[1]))
    thetas1 = []
    thetas2 = []
    x1 = train[:, 0]
    thetas1.append(x1[0] - 1)
    for i in range(size - 1):
        thetas1.append((x1[i] + x1[i+1]) / 2)
    thetas1.append(x1[-1] + 1)
    x2 = train2[:, 1]
    thetas2.append(x2[0] - 1)
    for i in range(size - 1):
        thetas2.append((x2[i] + x2[i+1]) / 2)
    thetas2.append(x2[-1] + 1)
    thetas1 = np.array(thetas1)
    thetas2 = np.array(thetas2)
    threshold = [thetas1, thetas2]
    return threshold

def decision_stump(X, Y, U, threshold):
    n = X.shape[0]
    x1 = X[:, 0]
    x2 = X[:, 1]
    d = 0
    index = 0
    best_Ein = 1
    s = 1

    for i in range(n+1):
        #第一類feature的Ein
        error1_1 = (np.sign(x1 - threshold[0][i]) != Y).dot(U)
        error1_2 = (np.sign(x1 - threshold[0][i]) != -Y).dot(U)
        if(error1_1 < best_Ein):
            d = 0
            index = i
            best_Ein = error1_1
            s = 1
        if(error1_2 < best_Ein):
            d = 0
            index = i
            best_Ein = error1_2
            s = -1
        #第二類feature的Ein
        error2_1 = (np.sign(x2 - threshold[1][i]) != Y).dot(U)
        error2_2 = (np.sign(x2 - threshold[1][i]) != -Y).dot(U)
        if(error2_1 < best_Ein):
            d = 1
            index = i
            best_Ein = error2_1
            s = 1
        if(error2_2 < best_Ein):
            d = 1
            index = i
            best_Ein = error2_2
            s = -1
    return best_Ein, s, d, index

def Adaptive_Boosting( X, Y, threshold, T):
    row = X.shape[0]
    u = np.ones(row) / row
    Alpha = np.array([])
    U = np.array([])
    Ein = np.array([])
    parameter_set = np.array([])

    x1 = X[:, 0]
    x2 = X[:, 1]
    x = [x1, x2]

    for t in range(T):
        best_ein, s, d, index = decision_stump( X, Y, u, threshold)
        epsilon = u.dot((s*np.sign(x[d] - threshold[d][index])) != Y) / np.sum(u)
        k = math.sqrt((1 - epsilon) / epsilon)
        #更新權重
        u = np.where(s*np.sign(x[d] - threshold[d][index]) != Y, u * k, u / k)
        alpha =  math.log(k)
        #存本次結果
        Ein = np.append(Ein, best_ein)
        if(t == 0):
            U = np.array([u])
        else:
            U = np.concatenate( (U, np.array([u])))
        Alpha = np.append(Alpha, alpha)
        #比起存整個陣列, 保存index資訊相對省空間
        para = [[s,d,index]]
        if(t == 0):
            parameter_set = np.array(para)
        else:
            parameter_set = np.concatenate( (parameter_set, np.array(para)))
    return Ein, U, Alpha, parameter_set

def predeict(X, Y, parameter_set, Alpha, i, threshold):
    x1 = X[:, 0]
    x2 = X[:, 1]
    x = [x1, x2]
    
    s = parameter_set[:i, 0]
    d = parameter_set[:i, 1]
    thresh = parameter_set[:i, 2]
    alpha = Alpha[:i]

    result = []
    for j in range(i):
        s1 = s[j]
        d1 = d[j]
        t1 = thresh[j]
        result.append(s1*np.sign(x[d1] - threshold[d1][t1]))
    result = np.matmul(alpha, np.array(result))
    return np.sum(np.sign(result) != Y) / len(Y)



if __name__ == '__main__':
    #Q13
    train_data = np.loadtxt("hw2_adaboost_train.dat")
    threshold = caulate_thres(train_data)
    train_X = train_data[:, 0: -1]
    train_Y = train_data[:, 2]
    Ein, U, Alpha, parameter_set = Adaptive_Boosting(train_X, train_Y, threshold, 300)
    print("Ein(gT) = %.6f" %Ein[-1])
    t = np.arange(300)
    plt.plot(t, Ein)
    plt.xlabel('t')
    plt.ylabel("Ein(gt)")
    plt.show()
    #Q14
    Ein_Gt = []
    for i in t:
        Ein_Gt.append( predeict(train_X, train_Y, parameter_set, Alpha, i, threshold))
    print("Ein(GT) =", Ein_Gt[-1])
    plt.plot(t, Ein_Gt)
    plt.xlabel('t')
    plt.ylabel("Ein(Gt)")
    plt.show()
    #Q15
    Ut = U.sum(axis = 1)
    print("U(T) = %.6f" % Ut[-1])
    plt.plot(t,Ut)
    plt.xlabel('t')
    plt.ylabel("Ut")
    plt.show()
    #Q16
    test_data = np.loadtxt("hw2_adaboost_test.dat")
    test_X = test_data[:, 0: -1]
    test_Y = test_data[:, 2]
    s = parameter_set[:, 0]
    d = parameter_set[:, 1]
    Eout_Gt = []
    for i in t:
        Eout_Gt.append( predeict(test_X, test_Y, parameter_set, Alpha, i, threshold))
    print("Eout(GT) =", Eout_Gt[-1])
    plt.plot(t, Eout_Gt)
    plt.xlabel('t')
    plt.ylabel("Eout(Gt)")
    plt.show()