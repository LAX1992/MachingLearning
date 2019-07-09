import numpy as np
import matplotlib.pyplot as plt

def knn(x, data, k):
    distance = np.sum((x[:-1] - data[:, :-1])**2, axis = 1)
    index_sorted = np.argsort(distance)
    label_sum = 0
    for i in range(k):
        label_sum += data[:, -1][index_sorted[i]]
    
    return np.sign(label_sum)

def error_in(dataset,k):
    error = 0
    for x in dataset:
        predict = knn(x, dataset, k)
        if predict != x[-1]:
            error += 1
    
    return error/dataset.shape[0]

def error_out(test_dataset, train_dataset, k):
    error = 0
    for x in test_dataset:
        predict = knn(x, train_dataset, k)
        if predict != x[-1]:
            error += 1
    
    return error/test_dataset.shape[0]

def Knn_uniform(x, data, r):
    distance = np.sum((x[:-1] - data[:, :-1])**2, axis = 1)
    exponent = np.exp(-r * distance)
    sub_sum = data[:, -1] * exponent
    predict_sum = np.sum(sub_sum)
    
    return np.sign(predict_sum)

def Error_in_uni(dataset, r):
    error = 0
    for x in dataset:
        predict = Knn_uniform(x, dataset, r)
        if predict != x[-1]:
            error += 1

    return error/dataset.shape[0]

def Error_out_uni(test_dataset, train_dataset, k):
    error = 0
    for x in test_dataset:
        predict = Knn_uniform(x, train_dataset, r)
        if predict != x[-1]:
            error += 1

    return error/test_dataset.shape[0]

def kMean(k, data):
    row, col = data.shape
    index = np.random.choice(range(row), size=k, replace=False )
    center = data[index]
    newcenter = np.copy(center)
    label = np.zeros(row)
    LOW_BOUND = 1e-5
    # 500次
    for times in range(500):
        center = np.copy(newcenter)
        for i in range(row):
            distance = np.sum((data[i] - center)**2, axis = 1)
            label[i] = np.argmin(distance)
        for i in range(k):
            # 同一個label的每個維度取平均當新的center
            newcenter[i] = np.mean(data[label == i], axis = 0)
    return center, label


# Q11
train_data = np.loadtxt("hw4_train.dat")
test_data = np.loadtxt("hw4_test.dat")
K = [1, 3, 5, 7, 9]
Ein = []
for k in K:
    Ein.append(error_in(train_data, k))

plt.scatter(K, Ein)
plt.xlabel("K")
plt.ylabel("Ein")
plt.show()
# Q12
Eout = []
for k in K:
    Eout.append(error_out(test_data, train_data, k))

plt.scatter(K, Eout)
plt.xlabel("K")
plt.ylabel("Eout")
plt.show()
# Q13
gamma = [0.001, 0.1, 1, 10, 100]
Ein_uniform = []
for r in gamma:
    Ein_uniform.append(Error_in_uni(train_data, r))
plt.scatter(gamma, Ein_uniform)
plt.xlabel("gamma")
plt.ylabel("Ein_uniform")
plt.show()
# Q14
Eout_uniform = []
for r in gamma:
    Eout_uniform.append(Error_out_uni(test_data, train_data, r))
plt.scatter(gamma, Eout_uniform)
plt.xlabel("gamma")
plt.ylabel("Eout_uniform")
plt.show()


k_dataset = np.loadtxt("hw4_nolabel_train.dat")
k_means = [2, 4, 6, 8, 10]
Ein_kmean = []
variance_kmean = []
for k in k_means:
    err = 0
    variance = []
    center, label = kMean(k, k_dataset)
    #算avgEin
    for i in range(k):
        variance.append(np.sum((k_dataset[label == i] - center[i])**2))
        err += np.sum((k_dataset[label == i] - center[i])**2)
    Ein_kmean.append(err/k_dataset.shape[0])
    #算varEin
    for j in range(len(variance)):
        var = (variance[j]-err/k_dataset.shape[0])**2/k_dataset.shape[0]
    variance_kmean.append(var)
# Q15
plt.scatter(k_means, Ein_kmean)
plt.xlabel("K")
plt.ylabel("Ein")
plt.show()
# Q16
plt.scatter(k_means, variance_kmean)
plt.xlabel("K")
plt.ylabel("variance")
plt.show()