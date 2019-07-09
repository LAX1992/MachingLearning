import numpy as np
import matplotlib.pyplot as plt

train_set = np.loadtxt("hw1_7_train.dat")

def sign(v):
    if v>0:
        return 1
    else :
        return -1

def random_pla():
    t = 0 #更新次数
    correct_num = 0 #the right number of prediction
    w = np.zeros([5,])
    
    while correct_num<row:
        for n in range(0,row,1):
            x = X[n,]
            y = Y[n]

            if sign(np.dot(x, w)) != y:
                w = w + y*x
                t = t + 1
                break
            else:
                correct_num = correct_num +1
    return t

#統計個別次數
statistics = []
#次數總和
total_times = 0
#主程式
for i in range(0,1126):
    #shuffle the dataset
    np.random.shuffle(train_set)
    row, col = train_set.shape
    X = np.zeros((row, 5)) 
    Y = np.zeros((row, 1))
    for j in range(row):
        X[j, 0] = 1.0
        X[j, 1] = np.float(train_set[j, 0])
        X[j, 2] = np.float(train_set[j, 1])
        X[j, 3] = np.float(train_set[j, 2])
        X[j, 4] = np.float(train_set[j, 3])
        Y[j, 0] = np.int(train_set[j, 4])
    times = random_pla()
    statistics.append(times)
    total_times = total_times + times
#算平均
averge_times = total_times / 1126
print(averge_times)
#畫圖
plt.xlabel('Number of updates')
plt.ylabel('frequency of the number')
plt.hist(statistics)
plt.show()


