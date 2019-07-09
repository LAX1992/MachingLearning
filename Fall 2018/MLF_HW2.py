import numpy as np
import matplotlib.pyplot as plt
import random

def reality(data_without_noise):
    data_with_noise = np.array([0.0]*20)
    for i in range (len(data_without_noise)-2):
        x = random.random()
        if x < 0.2 : 
            data_with_noise[i] = np.sign(data_without_noise[i+1])*(-1)
        else: 
            data_with_noise[i] = np.sign(data_without_noise[i+1])
    return data_with_noise

Ein = np.array([0.0]*1000)
Eout = np.array([0.0]*1000)
for iteration in range(1000):
    data = np.random.uniform(-1, 1, 20)
    data = np.append(data, [-2, 2]) #左右兩端各加入正負2 為了要能找到數線左右兩端超出1那兩段當threshold
    data = np.sort(data) #排序完比較好挑掉-2和+2
    dataset = reality(data) #20%的noise
    best_theta = 0
    best_s = 0
    min_err = 25
    for i in range(21):
        theta = (data[i]+data[i+1]) / 2.0
        hy = np.array([0] * 20)
        for j in range(20):
            hy[j] = np.sign(data[j] - theta)
        tmp_err = 0
        for k in range(20):
            if  hy[k] != dataset[k]:
                tmp_err += 1
        if tmp_err < min_err:
            min_err = tmp_err
            best_theta = theta
            best_s = 1
        tmp_err = 0
        for k in range(20):
            if (-1)*hy[k] != dataset[k]:
                tmp_err += 1
        if tmp_err < min_err:
            min_err = tmp_err
            best_theta = theta
            best_s = -1
    Ein[iteration] = min_err / 20.0
    Eout[iteration] = 0.5 + 0.3 * best_s * (abs(best_theta) - 1)

statistics = np.array([0.0]*1000) #統計圖表
for i in range(1000):
    statistics[i] = Ein[i] - Eout[i]
plt.xlabel('Ein - Eout')
plt.ylabel('Frequency')
plt.hist(statistics,edgecolor = 'k')
plt.show()

ave_Ein = round(np.mean(Ein),5)
ave_Eout = round(np.mean(Eout),5)
print('Average Ein:',ave_Ein)
print('Average Eout:',ave_Eout)

