import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

data_mat = loadmat("data/ex6data2.mat")
data = pd.DataFrame(data_mat['X'], columns=['X1','X2'])
data['y'] = data_mat['y']
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
# ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
# ax.legend()
# plt.show()

# 实现高斯函数
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.sum(np.power(x1-x2, 2))/(2*np.power(sigma, 2)))

from sklearn import svm
svc = svm.SVC(C=100, gamma=30, probability=True)

svc.fit(data[['X1', 'X2']], data['y'])
print(svc.score(data[['X1', 'X2']], data['y']))
data['SVM Confidence'] = svc.decision_function(data[['X1', 'X2']])
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM Confidence'], cmap='seismic')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()