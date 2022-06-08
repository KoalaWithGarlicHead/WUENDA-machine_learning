import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

# 查看第一个数据集 ex6data1.mat
data1_mat = loadmat("data/ex6data1.mat")
# 将其用散点图表示，其中类标签由符号表示（+表示正类，o表示负类）。
data1 = pd.DataFrame(data1_mat['X'], columns=['X1', 'X2'])
data1['y'] = data1_mat['y']
positive = data1[data1['y'].isin([1])]
negative = data1[data1['y'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
# ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
# ax.legend()
# plt.show()
# 看到有一个离散点(0.1, 4.1)

# 使用SVM进行训练
from sklearn import svm
# 选用线性支持向量机

# 首先c=1
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(data1[['X1', 'X2']], data1['y'])
print(svc.score(data1[['X1', 'X2']], data1['y']))

# c = 100
svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(data1[['X1', 'X2']], data1['y'])
svc2.score(data1[['X1', 'X2']], data1['y'])
print(svc2.score(data1[['X1', 'X2']], data1['y']))

# 通过查看每个类别预测的置信水平， 看出C的增大是数据决策边界不再合适
data1['SVM 1 Confidence'] = svc.decision_function(data1[['X1', 'X2']])
data1['SVM 2 Confidence'] = svc2.decision_function(data1[['X1', 'X2']])
# ax.scatter(data1['X1'], data1['X2'], s=50, c=data1['SVM 1 Confidence'], cmap='seismic')
ax.scatter(data1['X1'], data1['X2'], s=50, c=data1['SVM 2 Confidence'], cmap='seismic')
ax.set_title('SVM (C=1) Decision Confidence')
plt.show()

# C = 1时不会感知到那个outlier
# C = 100会竭力区分outlier

