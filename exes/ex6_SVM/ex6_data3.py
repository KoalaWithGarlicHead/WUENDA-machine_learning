import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

data_mat = loadmat("data/ex6data3.mat")
data = pd.DataFrame(data_mat['X'], columns=['X1','X2'])
data['y'] = data_mat['y']
dataval = pd.DataFrame(data_mat['Xval'], columns=['Xval1', 'Xval2'])
dataval['yval'] = data_mat['yval']
# positive = data[data['y'].isin([1])]
# negative = data[data['y'].isin([0])]
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
# ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
# ax.legend()
# plt.show()

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

from sklearn import svm
best_score = 0
best_params = {'C': None, "gamma":None}
for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(data[['X1', 'X2']], data['y'])
        score = svc.score(dataval[['Xval1', 'Xval2']], dataval['yval'])
        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print(best_score, best_params)
