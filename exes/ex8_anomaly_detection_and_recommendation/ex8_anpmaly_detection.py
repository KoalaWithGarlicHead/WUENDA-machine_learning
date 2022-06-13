from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

data = loadmat('data/ex8data1.mat')
X = data['X'] # 307*2
Xval = data['Xval'] # 307*2
yval = data['yval'] # 307*1

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(X[:,0], X[:,1])
# plt.show()

def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma

mu, sigma = estimate_gaussian(X)

# SciPy内置的计算数据点属于正态分布的概率的方法。
from scipy import stats
# dist = stats.norm(mu[0], sigma[0])
# print(dist.pdf(X[:, 0]))
p = np.zeros((X.shape[0], X.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:, 0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:, 1])

mu_val, sigma_val = estimate_gaussian(Xval)
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:, 0] = stats.norm(mu_val[0], sigma_val[0]).pdf(Xval[:, 0])
pval[:, 1] = stats.norm(mu_val[1], sigma_val[1]).pdf(Xval[:, 1])


def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0

    pval = np.array(np.prod(pval, axis=1))

    step = (np.max(pval) - np.min(pval)) / 1000

    for epsilon in np.arange(np.min(pval)+step, np.max(pval), step):
        tp = 0
        fp = 0
        fn = 0

        for i in range(pval.shape[0]):
            if pval[i] < epsilon:
                if yval[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if yval[i] == 1:
                    fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

best_epsilon, f1 = select_threshold(pval, yval)
print(best_epsilon, f1)
pval = np.array(np.prod(pval, axis=1))
outliers = np.where(p < best_epsilon)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')
plt.show()
