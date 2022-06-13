from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

data = loadmat("data/ex7data1.mat")

X = data['X']

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(X[:, 0], X[:, 1])
# plt.show()

def pca(X):
    # normalize the data
    X = (X-X.mean())/X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    sigma = (X.T*X)/X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(sigma)
    return U,S,V

def project_data(X, U, k):
    U_reduce = U[:, :k]
    return np.matrix(X)*U_reduce

U, S, V = pca(X)
z = project_data(X, U, 1)

def recover_data(z, U, k):
    U_reduce = U[:, :k]
    return np.dot(z, U_reduce.T)

new_X = recover_data(z, U, 1)
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(list(new_X[:, 0]), list(new_X[:, 1]))
# plt.show()

faces = loadmat("data/ex7faces.mat")
X = faces['X'] # 5000*1024 5000张照片 每张是32*32大小
print(X.shape)

# face = np.reshape(X[3,:], (32, 32))
# plt.imshow(face)
# plt.show()

U, S, V = pca(X)
Z = project_data(X, U, 100)
X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[3,:], (32, 32))
plt.imshow(face)
plt.show()