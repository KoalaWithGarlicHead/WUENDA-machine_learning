import numpy as np
from scipy.io import loadmat
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def findClosestCentroid(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_distance = np.sum((X[i, :]-centroids[0, :])**2)
        idx[i] = 0
        for j in range(1, k):
            this_min = np.sum((X[i, :]-centroids[j, :])**2)
            if this_min < min_distance:
                min_distance = this_min
                idx[i] = j

    return idx

data = loadmat("data/ex7data2.mat")
X = data['X']

# data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
# sb.set(context="notebook", style="white")
# sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
# plt.show()

def computeCentroids(X, idx, k):
    n = X.shape[1]
    centroids = np.zeros((k,n))

    for i in range(k):
        indices = np.where(idx == i)
        for j in range(n):
            centroids[i, j] = np.sum(X[indices, j], axis=1)/len(indices[0])

    return centroids

def run_k_means(X, initial_centroids, max_iters):
    centroids = initial_centroids
    for i in range(max_iters):
        idx = findClosestCentroid(X, centroids)
        centroids = computeCentroids(X, idx, initial_centroids.shape[0])
    return idx, centroids

# initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
# idx, centroids = run_k_means(X, initial_centroids, 10)

# cluster1 = X[np.where(idx == 0)[0],:]
# cluster2 = X[np.where(idx == 1)[0],:]
# cluster3 = X[np.where(idx == 2)[0],:]

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
# ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
# ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
# ax.legend()
# plt.show()

def init_centroids(X,k):
    centroids = np.zeros((k,X.shape[1]))
    idx = np.random.randint(0, X.shape[0], k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    return centroids

image_data = loadmat("data/bird_small.mat")
A = image_data['A']
# print(A.shape)

# # 数据预处理
# # 为什么要转化成浮点数
# # cast to float, you need to do this otherwise the color would be weird after clustring
# A = A/255
# # reshape the array
# print(A)
# X_a = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
# print(X_a)
# initial_centroids = init_centroids(X_a, 16)
# idx, centroids = run_k_means(X_a, initial_centroids, 10)
#
# # map each pixel to the centroid value
# X_recovered = centroids[idx.astype(int),:]
# # reshape to the original dimensions
# X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
# plt.imshow(X_recovered)
# # (M, N, 3)
# # RGB三通道图像，元素值可以是0−1之间的float或者0−255之间的int
# plt.show()


# 使用scikit-learn来实现k-means
from skimage import io
pic = io.imread('data/bird_small.png') / 255. # pic.shape: 128*128*3
io.imshow(pic)
# plt.show()

# serialize data
data_pic = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2]) # shape: (16384, 3)

from sklearn.cluster import KMeans#导入kmeans库

model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
model.fit(data_pic)
centroids = model.cluster_centers_# 中心点： 16*3
C = model.predict(data_pic) # 预测结果 shape: (16384,)
print(centroids[C].shape) # (16384, 3)

compressed_pic = centroids[C].reshape((128,128,3))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()