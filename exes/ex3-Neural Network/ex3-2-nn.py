from scipy.io import loadmat
import numpy as np
data = loadmat('ex3weights.mat')

theta1 = np.matrix(data['Theta1'])
theta2 = np.matrix(data['Theta2'])

print(theta1.shape, theta2.shape)

def sigmoid(x):
    return 1/(1+np.exp(-x))

original_data = loadmat('ex3data1.mat')
X = original_data['X']
X = np.matrix(X)
rows = X.shape[0]
X = np.insert(X, 0, values=np.ones(rows), axis=1)
print(X.shape)

# a2.shape: 5000*25
# a2 = list([0 for i in range(25)] for j in range(5000))
# a2 = np.matrix(a2)
# print(a2.shape)

z2 = X*theta1.T
a2 = sigmoid(z2)
print(a2.shape)
a2 = np.insert(a2, 0, values = np.ones(rows), axis=1)
z3 = a2*theta2.T
h = sigmoid(z3)
print(h.shape)

l1 = [[5,6,7],[3,2,3],[7,8,9]]
l2 = [1,2,2]
l1 = np.array(l1)
l2 = np.array(l2)
print(l1)
print(l2)
print(l2.T)
print(l1*l2.T)
print(l1.dot(l2.T))