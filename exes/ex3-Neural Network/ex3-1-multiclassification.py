# matlab格式数据 需要首先引包导入
from scipy.io import loadmat
import numpy as np
data = loadmat('ex3data1.mat')
print(data)
print(data['X'].shape, data['y'].shape)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def costReg(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = learning_rate / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    cost = np.sum(first + second) / len(X) + reg
    return cost

def gradientReg(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    for i in range(parameters):
        if i == 0:
            middle = np.multiply(sigmoid(X * theta.T) - y, X[:, i])
            grad[i] = np.sum(middle)/X.shape[0]
        else:
            middle = np.multiply(sigmoid(X * theta.T) - y, X[:, i])
            grad[i] = np.sum(middle)/X.shape[0] + learning_rate*theta[:,i]/len(X)

    return grad

def do_classification(data, num):
    X = data['X']
    rows = X.shape[0]
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # 在矩阵X上，index为0的位置，加入[1,1,1,...,1](共rows个)，轴为1
    old_y = data['y']
    y = [[0] for i in range(rows)]
    for i in range(rows):
        if old_y[i] == num:
            y[i][0] = 1
    y = np.array(y)
    params = X.shape[1]
    theta = np.array(np.zeros(params))

    # learning_rate = 0.1
    #
    # # 使用opt.fmin_tnc函数
    # import scipy.optimize as opt
    # result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, learning_rate))
    # print(result)


    # self-written...
    def compute_cost(X,y,theta, lam):

        first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
        second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
        # cost = -np.sum(first+second)/len(X)+lam/(2*len(X))*np.sum(np.multiply(theta, theta))
        # 这样的写法有问题 因为最后的正则项的theta是从j=1开始计算的
        reg = lam/(2*len(X))*np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
        cost = np.sum(first+second)/len(X)+reg
        return cost

    def accu(X, y, theta):

        acc = 0
        terms = X * theta.T
        print(y.shape)
        for i in range(X.shape[0]):
            pred_y = 1 / (1 + np.exp(-terms[i, 0]))
            if pred_y >= 0.5:
                if y[i, 0] == 1:
                    acc += 1
            else:
                if y[i, 0] == 0:
                    acc += 1
        return acc / X.shape[0]

    def gradient_descent(X,y,theta, alpha, iter, lam):


        temp = np.matrix(np.zeros(X.shape[1]))
        para_cnt = X.shape[1]
        cost = np.zeros(iter)
        for i in range(iter):
            print(i)
            for j in range(para_cnt):
                if j==0:
                    middle = np.multiply(sigmoid(X*theta.T)-y, X[:,j])
                    temp[0,j] = theta[0,j]-alpha/X.shape[0]*np.sum(middle)
                else:
                    middle = np.multiply(sigmoid(X * theta.T) - y, X[:, j])
                    temp[0,j] = theta[0,j]-alpha/X.shape[0]*(np.sum(middle)+lam*theta[0,j])
            theta = temp
            cost[i] = compute_cost(X,y,theta,lam)

        print(i, theta, cost[i], accu(X, y, theta))
        return theta, cost[-1]

    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    print(X.shape, y.shape, theta.shape)

    g, s = gradient_descent(X,y,theta,0.01, 100, 1)
    print(g)

do_classification(data, 0)