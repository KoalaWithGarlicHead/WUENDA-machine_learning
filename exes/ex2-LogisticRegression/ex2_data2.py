import pandas
import numpy as np

def my():
    data = pandas.read_csv("ex2data2.txt",header=None,names=["attr1", "attr2", "rst"])
    data.insert(0, "Ones", 1)

    cols = data.shape[1]
    X = np.matrix(data.iloc[:,0:cols-1].values)
    y = np.matrix(data.iloc[:,cols-1:cols].values)
    theta = np.matrix(np.zeros(cols-1))

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def compute_cost(X,y,theta, lam):
        first = np.multiply(y,np.log(sigmoid(X*theta.T)))
        second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
        cost = -np.sum(first+second)/len(X)+lam/(2*len(X))*np.sum(np.multiply(theta, theta))
        return cost

    def accu(X, y, theta):
        acc = 0
        terms = X * theta.T
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
            for j in range(para_cnt):
                if j==0:
                    middle = np.multiply(sigmoid(X*theta.T)-y, X[:,j])
                    temp[0,j] = theta[0,j]-alpha/X.shape[0]*np.sum(middle)
                else:
                    middle = np.multiply(sigmoid(X * theta.T) - y, X[:, j])
                    temp[0,j] = theta[0,j]-alpha/X.shape[0]*(np.sum(middle)+lam*theta[0,j])
            theta = temp
            cost[i] = compute_cost(X,y,theta,lam)
            if i%1000 == 0:
                print(i, theta, cost[i], accu(X,y,theta))
        return theta, cost[-1]

    g, s = gradient_descent(X,y,theta,0.01, 100000, 0.1)
    print(g)

my()
