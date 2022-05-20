import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def my():
    # 数据集含义：前两项为学生分数，后面为录取结果
    path = "ex2data1.txt"
    data = pd.read_csv(path, header=None, names=["class1", "class2", "admit"])

    data.insert(0, "Ones", 1)
    cols = data.shape[1]
    X = np.matrix(data.iloc[:,0:cols-1].values)
    y = np.matrix(data.iloc[:,cols-1:cols].values)
    theta = np.matrix(np.zeros(cols-1))

    def compute_cost(X,y,theta):
        cost = 0
        terms = X * theta.T
        for i in range(X.shape[0]):
            # cost += y[i,0]*math.log(1/(1+math.exp(-terms[i,0])))+(1-y[i,0])*math.log(1-1/(1+math.exp(-terms[i,0])))
            cost += y[i,0]*math.log(1/(1+math.exp(-terms[i,0])))
            cost += (1-y[i,0])*math.log(math.exp(-terms[i,0])/(1+math.exp(-terms[i,0])))
        cost /= X.shape[0]
        cost = -cost
        return cost

    print(compute_cost(X,y,theta))

    def acc(X, theta, y):
        acc = 0
        terms = X * theta.T
        for i in range(X.shape[0]):
            pred_y = 1 / (1 + math.exp(-terms[i, 0]))
            if pred_y >= 0.5:
                if y[i, 0] == 1:
                    acc += 1
            else:
                if y[i, 0] == 0:
                    acc += 1
        return acc / X.shape[0]

    def gradient_descent(X, y, theta, alpha, iter):
        temp = np.matrix(np.zeros(theta.shape))
        cost = np.zeros(iter)
        accu = 0
        for i in range(iter):
            for j in range(X.shape[1]):
                terms = X*theta.T
                error = 0
                for k in range(X.shape[0]):
                    error += (1/(1+math.exp(-terms[k,0]))-y[k,0])*X[k,j]
                error /= X.shape[0]
                temp[0,j] = theta[0,j]-alpha*error
            theta = temp
            cost[i] = compute_cost(X,y,theta)
            accu = acc(X,theta,y)
            if i%1000==0:
                print(i, theta, cost[-1], accu)
        return theta, cost[-1], accu
    
    alpha = 0.01
    iter = 100000
    g,s,a = gradient_descent(X,y,theta,alpha,iter)
    print(g,s,a)



    def classify_and_draw(data, label, color):
        xlist = []
        ylist = []
        for i in range(data.shape[0]):
            if data.values[i][2] == label:
                xlist.append(data.values[i][0])
                ylist.append(data.values[i][1])
        plt.scatter(xlist, ylist, color=color)

    # classify_and_draw(data, 0, 'red')
    # classify_and_draw(data, 1, 'blue')
    # plt.show()

def standardized():
    path = "ex2data1.txt"
    data = pd.read_csv(path, header=None, names=["class1", "class2", "admit"])

    data.insert(0, "Ones", 1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    # convert to numpy arrays and initalize the parameter array theta
    # 这是因为下面会用到scipy的函数，输入需要是np.array
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(3)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def cost(theta, X, y): # 注意参数顺序
        X = np.matrix(X)
        y = np.matrix(y)
        theta = np.matrix(theta)

        print(X.shape, y.shape, theta.shape)
        first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
        second = np.multiply(1-y, np.log(1-sigmoid(X*theta.T)))
        return np.sum(first-second)/len(X)


    def gradient(theta, X, y): # 注意参数顺序
        # 注意，我们实际上没有在这个函数中执行梯度下降，我们仅仅在计算一个梯度步长。
        X = np.matrix(X)
        y = np.matrix(y)
        theta = np.matrix(theta)
        parameters = X.shape[1]
        grad = np.zeros(parameters)
        for i in range(parameters):
            term = np.multiply(sigmoid(X*theta.T)-y, X[:,i])
            grad[i] = np.sum(term)/X.shape[0]

        return grad

    # 现在可以用SciPy's truncated newton（TNC）实现寻找最优参数。
    import scipy.optimize as opt
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X,y))
    # 处理有约束的多元函数问题
    # func 优化的目标函数，x0：初值，fprime：梯度函数，args:传递给优化参数的函数
    print(result)

def sk_learn():
    path = "ex2data1.txt"
    data = pd.read_csv(path, header=None, names=["class1", "class2", "admit"])
    cols = data.shape[1]
    X = np.matrix(data.iloc[:, 0:cols - 1].values)
    y = np.matrix(data.iloc[:, cols - 1:cols].values)
    from sklearn import linear_model
    model = linear_model.LogisticRegression(penalty='l2', C=1.0)
    model.fit(X, y)

    print(model.coef_, model.intercept_)  # 权重矩阵 偏移量
    print(model.score(X, y))

# my()
# standardized()
sk_learn()