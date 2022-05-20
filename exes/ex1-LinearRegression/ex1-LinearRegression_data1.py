import matplotlib.pyplot as plt
import numpy as np


def my():

    readfile = open("ex1data1.txt", "r")
    contents = readfile.read().splitlines()
    readfile.close()

    x_list = []
    y_list = []

    # 自己写的
    for everyline in contents:
        if len(everyline) > 1:
            x_ele = float(everyline.split(",")[0])
            y_ele = float(everyline.split(",")[1])
            x_list.append(x_ele)
            y_list.append(y_ele)

    m = len(x_list)
    theta0 = 0
    theta1 = 0

    def getJ(theta0, theta1):
        j_sum = 0
        for i in range(0, m):
            j_sum += (theta0+theta1*x_list[i]-y_list[i])**2
        return j_sum/(2*m)

    def getSum1(theta0, theta1):
        sum1 = 0
        for i in range(0, m):
            sum1 += theta0+theta1*x_list[i]-y_list[i]
        return sum1/m

    def getSum2(theta0, theta1):
        sum2 = 0
        for i in range(0, m):
            sum2 += (theta0+theta1*x_list[i]-y_list[i])*x_list[i]
        return sum2/m

    alpha = 0.01
    j = getJ(theta0, theta1)
    cnt = 0
    minJ = j
    minLst = [theta0, theta1]
    while cnt < 1000:
        cnt += 1
        temp0 = theta0-alpha*getSum1(theta0, theta1)
        temp1 = theta1-alpha*getSum2(theta0, theta1)
        theta0 = temp0
        theta1 = temp1
        j = getJ(theta0, theta1)
        if j < minJ:
            minJ = j
            minLst = [theta0, theta1]

    print(minLst)

    plt.scatter(x_list, y_list)

    x = np.arange(0, 30, 0.1)
    y = minLst[1]*x+minLst[0]
    plt.plot(x, y)
    plt.show()

# 标准写法
def standardized():
    import pandas as pd
    path = "ex1data1.txt"
    data = pd.read_csv(path, header=None, names=["Population", "Profit"])
    def computeCost(X, y, theta):
        inner = np.power(((X * theta.T)-y), 2)
        return sum(inner)/2*(len(X))

    # 在训练集中添加一列，用向量化的方法解决问题
    data.insert(0, "Ones", 1) # "Ones"为列名称

    cols = data.shape[1] # 列数
    X = data.iloc[:, 0:cols-1] # 去除y列数据 得到矩阵X :的含义是取所有行
    y = data.iloc[:, cols-1:cols] # y列数据
    # 转换为矩阵
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0,0]))

    def gradient_descent(X, y, theta, alpha, iter):
        temp = np.matrix(np.zeros(theta.shape))
        cost = np.zeros(iter)
        parameters = int(theta.ravel().shape[1])

        for i in range(iter):
            error = (X*theta.T)-y
            for j in range(parameters):
                term = np.multiply(error, X[:,j])
                temp[0,j] = theta[0,j]-(alpha/len(X))*np.sum(term)
            theta = temp
            cost[i] = computeCost(X,y,theta)

        return theta, cost

    alpha = 0.01
    iter = 1000
    g, cost = gradient_descent(X,y,theta, alpha, iter)
    print(g)

def sk_model():
    # 直接使用skicit-learn的模型
    import pandas as pd
    path = "ex1data1.txt"
    data = pd.read_csv(path, header=None, names=["Population", "Profit"])

    cols = data.shape[1]  # 列数
    X = data.iloc[:, 0:cols - 1]  # 去除y列数据 得到矩阵X :的含义是取所有行
    y = data.iloc[:, cols - 1:cols]  # y列数据
    # 转换为矩阵
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    from sklearn import linear_model
    model = linear_model.LinearRegression()
    model.fit(X, y)
    print(model.coef_, model.intercept_) #权重矩阵 偏移量
    print(model.score(X, y))

def normal_equation():
    import pandas as pd
    path = "ex1data1.txt"
    data = pd.read_csv(path, header=None, names=["Population", "Profit"])
    data.insert(0, "Ones", 1)  # "Ones"为列名称

    cols = data.shape[1]  # 列数
    X = data.iloc[:, 0:cols - 1]  # 去除y列数据 得到矩阵X :的含义是取所有行
    y = data.iloc[:, cols - 1:cols]  # y列数据
    # 转换为矩阵
    X = np.matrix(X.values)
    y = np.matrix(y.values)


    theta = (np.linalg.inv(X.T*X))*(X.T)*y
    print(theta)
    # 值有差距

def scipy_func():
    import pandas as pd
    path = "ex1data1.txt"
    data = pd.read_csv(path, header=None, names=["Population", "Profit"])

    def computeCost(theta, X, y):
        X = np.matrix(X)
        y = np.matrix(y)
        theta = np.matrix(theta)
        inner = np.power(((X * theta.T) - y), 2)
        return sum(inner) / 2 * (len(X))

    # 在训练集中添加一列，用向量化的方法解决问题
    data.insert(0, "Ones", 1)  # "Ones"为列名称

    cols = data.shape[1]  # 列数
    X = data.iloc[:, 0:cols - 1]  # 去除y列数据 得到矩阵X :的含义是取所有行
    y = data.iloc[:, cols - 1:cols]  # y列数据
    # 转换为矩阵
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.array([0, 0])

    def gradient_descent(theta, X, y):
        X = np.matrix(X)
        y = np.matrix(y)
        theta = np.matrix(theta)
        parameters = int(theta.ravel().shape[1])
        grad = np.zeros(parameters)

        error = (X*theta.T)-y
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            grad[j] = np.sum(term)/len(X)

        return grad

    import scipy.optimize as opt
    result = opt.fmin_tnc(func=computeCost, x0=theta, fprime=gradient_descent, args=(X, y))
    print(result)

if __name__=="__main__":
    # my()
    standardized()
    sk_model()
    # normal_equation()
    # scipy_func()