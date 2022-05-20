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

def standardized():
    # 其实此时的分类并不是线性的，所以用上面的并不能良好拟合（结果正确率在0.5左右）
    # 因此考虑建立高次多项式
    data = pandas.read_csv("ex2data2.txt", header=None, names=["attr1", "attr2", "rst"])
    data.insert(0, "Ones", 1)
    x1 = data["attr1"]
    x2 = data["attr2"]
    print(x2.head())

    # 增加高次多项式
    col = 3
    degree = 5
    for i in range(1,degree):
        for j in range(0, i):
            data.insert(col, "F"+str(i)+str(j), np.power(x1, i)*np.power(x2, j))
            col += 1
            data.insert(col, "F"+str(j)+str(i), np.power(x1, j)*np.power(x2, i))
            col += 1

    data.drop('attr1', axis=1, inplace=True) # 删除原始数据
    data.drop('attr2', axis=1, inplace=True) # 删除原始数据
    print(data.head())

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    # 修改损失函数以及梯度下降，加上正则
    def costReg(theta, X, y, learning_rate):
        theta = np.matrix(theta)
        X = np.matrix(X)
        y = np.matrix(y)

        first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
        second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
        reg = learning_rate / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
        cost = np.sum(first + second) / len(X) + reg
        return cost

    # 修改梯度下降，加上正则
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
                middle = np.multiply(sigmoid(X * theta.T) - y, X[:, j])
                grad[i] = np.sum(middle)/X.shape[0] + learning_rate*theta[:,i]/len(X)

        return grad

    # set X and y (remember from above that we moved the label to column 0)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # convert to numpy arrays and initalize the parameter array theta
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.array(np.zeros(X.shape[1]))

    learning_rate = 0.1

    # 使用opt.fmin_tnc函数
    import scipy.optimize as opt
    result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, learning_rate))
    print(result)

    # 测试准确率
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
    theta = np.matrix(result[0])
    print(accu(X,y,theta))
    # 准确率达到0.61

def sk_learn():

    # 原始数据处理
    data = pandas.read_csv("ex2data2.txt", header=None, names=["attr1", "attr2", "rst"])
    x1 = data["attr1"]
    x2 = data["attr2"]
    col = 2
    degree = 5
    for i in range(1, degree):
        for j in range(0, i):
            data.insert(col, "F" + str(i) + str(j), np.power(x1, i) * np.power(x2, j))
            col += 1
            data.insert(col, "F" + str(j) + str(i), np.power(x1, j) * np.power(x2, i))
            col += 1

    data.drop('attr1', axis=1, inplace=True)  # 删除原始数据
    data.drop('attr2', axis=1, inplace=True)  # 删除原始数据

    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    X = np.matrix(X.values)
    y = np.matrix(y.values)

    from sklearn import linear_model  # 调用sklearn的线性回归包
    model = linear_model.LogisticRegression(penalty='l2', C=1.0)
    model.fit(X, y)

    print(model.coef_, model.intercept_)  # 权重矩阵 偏移量
    print(model.score(X, y))

    theta = list(model.intercept_)
    for i in range(len(model.coef_[0])):
        theta.append(model.coef_[0][i])
    theta = np.matrix(theta)

    data.insert(0, "ones", 1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    X = np.matrix(X.values)
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
    print(accu(X,y,theta))

    # accuracy与model.score一样

# standardized()
sk_learn()
