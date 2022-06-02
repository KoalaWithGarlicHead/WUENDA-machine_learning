from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = loadmat("ex5data1.mat")
X = np.array(data['X'])
y = np.array(data['y'])
Xtest = np.array((data['Xtest']))
ytest = np.array((data['ytest']))
Xval = np.array((data['Xval']))
yval = np.array((data['yval']))


def show_data():
    plt.scatter(X, y, marker='x')
    plt.scatter(Xtest, ytest, marker='*')
    plt.scatter(Xval, yval, marker='o')
    plt.show()
# show_data()

def compute_cost(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    middle = np.power(X*theta.T-y, 2)
    J = np.sum(middle)/(2*X.shape[0])

    right = np.power(theta[:,1:], 2)
    J += learning_rate/(2*X.shape[0])*np.sum(right)
    return J

X = np.matrix(X)
y = np.matrix(y)
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

Xval = np.matrix(Xval)
yval = np.matrix(yval)
Xval = np.insert(Xval, 0, np.ones(Xval.shape[0]), axis=1)

Xtest = np.matrix(Xtest)
ytest = np.matrix(ytest)
Xtest = np.insert(Xtest, 0, np.ones(Xtest.shape[0]), axis=1)

theta = np.matrix([1,1]) # shape: 1*2

def gradient(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    temp = np.zeros(theta.shape)
    for i in range(X.shape[1]):
        if i == 0:
            temp[0,i] = np.sum(np.multiply(X*theta.T-y, X[:, i]))/X.shape[0]
        else:
            temp[0,i] = np.sum(np.multiply(X*theta.T-y, X[:, i]))/X.shape[0]+learning_rate*theta[0, i]/X.shape[0]
    return temp

def linear_regression(X, y, theta, learning_rate, alpha, iter):
    # 自己写的
    for i in range(iter):
        theta = theta-alpha*gradient(theta, X, y, learning_rate)
    return theta


def linear_regression_sci(X, y, l):
    # 调包
    theta = np.matrix(np.ones(X.shape[1]))
    res = opt.minimize(fun=compute_cost, x0=theta, args=(X,y,l), method="TNC",jac=gradient, options={'disp': True})
    return np.matrix(res.get('x'))

# final_theta = linear_regression_sci(X, y, 0)
# print(final_theta)

def showLinearRegression(theta):
    x_draw = np.arange(-60, 50, 0.1)
    y_draw = theta[0,1]*x_draw+theta[0,0]
    plt.plot(x_draw, y_draw)
    plt.show()
# showLinearRegression(final_theta)

def choose_datasets_i(X, y, i):
    return X[:i, :], y[:i, :]

def compare_bias_and_variance(X, y, Xval, yval, learning_rate):
    train_cost_list = []
    cv_cost_list = []
    for i in range(1, X.shape[0]+1):
        X_train, y_train = choose_datasets_i(X, y, i)
        theta = linear_regression_sci(X_train, y_train, learning_rate)
        error_train = compute_cost(theta, X_train, y_train, learning_rate)
        train_cost_list.append(error_train)
        error_val = compute_cost(theta, Xval, yval, learning_rate)
        cv_cost_list.append(error_val)
        print(i, error_train, error_val)

    plt.plot(np.arange(1, X.shape[0]+1), train_cost_list,label='training cost')
    plt.plot(np.arange(1, X.shape[0]+1), cv_cost_list, label='cv cost')
    plt.legend(loc=1)
    plt.show()

# compare_bias_and_variance(X, y, Xval, yval, 0)

def make_polinomial(X, p):
    X_new = X
    X_unit = np.array(X[:, 1]).flatten()
    for i in range(2, p+1):
        X_add = np.power(X_unit, i)
        X_new = np.insert(X_new, i, X_add, axis=1)
    return X_new

X_poly_8 = make_polinomial(X, 8)
# print(X)

def normalize_feature(X):
    mu = np.mean(X[:, 1:], axis=0)
    sigma = np.std(X[:, 1:], axis=0)
    X[:, 1:] = (X[:, 1:]-mu)/sigma
    return X

X_poly_8_normal = normalize_feature(X_poly_8)
# print(X_poly_8_normal[:3,:])
Xval_poly_8_normal = normalize_feature(make_polinomial(Xval, 8))
Xtest_poly_8_normal = normalize_feature(make_polinomial(Xtest, 8))
# 测试lambda
# compare_bias_and_variance(X_poly_8_normal,y, Xval_poly_8_normal, yval, 0) # lambda = 0
# compare_bias_and_variance(X_poly_8_normal,y, Xval_poly_8_normal, yval, 1) # lambda = 1
# compare_bias_and_variance(X_poly_8_normal,y, Xval_poly_8_normal, yval, 100) # lambda = 100

def find_best_lambda(X, y, Xval, yval, Xtest, ytest):
    lamda_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    training_cost, cv_cost = [], []
    for l in lamda_list:
        theta = linear_regression_sci(X, y, l)
        error_train = compute_cost(theta, X, y, l)
        training_cost.append(error_train)
        error_val = compute_cost(theta, Xval, yval, l)
        cv_cost.append(error_val)
    plt.plot(lamda_list, training_cost, label='training')
    plt.plot(lamda_list, cv_cost, label='cross validation')
    plt.legend(loc=2)

    plt.xlabel('lambda')

    plt.ylabel('cost')
    plt.show()
    # cv数据集上的最佳lambda
    print(lamda_list[np.argmin(cv_cost)])

    # 测试数据集
    mes = ""
    for l in lamda_list:
        theta = linear_regression_sci(Xtest, ytest, l)
        mes += ("l={0}, test cost:{1:.2f}\n".format(l, compute_cost(theta, Xtest, ytest, l)))
    print(mes)

find_best_lambda(X_poly_8_normal, y, Xval_poly_8_normal, yval, Xtest_poly_8_normal, ytest)

