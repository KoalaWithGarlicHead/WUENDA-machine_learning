from scipy.io import loadmat
import numpy as np
data = loadmat('ex4data1.mat')
X = np.matrix(data['X']) # shape: 5000*400
y = np.matrix(data['y']) # shape: 5000*1
# 把y变成5000*10的独热编码的形式
y_onehot = list([0 for i in range(10)] for j in range(y.shape[0]))
for i in range(y.shape[0]):
    y_onehot[i][y[i,0]-1] = 1
y_onehot = np.matrix(y_onehot) # 5000*10

# one-hot 编码将类标签n（k类）转换为长度为k的向量，其中索引n为“hot”（1），而其余为0。 Scikitlearn有一个内置的实用程序，我们可以使用这个。
def generate_y_onehot(y):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y_onehot = np.matrix(encoder.fit_transform(y))
    return y_onehot
# print(generate_y_onehot(y).shape)

weights = loadmat('ex4weights.mat')
theta1 = np.matrix(weights['Theta1']) # shape: 25*401
theta2 = np.matrix(weights['Theta2']) # shape: 10*26

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forwardprop(X, theta1, theta2):
    # 给X加1变为5000*401
    a1 = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
    z2 = a1*theta1.T
    a2 = sigmoid(z2) # shape: 5000*25
    # 给a2加1变为5000*26
    a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
    z3 = a2*theta2.T
    h = sigmoid(z3) # 5000*10
    return a1, z2, a2, z3, h

def cost_without_regu(X, y, theta1, theta2):
    m = X.shape[0]
    J = 0

    a1, z2, a2, z3, h = forwardprop(X, theta1, theta2)
    print(h.shape)
    for i in range(m):
        first = np.multiply(-y[i, :], np.log(h[i, :]))
        second = np.multiply(1-y[i, :], np.log(1-h[i, :]))
        J += np.sum(first-second)
    J = J/m
    return J

def cost_with_regu(X, y, theta1, theta2, learning_rate):
    m = X.shape[0]
    J = 0

    a1, z2, a2, z3, h = forwardprop(X, theta1, theta2)
    for i in range(m):
        first = np.multiply(-y[i, :], np.log(h[i, :]))
        second = np.multiply(1 - y[i, :], np.log(1 - h[i, :]))
        J += np.sum(first - second)
    J = J / m

    def get_theta(theta):
        rst = 0
        for i in range(theta.shape[0]):
            for j in range(1,theta.shape[1]):
                rst += theta[i, j]**2
        return rst
    theta_sum = get_theta(theta1)+get_theta(theta2)
    theta_sum = theta_sum*learning_rate/(2*m)
    return J+theta_sum

def accu(X, y, theta1, theta2):
    a1, z2, a2, z3, h = forwardprop(X, theta1, theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1) # np.argmax 返回沿轴axis最大值的索引。
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    return accuracy

print(cost_without_regu(X, y_onehot, theta1, theta2)) # 0.2876
print(cost_with_regu(X, y_onehot, theta1, theta2, 1)) # 0.3837

def sigmoidgradient(x):
    return np.multiply(sigmoid(x), (1-sigmoid(x))) # 注意这边使用的是np.multiply, 不是简单的*

def backpropo(X, y, theta1, theta2, learning_rate, alpha, iter, y_origin):

    for k in range(iter):
        cost = cost_with_regu(X, y_onehot, theta1, theta2, learning_rate)
        acc = accu(X, y_origin, theta1, theta2)
        print("epoch: "+str(k)+", cost: "+str(cost)+", acc: "+str(acc))
        a1, z2, a2, z3, h = forwardprop(X, theta1, theta2)
        delta1 = np.zeros(theta1.shape)  # 25*401
        delta2 = np.zeros(theta2.shape)  # 10*26

        m = X.shape[0]

        for i in range(m):
            h_i = h[i, :] # 1*10
            y_i = y[i, :] # 1*10
            z2_i = z2[i, :] # 1*25
            z2_i = np.insert(z2_i, 0, values=np.ones(z2_i.shape[0]), axis=1) # 1*26
            a2_i = a2[i, :] # 1*26
            a1_i = a1[i, :] # 1*401
            d3_i = h_i-y_i # 1*10
            # theta2: 10*26
            d2_i = np.multiply(d3_i*theta2, sigmoidgradient(z2_i)) # 1*26
            delta2 = delta2 + d3_i.T*a2_i
            delta1 = delta1 + d2_i[:, 1:].T*a1_i

        delta1 = delta1/m
        delta2 = delta2/m

        # 正则
        delta1[:, 1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
        delta2[:, 1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

        theta1 = theta1-alpha*delta1
        theta2 = theta2-alpha*delta2

    return theta1, theta2

# 如果没有事先给出theta, 应该怎么设计theta?
def initTheta():
    # 初始化设置
    input_size = 400
    hidden_size = 25
    num_labels = 10
    # 随机初始化完整网络参数大小的参数数组
    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
    # 将参数数组解开为每个层的参数矩阵
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    return theta1, theta2

backpropo(X, y_onehot, theta1, theta2, 1, 1, 100, y)

