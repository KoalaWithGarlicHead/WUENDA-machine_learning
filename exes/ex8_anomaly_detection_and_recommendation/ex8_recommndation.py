from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

data = loadmat("data/ex8_movies.mat") # Movie Review Dataset
Y = data['Y'] # 1682*943 943名观众给1682部电影的分数
R = data['R'] # 1682*943 943名观众是否给1682部电影打分

# 我们还可以通过将矩阵渲染成图像来尝试“可视化”数据。 我们不能从这里收集太多，但它确实给我们了解用户和电影的相对密度。
# fig, ax = plt.subplots(figsize=(12,12))
# ax.imshow(Y)
# ax.set_xlabel('Users')
# ax.set_ylabel('Movies')
# fig.tight_layout()
# plt.show()

data_para = loadmat("data/ex8_movieParams.mat")
X = data_para['X'] # 1682*10
Theta = data_para['Theta'] # 943*10

def generatePara(X, Theta):
    para = np.concatenate((np.ravel(X), np.ravel(Theta)))
    return para

def cost(params, Y, R, num_features, learning_rate):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]

    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)

    error = np.multiply(X*Theta.T-Y, R) # 1682*943
    squared_error = np.power(error, 2)
    J = np.sum(squared_error)/2
    J += learning_rate*np.sum(np.power(X, 2))/2
    J += learning_rate*np.sum(np.power(Theta, 2))/2

    # calculate the gradient
    X_grad = error*Theta+learning_rate*X
    Theta_grad = error.T*X+learning_rate*Theta

    para_grad = generatePara(X_grad, Theta_grad)

    return J, para_grad

movie_idx = {}
f = open('data/movie_ids.txt', encoding= 'gbk')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

# 创建自己的评分
ratings = np.zeros((Y.shape[0], 1))
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))

# 增加自己的数据
Y = np.append(Y, ratings, axis=1) # 1682*944
R = np.append(R, ratings != 0, axis=1) # 1682*944

# 定义变量
num_movies = Y.shape[0]
num_users = Y.shape[1]
num_features = 10
X = np.random.random((num_movies, num_features))
Theta = np.random.random((num_users, num_features))
para = generatePara(X, Theta) # (26260,))

# 将Y归一化
Ymean = np.zeros((num_movies, 1))
Ynorm = np.zeros((Y.shape[0], Y.shape[1]))
for i in range(num_movies):
    idx = np.where(R[i, :] == 1)[0]
    Ymean[i] = Y[i, idx].mean()
    Ynorm[i, idx] = Y[i, idx] - Ymean[i]


from scipy.optimize import minimize

fmin = minimize(fun=cost, x0=para, args=(Ynorm, R, num_features, 10),
                method='CG', jac=True, options={'maxiter': 100})

para = fmin.x
X = np.reshape(para[:num_movies*num_features], (num_movies, num_features))
Theta = np.reshape(para[num_movies*num_features:], (num_users, num_features))
predictions = np.matrix(X)*np.matrix(Theta.T)
my_preds = predictions[:, -1] + Ymean
sorted_preds = np.sort(my_preds, axis=0)[::-1]
idx = np.argsort(my_preds, axis=0)[::-1]

print("Top 10 movie predictions:")
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))