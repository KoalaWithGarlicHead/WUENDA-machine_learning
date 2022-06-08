from scipy.io import loadmat
from sklearn import svm

spam_train = loadmat('data/spamTrain.mat')
spam_test = loadmat('data/spamTest.mat')

X = spam_train['X']
y = spam_train['y']
Xtest = spam_test['Xtest']
ytest = spam_test['ytest']

print(X.shape, y.shape, Xtest.shape, ytest.shape)

svc = svm.SVC()
svc.fit(X, y)
print('Training accuracy = '+str(svc.score(X,y)))
print('Test accuracy = '+str(svc.score(Xtest,ytest)))
