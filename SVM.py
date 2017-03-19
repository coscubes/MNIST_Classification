import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from nolearn.dbn import DBN
import timeit

train = np.loadtxt('mnist_train.csv', delimiter=',', dtype=None)
test = np.loadtxt('mnist_test.csv', delimiter=',', dtype=None)

X_train = np.delete(train, [0], axis = 1)
y_train = train[:,0]
X_test = np.delete(test, [0], axis = 1)
y_test = test[:,0]

'''
print X_train.shape, X_train.size
print y_train.shape, y_train. size
print X_test.shape, X_test.size
print y_test.shape, y_test. size
'''

clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print "random forest accuracy: ",acc_rf

clf_sgd = SGDClassifier()
clf_sgd.fit(X_train, y_train)
y_pred_sgd = clf_sgd.predict(X_test)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print "stochastic gradient descent accuracy: ",acc_sgd

clf_svm = LinearSVC()
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print "Linear SVM accuracy: ",acc_svm

clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print "nearest neighbors accuracy: ",acc_knn

clf_nn = DBN([X_train.shape[1], 300, 10],learn_rates=0.3,learn_rate_decays=0.9,epochs=15)
clf_nn.fit(X_train, y_train)
acc_nn = clf_nn.score(X_test,y_test)
print "neural network accuracy: ",acc_nn
