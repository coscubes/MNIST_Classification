import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from nolearn.dbn import DBN
import timeit
import matplotlib.pyplot as plt

train = np.loadtxt('mnist_train.csv', delimiter=',', dtype=None)
test = np.loadtxt('mnist_test.csv', delimiter=',', dtype=None)

X_train = np.delete(train, [0], axis = 1)
y_train = train[:,0]
X_test = np.delete(test, [0], axis = 1)
y_test = test[:,0]

clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print "nearest neighbors accuracy: ",acc_knn
joblib.dump(clf_knn, 'classifiers/KNN/knn.pkl')

clf_nn = DBN([X_train.shape[1], 300, 10],learn_rates=0.3,learn_rate_decays=0.9,epochs=10)
clf_nn.fit(X_train, y_train)
acc_nn = clf_nn.score(X_test,y_test)
print "neural network accuracy: ",acc_nn

clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print "random forest accuracy: ",acc_rf
joblib.dump(clf_rf, 'classifiers/RF/rf.pkl')

clf_sgd = SGDClassifier()
clf_sgd.fit(X_train, y_train)
y_pred_sgd = clf_sgd.predict(X_test)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print "stochastic gradient descent accuracy: ",acc_sgd
joblib.dump(clf_sgd, 'classifiers/SGD/sgd.pkl')

clf_svm = LinearSVC()
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print "Linear SVM accuracy: ",acc_svm
joblib.dump(clf_svm, 'classifiers/SVM/svm.pkl')

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

titles = ['KNN with N = 5',
          'Random Forest',
          'SGD',
          'Linear SVC']

for i, clf in enumerate((clf_knn, clf_rf, clf_sgd, clf_svm)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
