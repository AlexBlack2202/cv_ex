
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.colors as colors
from sklearn.cross_validation import train_test_split

from random import shuffle
import numpy as np
sns.set(style="white", color_codes=True)

iris = pd.read_csv('iris.csv')
iris['targets'] = pd.Categorical(iris['species']).codes

features =['sepal_length','sepal_width','petal_length','petal_width']

iris_train,iris_test =  train_test_split(iris)
xtrain , ytrain, xtest, ytest = iris_train[features], iris_train['targets'],iris_test[features], iris_test['targets']


from sklearn import svm
from sklearn.model_selection import GridSearchCV # dùng để search tham số tối ưu

# xây dựng grid search cho svm

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [100,10,1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf']},
 ]


clf = GridSearchCV(svm.SVC(), param_grid,cv=5)
clf.fit(xtrain,ytrain)


print('best score')
print(clf.best_score_)
print('best param')
print(clf.best_params_)

#Kết quả cho ra tốt nhất là linear kernel với C = 1

svc_clf = svm.SVC(kernel='linear',C=1)
svc_clf.fit(xtrain,ytrain)
ypred =  svc_clf.predict(xtest)
print('acc: %2f' % (sum(ypred == ytest.values)/len(ytest)*100))