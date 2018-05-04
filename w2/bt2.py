import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.colors as colors

from random import shuffle
import numpy as np
sns.set(style="white", color_codes=True)

iris = pd.read_csv('iris.csv')

# in ra top 10 dòng đầu tiên
print(iris.head())

# Đếm số lượng phần tử của mỗi loài 
print(iris["species"].value_counts())

#plot dữ liệu lên xem thử

iris.plot(kind='scatter',x = "sepal_length", y = "sepal_width" )
plt.show()

# plot lên với các màu phân biệt
sns.FacetGrid(iris,hue='species',size=5).map(plt.scatter,'sepal_length','sepal_width').add_legend()
plt.show()
#Quan sát dữ liệu, ta thấy loài setosa nằm tách biệt hoàn toàn với hai loài còn lại, 
#versicolor và virginica hoà trộn với nhau, khó phân biệt

# thử hiện thị dữ liệu với thuộc tính petal_lenth và petal_width xem sao

sns.FacetGrid(iris,hue='species',size=5).map(plt.scatter,'petal_length','petal_width').add_legend()
plt.show()

# với hai thuộc tính này, ta có thể nhận thấy rõ ràng rằng setosa nằm tách biệt hoàn toàn so với hai loại còn lại.

# giữa versicolor và virginica cũng nằm tách biệt riêng hẵn ra, vùng overlap còn hầu như là không có
# -> có thể thí nghiệm phân lớp hoặc gom cụm bằng hai thuộc tính versicolor và virginica xem kết quả như thế nào.


#Thử show biểu đồ phân bố của dữ liệu lên
sns.boxplot(x="species", y="petal_length", data=iris)
plt.show()

#Nhìn bản đồ, ta thấy, giá trị pental_length của viginica nằm ở đoạn từ phần phân vị thứ 3 và max của versicolor

#Phân lớp dữ liệu dựa vào cây quyết định
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
#tạo tập train và tập test

iris_train,iris_test =  train_test_split(iris)

features =['sepal_length','sepal_width','petal_length','petal_width']

xtrain , ytrain, xtest, ytest = iris_train[features], iris_train['species'],iris_test[features], iris_test['species']

xtrain_len = xtrain.shape[0]


#decisionTree = DecisionTreeClassifier(criterion = 'entropy')
#search xem depth bao nhiêu là tốt nhất
for tree_depth in range(2,int( np.sqrt(xtrain_len))+1):

    clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=tree_depth)
    scores = cross_val_score(estimator=clf, X=xtrain, y=ytrain, cv=7)
    print((tree_depth,scores.mean()))

# kq lúc thực nghiệm
# (2, 0.9726190476190476)
# (3, 0.9646825396825397)
# (4, 0.9736111111111112)
# (5, 0.9646825396825397)
# (6, 0.9736111111111112)
# (7, 0.9646825396825397)
# (8, 0.9646825396825397)
# (9, 0.9736111111111112)
# (10, 0.9646825396825397)

# nhận xét là với max_depth = 4 cho kết quả tốt nhất
# Huấn luyện lại kế
clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=4)
clf.fit(xtrain,ytrain)
ypred =  clf.predict(xtest)

print('acc: %2f' % (sum(ypred == ytest.values)/len(ytest)*100))
#acc: 97.368421