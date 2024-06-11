import pandas as pd
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = pd.DataFrame(iris.data,columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=['iris_type'])
y = target['iris_type']

xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.33,random_state=1)

dtree = tree.DecisionTreeClassifier()
dtree.fit(xtrain,ytrain)
ypred = dtree.predict(xtest)
print('準確率:',dtree.score(xtest, ytest))
print(dtree.score(xtest, ytest))
print(pd.crosstab(ypred, ytest))

plt.figure(figsize=(25,20))
tree.plot_tree(dtree, feature_names=iris.feature_names,
               class_names=iris.target_names,filled=1)
plt.savefig('tree.png')

