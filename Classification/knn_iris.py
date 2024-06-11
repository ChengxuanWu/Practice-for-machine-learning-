#利用knn來進行花朵分類
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn import neighbors 

iris = datasets.load_iris()

x = pd.DataFrame(iris.data,columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=['iris_type'])
y = target['iris_type']

xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.33,random_state=1)

k=3
knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(xtrain,ytrain)
ypred = knn.predict(xtest)
print('準確率:', knn.score(xtest, ytest))
print(knn.score(xtest, ytest))
print(pd.crosstab(ypred, ytest))

colmap = np.array(['r','g','y'])
plt.figure(figsize=(10,5))
f = plt.subplot(121)
s = plt.scatter(x['sepal length (cm)'],x['sepal width (cm)'],c=colmap[y])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.subplot(122)
plt.scatter(x['petal length (cm)'],x['petal width (cm)'],c=colmap[y])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

legend1 = f.legend(*s.legend_elements(),loc='upper right',title='class')
f.add_artist(legend1)
plt.show()