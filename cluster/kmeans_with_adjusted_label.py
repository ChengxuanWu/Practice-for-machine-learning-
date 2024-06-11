#kmeans 與 iris分群並調整標籤
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import cluster
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = pd.DataFrame(iris.data,columns=iris.feature_names)
x.columns = ['petal_length','petal_width','sepal_length','sepal_width']
target = pd.DataFrame(iris.target)
y = iris.target
k = 3

kmeans = cluster.KMeans(n_clusters=k,random_state=12,n_init='auto')
kmeans.fit(x)
print(kmeans.labels_)
print(y)

colmap = np.array(['r','g','y'])
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(x['petal_length'],x['petal_width'],c=colmap[y])
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('Real clustering')
plt.subplot(122)
plt.scatter(x['petal_length'],x['petal_width'],c=colmap[kmeans.labels_])
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('kmeans clustering')
plt.show()

#adjust labels
pred_y = np.choose(kmeans.labels_, [1,0,2])
print('adjusted kmeans clustering')
print(pred_y)
print('real classification')
print(y)

colmap = np.array(['r','g','y'])
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(x['petal_length'],x['petal_width'],c=colmap[y])
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('Real clustering')
plt.subplot(122)
plt.scatter(x['petal_length'],x['petal_width'],c=colmap[pred_y])
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('kmeans clustering')
plt.show()
