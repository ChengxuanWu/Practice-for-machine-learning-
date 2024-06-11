import os
os.environ['OMP_NUM_THREADS']='1'
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
#raw data
x, y = make_blobs()
plt.scatter(x[:,0],x[:,1])
print(x)

#test the inertia with k form 2 ~10
data = {}
for i in range(2,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=200,n_init=15,random_state=0)
    kmeans.fit(x)
    if kmeans.inertia_ < 100:
        data[i] = kmeans.inertia_
        
print('inertia < 100:',len(data))
print('minimum of inertia:', min(data.values()))

kmeans = KMeans(n_clusters=7,init='k-means++',max_iter=200,n_init=15,random_state=0)
kmeans.fit(x)
kmeans_pred = kmeans.predict(x)

print('cluster centers:',kmeans.cluster_centers_)
print('minimum of x:', min(kmeans.cluster_centers_[:,0]))
print('maximum of y:', max(kmeans.cluster_centers_[:,1]))

