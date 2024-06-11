import numpy as np
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt

Length = [51,46,51,45,51,50,33,38,37,33,33,21,23,24]
Weight = [10.2,8.8,8.1,7.7,9.8,7.2,4.8,4.6,3.5,3.3,4.3,2.0,1.0,2.0]

df = pd.DataFrame({'Length':Length,'Weight':Weight})

k = 3

kmeans = cluster.KMeans(n_clusters=k,random_state=12)
kmeans.fit(df)
print(kmeans.labels_)

colmap = np.array(['r','g','y'])
plt.scatter(df['Length'],df['Weight'],c=colmap[kmeans.labels_])
plt.show()