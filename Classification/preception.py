import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

class Perception(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # self.w_[0]+= update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 0, 1)
    
data = datasets.load_iris()
X = pd.DataFrame(data.data[:100,[0, 2]])
y = data.target[0:100]
X.tail()

plt.scatter(X.iloc[:50, 0], X.iloc[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X.iloc[50:100, 0], X.iloc[50:100, 1], color='blue', 
            marker='x', label='versicoclor')
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.legend(loc='upper left')
plt.show()

ppn = Perception(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassification')
plt.show()















