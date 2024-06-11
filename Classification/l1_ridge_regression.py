#紅酒分類問題(使用正規化與降維分析)
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = datasets.load_wine()
X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.DataFrame(data.target)
x_train, x_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=0, stratify=y)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(x_train)
X_test_std = stdsc.fit_transform(x_test)

lr = LogisticRegression(penalty='l1', C=1.0, solver='saga')
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))    

print(lr.intercept_) 
print(lr.coef_)

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'yellow', 'magenta', 'black', 'pink', 
          'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='saga', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=X.columns[column], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.ylabel('weight coef')
plt.xlabel('c')
plt.xscale('log')
plt.legend(loc='upper left', bbox_to_anchor=(1.05,1), ncol=1, fancybox= True)
plt.show()

#循序特徵選擇(SBS)挑選對於準確度影響較大的參數
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, 
                 random_state = 1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = tts(X, y, test_size=self.test_size,
                                               random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1 
            
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    
knn = KNeighborsClassifier(n_neighbors=5)

sbs = SBS(knn, k_features = 1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]

plt.figure()
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylabel('accuracy')
plt.xlabel('number of features')
plt.ylim([0.7, 1.02])
plt.grid()
plt.show()