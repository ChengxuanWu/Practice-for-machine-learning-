import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

ca = datasets.fetch_california_housing()

x = pd.DataFrame(ca.data, columns= ca.feature_names)
print(x.head())

y = pd.DataFrame(ca.target, columns=['MedHouseVal'])

xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.33,random_state=5)

lm = lr()
lm.fit(xtrain,ytrain)

ypred = lm.predict(xtest)

plt.scatter(ytest,ypred,edgecolors='#ed5386',color='#ede953')
plt.ylabel('predicted y', fontsize=14)
plt.xlabel('test y',fontsize=14)
plt.title('price - predicted price',fontsize=14)


mae_test = mae(y, ypred)
mse_test = mse(y, ypred)
rmse_test = mse_test ** 0.5 
r2_test = lm.score(xtest,ytest)

plt.figure()
plt.scatter(ypred, ypred-ytest)

plt.show()