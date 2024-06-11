from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
# TODO
dm = datasets.load_diabetes()
#get x
# TODO 
x = pd.DataFrame(dm.data,columns=dm.feature_names)
y = dm.target
#Total number of examples
# TODO 
lm = lr()
lm.fit(x,y)
y_pred_tot = lm.predict(x)
msetot = mse(y, y_pred_tot)
rtot = lm.score(x, y)
print('Total number of examples')
print('MSE=', format(msetot,'.4f'))
print('R-squared=',format(rtot,'.4f'))
#3:1 100
xTrain, xTest, yTrain, yTest= tts(x,y,test_size=0.25,random_state=100)
lm2=lr()
lm2.fit(xTrain, yTrain)
# TODO 
y_pred_train = lm2.predict(xTrain)
y_pred_test = lm2.predict(xTest)
msetrain = mse(yTrain, y_pred_train)
msetest = mse(yTest, y_pred_test)
rtrain = lm2.score(xTrain,yTrain)
rtest = lm2.score(xTest, yTest)
print('Split 3:1')
print('train MSE=', format(msetrain,'.4f'))
print('test MSE=', format(msetest,'.4f'))
print('train R-squared=', format(rtrain,'.4f'))
print('test R-squared=', format(rtest,'.4f'))
