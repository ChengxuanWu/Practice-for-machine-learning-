#羅吉斯回歸(二元回歸)來進行鐵達尼號的生存預測
import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model

data = pd.read_csv('titanic.csv')
data.info()

median = np.nanmedian(data['Age'])
newdata = np.where(data['Age'].isnull(), median, data['Age'])

data['Age'] = newdata
data.info()

#產生labelencoder物件
label = preprocessing.LabelEncoder()
label_class = label.fit_transform(data['PClass'])

x = pd.DataFrame([data['Age'],data['SexCode'],label_class]).T
y = data['Survived']

logistic = linear_model.LogisticRegression()
logistic.fit(x,y)

print('迴歸係數:', logistic.coef_)
print('截距:', logistic.intercept_)

#使用crosstab來產生混淆矩陣來計算出sensitivity and specificity (accuracy)
pred = logistic.predict(x)
print(pd.crosstab(data['Survived'], pred))
#equal to (805+265) / 1313,
print(logistic.score(x,y)) 
