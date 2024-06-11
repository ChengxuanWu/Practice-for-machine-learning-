import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#建立原始數據並轉置成DF
temp = np.random.randint(22,30,20)
sale = np.random.rand(20)*10
x = pd.DataFrame(temp,columns=['temprature'])
y = pd.DataFrame(sale,columns=['sale'])

lm = LinearRegression()
lm.fit(x,y)
print('coef:',lm.coef_)
print('intercept:',lm.intercept_)

tempn = pd.DataFrame(np.array([26,32]))
salepre = lm.predict(tempn)
print(salepre)

plt.scatter(x, y)
regression_sale = lm.predict(x)
plt.plot(x,regression_sale,color='b')
plt.plot(tempn, salepre, color = 'red', markersize = 10, marker ='o')
plt.xlabel('temp',fontsize=14)
plt.ylabel('sale',fontsize=14)
plt.title('ML excersise',fontsize=14)
plt.show()