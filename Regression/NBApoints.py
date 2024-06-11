import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from sklearn.feature_selection import f_regression
from sklearn import preprocessing

NBApoints_data= pd.read_csv("NBApoints.csv")
#TODO

#將Pos欄位及Tm欄位資料轉換為數值

label_encoder_conver = preprocessing.LabelEncoder()
Pos_encoder_value = label_encoder_conver.fit_transform(NBApoints_data['Pos'])
# print(Pos_encoder_value)
# print("\n")

label_encoder_conver = preprocessing.LabelEncoder()
Tm_encoder_value = label_encoder_conver.fit_transform(NBApoints_data['Tm'])
# print(Tm_encoder_value)

train_X = pd.DataFrame([Pos_encoder_value, NBApoints_data['Age'], Tm_encoder_value],index=['pos','age','tm']).T
                        
NBApoints_linear_model = lr()
NBApoints_linear_model.fit(train_X, NBApoints_data["3P"])

NBApoints_linear_model_predict_result= NBApoints_linear_model.predict([[5,28,10]])
print("三分球得球數=",NBApoints_linear_model_predict_result)

r_squared = NBApoints_linear_model.score(train_X, NBApoints_data["3P"])
print("R_squared值=",r_squared)
fvalue = f_regression(train_X, NBApoints_data["3P"])[0]
pvalue = f_regression(train_X, NBApoints_data["3P"])[1]
value = pd.DataFrame([fvalue,pvalue], columns=['pos','age','tm'])
print(value)
# print("f_regresstion\n")
# print("P值="              )
# print("\n")
