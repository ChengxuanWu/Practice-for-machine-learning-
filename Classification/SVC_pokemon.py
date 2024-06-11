import pandas as pd
# 載入寶可夢資料集
# TODO
data = pd.read_csv('pokemon.csv')
# 處理遺漏值
features = ['Attack', 'Defense']
# TODO
data.dropna(axis=0, how='any', subset=features,inplace=True)
# 取出目標寶可夢的 Type1 與兩個特徵欄位
# TODO
mask1 = data['Type1'] == 'Normal'
mask2 = data['Type1'] == 'Fighting'
mask3 = data['Type1'] == 'Ghost'
data = data[mask1|mask2|mask3]
x = data.iloc[:,6:8]
y = data.iloc[:,2]
# 編碼 Type1
from sklearn.preprocessing import LabelEncoder
# TODO
label = LabelEncoder()
y = label.fit_transform(y)
# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
std = StandardScaler()
x = std.fit_transform(x)

# 建立線性支援向量分類器，除以下參數設定外，其餘為預設值
# #############################################################################
# C=0.1, dual=False, class_weight='balanced'
# #############################################################################
from sklearn.svm import LinearSVC
# TODO
lm = LinearSVC(C=0.1, dual=False, class_weight='balanced')
lm.fit(x, y)
y_pred = lm.predict(x)
# 計算分類錯誤的數量
# TODO
print((y_pred != y).sum())
# 計算準確度(accuracy)
from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(y, y_pred))

# 計算有加權的 F1-score (weighted)
from sklearn.metrics import f1_score
# TODO
print('F1-score: ', f1_score(y,y_pred,average='weighted'))

# 預測未知寶可夢的 Type1
# TODO


