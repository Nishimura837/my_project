# 線形重回帰分析
import pandas as pd
from sklearn.preprocessing import StandardScaler
# 線形モデル
from sklearn import linear_model

#df_test.csv,df_train.csvを取得
df_test_path = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/df_test.csv"
df_train_path = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/df_train.csv"
df_test = pd.read_csv(df_test_path)
df_train = pd.read_csv(df_train_path)

X_train = df_train.drop(columns=['case_name', 'RoI'])
y_train = df_train["RoI"]
X_test = df_test.drop(columns=['case_name', 'RoI'])
y_test = df_test["RoI"]


#重回帰分析では説明変数が複数存在するため、説明変数を標準化する
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

# 標準化されたデータを新しいデータフレームに格納
scaled_X_train = pd.DataFrame(X_train_sc, columns=X_train.columns)
scaled_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#csvファイルとして出力
scaled_X_train.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/scaled_X_train.csv",encoding='utf_8_sig', index=False)
scaled_X_test.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/scaled_X_test.csv",encoding='utf_8_sig', index=False)

#モデルの作成と適用
model = linear_model.LinearRegression()
model.fit(scaled_X_train, y_train)

print('回帰係数')
print(model.coef_)

#決定係数について
print('決定係数')
print(model.score(scaled_X_train, y_train))
print(model.score(scaled_X_test, y_test))

#RMSE(二乗平均平方根誤差)について





#MSE(平均二乗誤差)について




#MAE(平均絶対誤差)について

