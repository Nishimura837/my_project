import pandas as pd
from sklearn.preprocessing import StandardScaler

#df_test.csv,df_train.csvを取得
df_test_path = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/df_test.csv"
df_train_path = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/df_train.csv"
df_test = pd.read_csv(df_test_path)
df_train = pd.read_csv(df_train_path)
print(df_test)
print(df_train)
    # #df_test,df_trainをcsvファイルとして出力
    # df_test.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/df_test_9_1.csv"\
    #                   , encoding='utf_8_sig', index=False)
    # df_train.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/df_train_9_1.csv"\
    #                   , encoding='utf_8_sig', index=False)

#重回帰分析では説明変数が複数存在するため、説明変数を標準化する必要がある
X = df_test.drop(columns=['case_name', 'RoI'])
print(X)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
print('X_sc')
print(X_sc)

# 標準化されたデータを新しいデータフレームに格納
scaled_df = pd.DataFrame(X_sc, columns=X.columns)
print(scaled_df)

