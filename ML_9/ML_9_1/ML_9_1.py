import pandas as pd

#df_test.csv,df_train.csvを取得
df_test_path = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/df_test.csv"
df_train_path = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/df_train.csv"
df_test = pd.read_csv(df_test_path)
df_train = pd.read_csv(df_train_path)
# print(df_test)
# print(df_train)
# #df_test,df_trainをcsvファイルとして出力
# df_test.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/df_test_9_1.csv",encoding='utf_8_sig', index=False)
# df_train.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/df_train_9_1.csv", encoding='utf_8_sig', index=False)