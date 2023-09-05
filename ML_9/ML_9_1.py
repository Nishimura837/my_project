import os
import pandas as pd

#csvファイルをpandasを使って読み込む
#csvファイルが保存されているルートディレクトリのパス
root_directory = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/"

#フォルダごとに処理を繰り返す
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    csv_file_path = os.path.join(folder_path, "inputdata.csv")  #各フォルダ内のinputdata.csvファイルのパス
    if os.path.isfile(csv_file_path):
        df_name = f"df_{folder_name}"   #データフレーム名をフォルダ名に基づいて作成

        #csvファイルをデータフレームとして読み込む
        globals()[df_name] = pd.read_csv(csv_file_path)

        #カテゴリ変数である"exhaust"を[Label Encoding]により数値化する
        #"exhaust"の値に応じて、"a"なら"0"、"b"なら1、"off"なら"2"に変換する
        globals()[df_name].loc[globals()[df_name]['exhaust'] == "a", 'exhaust'] = 0
        globals()[df_name].loc[globals()[df_name]['exhaust'] == "b", 'exhaust'] = 1
        globals()[df_name].loc[globals()[df_name]['exhaust'] == "off", 'exhaust'] = 2

        print(df_name)
        print(globals()[df_name])


#作成されたすべてのデータフレームの名前を取得
dataframe_names = [var_name for var_name in globals() if isinstance(globals()[var_name], pd.DataFrame)]
#データフレームの名前を表示
for name in dataframe_names:
    print(name)

