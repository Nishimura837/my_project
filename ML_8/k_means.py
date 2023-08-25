import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
print(df)

#列名を指定して削除
df1 = df.drop(columns=['sepal length (cm)', 'sepal width (cm)'])
print(df1)

#教師なし学習のk-meansを使ってクラスタリングを行う
X = df.iloc[:,:2]
Y = df.iloc[:,2]

kmeans = KMeans(n_clusters=3, init='random', random_state=42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)

#クラスタリングしたデータをプロットする





































#元データの図示

# カラーマップの色を定義
colors = [  (1, 0, 0),   # 赤
            (0, 1, 0),   # 緑
            (0, 0, 1)]   # 青

# カラーマップを作成
cmap = ListedColormap(colors)

#データを図示する
d= df1.iloc
import matplotlib.pyplot as plt 
plt.scatter(d[0:50, 0], d[:50, 1], color='r', marker='s', edgecolor='black', label='setosa')
plt.scatter(d[50:100, 0], d[50:100, 1], color='g', marker='^', edgecolor='black', label='versicolor')
plt.scatter(d[100:, 0], d[100:, 1], color='b', marker='o', edgecolor='black', label='virginica')
plt.xlabel("petal length(cm)")
plt.ylabel("petal width(cm)")
plt.legend(loc='upper right')
plt.show()

