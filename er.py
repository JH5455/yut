import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# 匯入資料
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 建立DataFrame
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
df = pd.DataFrame(data, columns=columns)
df['MEDV'] = target

# 檢查是否有缺失值
print(df.isnull().sum())

# 列出最高房價、最低房價、平均房價、中位數房價
print("最高房價:", df['MEDV'].max())
print("最低房價:", df['MEDV'].min())
print("平均房價:", df['MEDV'].mean())
print("中位數房價:", df['MEDV'].median())

# 繪製房價分布直方圖
plt.hist(df['MEDV'], bins=range(0, 60, 10), edgecolor='black')
plt.xlabel('房價')
plt.ylabel('頻率')
plt.title('房價分布直方圖')
plt.show()

# RM的值四捨五入到個位數，分析不同RM值的平均房價
df['RM_Rounded'] = df['RM'].round()
rm_grouped = df.groupby('RM_Rounded')['MEDV'].mean()
print(rm_grouped)

# 繪製不同RM值的平均房價直方圖
rm_grouped.plot(kind='bar', edgecolor='black')
plt.xlabel('平均房間數 (RM)')
plt.ylabel('平均房價')
plt.title('不同RM值的平均房價')
plt.show()

# 使用線性回歸模型來預測房價
X = df[['RM']]
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
print("均方誤差 (MSE):", mse)

# 繪製預測結果
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.xlabel('平均房間數 (RM)')
plt.ylabel('房價')
plt.title('房價預測')
plt.legend()
plt.show()