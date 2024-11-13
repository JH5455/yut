import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 匯入資料，並指定第一行作為列名
df = pd.read_csv('BOSTON_HOUSE_PRICES.CSV', header=1)

# 顯示前幾行資料
print(df.head())

# 檢查是否有缺失值
print(df.isnull().sum())

# 列出統計數據
print("最高房價:", df['MEDV'].max())
print("最低房價:", df['MEDV'].min())
print("平均房價:", df['MEDV'].mean())
print("中位數房價:", df['MEDV'].median())

# 繪製房價分布直方圖，以10為區間
plt.hist(df['MEDV'], bins=range(0, int(df['MEDV'].max()) + 10, 10), edgecolor='black')
plt.title('Distribution of House Price')
plt.xlabel('House Price Range (thousand dollars)')
plt.ylabel('Count')
plt.show()

# 將RM值四捨五入到個位數，並分析不同RM值的平均房價
df['RM_Rounded'] = df['RM'].round()
rm_grouped = df.groupby('RM_Rounded')['MEDV'].mean()
print(rm_grouped)

# 繪製不同RM值的平均房價直方圖
rm_grouped.plot(kind='bar', edgecolor='black')
plt.title('Distribution of Boston Housing Price Group by RM')
plt.xlabel('MEDV')
plt.ylabel('RM')
plt.show()

# 準備線性回歸的數據
X = df[['RM']]  # 特徵
y = df['MEDV']  # 目標

# 將數據分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 進行預測
y_pred = model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("均方誤差:", mse)
print("R^2 分數:", r2)

# 繪製結果
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('線性回歸: RM vs MEDV')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()