import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 使用相對路徑讀取資料集
train_file_path = os.path.join('week10', 'train.csv')
test_file_path = os.path.join('week10', 'test.csv')

# 檢查文件是否存在
if not os.path.exists(train_file_path):
    print(f"文件不存在: {train_file_path}")
else:
    train_data = pd.read_csv(train_file_path, delimiter=';')

if not os.path.exists(test_file_path):
    print(f"文件不存在: {test_file_path}")
else:
    test_data = pd.read_csv(test_file_path, delimiter=';')

# 如果文件存在，繼續執行後續操作
if os.path.exists(train_file_path) and os.path.exists(test_file_path):
    # 檢查有無缺失值
    print(train_data.isnull().sum())
    print(test_data.isnull().sum())

    # 將 age 列分成不同的區間
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    train_data['age_group'] = pd.cut(train_data['age'], bins=bins, labels=labels, right=False)

    # 繪製直方圖
    features = ['age_group', 'job', 'marital', 'education', 'loan']
    for feature in features:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=train_data, y=feature, hue='y')
        plt.xlabel('Count')
        if feature == 'age_group':
            plt.ylabel('Age Group')
            plt.title('Count of y by Age Group')
            plt.yticks(ticks=range(len(labels)), labels=labels)
            plt.gca().invert_yaxis()  # 反轉 y 軸
        else:
            plt.ylabel(f'{feature.capitalize()} Group')
            plt.title(f'Count of y by {feature.capitalize()} Group')
        # 在直方條右方標示出個別的數值
        for p in ax.patches:
            width = p.get_width()
            if width != 0:  # 跳過數值為 0 的標示
                plt.text(width + 700, p.get_y() + p.get_height() / 2, int(width), ha='center', va='center')
        plt.xticks([0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
        plt.show()

    # 保留age、balance、loan三個特徵來訓練羅吉斯回歸的模型
    train_data['loan'] = train_data['loan'].apply(lambda x: 1 if x == 'yes' else 0)
    X_train = train_data[['age', 'balance', 'loan']]
    y_train = train_data['y'].apply(lambda x: 1 if x == 'yes' else 0)

    test_data['loan'] = test_data['loan'].apply(lambda x: 1 if x == 'yes' else 0)
    X_test = test_data[['age', 'balance', 'loan']]
    y_test = test_data['y'].apply(lambda x: 1 if x == 'yes' else 0)

    # 訓練羅吉斯回歸模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 預測測試集
    y_pred = model.predict(X_test)

    # 計算準確度
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Logistic Regression Accuracy: {accuracy:.2f}')
