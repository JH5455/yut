import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 載入訓練資料集
train_data = pd.read_csv('train.csv')

# 2. 印出前10筆資料，了解資料結構
print(train_data.head(10))

# 3. 資料前處理

# (1) 檢查缺失值並進行處理
print(train_data.isnull().sum())  # 檢查每個欄位的缺失值數量

# 填補缺失值
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())  # 年齡缺失用中位數填補
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])  # Embarked缺失用眾數填補
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())  # Fare缺失用中位數填補

# (2) 類別標籤轉換：將性別（Sex）轉換成數字：male -> 0, female -> 1
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# (3) 將 Embarked 進行獨熱編碼（One-Hot Encoding），將類別變數轉換為數字
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

# (4) 標準化處理：將數值特徵（如Age, Fare）進行標準化，使其有相似的尺度
scaler = StandardScaler()
train_data[['Age', 'Fare']] = scaler.fit_transform(train_data[['Age', 'Fare']])

# 4. 定義特徵與目標變數
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]  # 特徵
y = train_data['Survived']  # 目標變數（是否生還）

# 5. 分割訓練資料集與測試資料集，比例為80%訓練、20%測試
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 訓練羅吉斯回歸（Logistic Regression）模型
log_reg = LogisticRegression(max_iter=1000)  # 設置最大迭代次數，以避免收斂問題
log_reg.fit(X_train, y_train)

# 使用羅吉斯回歸模型進行預測
y_pred_log_reg = log_reg.predict(X_test)

# 7. 訓練決策樹（Decision Tree）模型
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# 使用決策樹模型進行預測
y_pred_dt = dt.predict(X_test)

# 8. 評估模型：計算各項指標（準確度、精確度、召回率、F1-score）
def evaluate_model(y_test, y_pred, model_name):
    print(f"{model_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print("-" * 30)

# 評估羅吉斯回歸模型
evaluate_model(y_test, y_pred_log_reg, "Logistic Regression")

# 評估決策樹模型
evaluate_model(y_test, y_pred_dt, "Decision Tree")

# 9. 混淆矩陣圖：比較兩個模型的預測結果
def plot_confusion_matrices(cm1, cm2, model_name1, model_name2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 繪製第一個模型的混淆矩陣
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Survived', 'Survived'], 
                yticklabels=['Not Survived', 'Survived'], 
                ax=axes[0])
    axes[0].set_title(f'Confusion Matrix: {model_name1}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # 繪製第二個模型的混淆矩陣
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Not Survived', 'Survived'], 
                yticklabels=['Not Survived', 'Survived'], 
                ax=axes[1])
    axes[1].set_title(f'Confusion Matrix: {model_name2}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    # 調整圖表間距
    plt.tight_layout()
    plt.show()

# 計算羅吉斯回歸與決策樹的混淆矩陣
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
cm_dt = confusion_matrix(y_test, y_pred_dt)

# 繪製混合圖表
plot_confusion_matrices(cm_log_reg, cm_dt, "Logistic Regression", "Decision Tree")
