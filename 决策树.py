import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

Epoches = 20

data = pd.read_csv('rides_per_10_minutes_december_2016.csv')

# 将 'lpep_pickup_datetime' 转换为时间戳
data['lpep_pickup_datetime'] = pd.to_datetime(data['lpep_pickup_datetime'])
data['timestamp'] = data['lpep_pickup_datetime'].astype(np.int64) // 10**9

# 检查数据是否存在 NaN 或 inf
if np.isnan(data['timestamp'].values).any() or np.isinf(data['timestamp'].values).any():
    print("Data contains NaN or inf values. Please clean the data.")
    exit()

# 绘制时间戳与骑行次数关系的折线图
plt.figure(figsize=(10, 6))
plt.plot(data['timestamp'].values, data['Number of Rides'].values, label='Number of Rides vs Time')
plt.xlabel('Time (Unix Timestamp)')
plt.ylabel('Number of Rides')
plt.title('Number of Rides over Time')
plt.legend()
plt.show()


# 对时间戳进行标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(data['timestamp'].values.reshape(-1, 1))

# 对y进行标准化
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(data['Number of Rides'].values.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, shuffle=True)

# 创建决策树回归模型
model = DecisionTreeRegressor(max_depth=5)

# 训练模型
model.fit(X_train, Y_train)

# 在训练集和测试集上进行预测
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# 计算测试集上的损失
test_loss = mean_squared_error(Y_test, test_pred)

# 计算测试集的准确率（R2）
test_r2 = r2_score(Y_test, test_pred)

# 打印测试损失和R2
print(f'Test MSE = {test_loss:.4f}')
print(f'Test R2: {test_r2:.4f}')

# 创建一个包含预测结果图的图形
plt.figure(figsize=(10, 6))

# 绘制预测结果图
plt.plot(X_test, test_pred, label='Predicted Values', c='r')
plt.scatter(X_test, Y_test, label='True Values')
plt.xlabel('Time (Unix Timestamp)')
plt.ylabel('Number of Rides')
plt.title('Prediction vs True Values')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
