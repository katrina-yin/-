import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

Epoches = 20

class EIModel(nn.Module):
    def __init__(self, in_features):
        super(EIModel, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, x):
        y = self.linear(x)
        return y

# 假设CSV文件和Python脚本在同一个文件夹中
data_file_path = 'rides_per_10_minutes_december_2016.csv'  # 使用相对路径
data = pd.read_csv(data_file_path)

# 将 'lpep_pickup_datetime' 转换为时间戳
data['lpep_pickup_datetime'] = pd.to_datetime(data['lpep_pickup_datetime'])
data['timestamp'] = data['lpep_pickup_datetime'].astype(np.int64) // 10**9

# 计算每个数据点所在的10分钟时间段
data['interval'] = (data['timestamp'] - data['timestamp'].min()) // 600

# 检查 NaN 或 inf
if np.isnan(data['timestamp'].values).any() or np.isinf(data['timestamp'].values).any():
    print("Data contains NaN or inf values. Please clean the data.")
    exit()

# 折线图
plt.figure(figsize=(10, 6))
plt.plot(data['interval'].values, data['Number of Rides'].values, label='Number of Rides vs Time')
plt.xlabel('Interval (10-Minute Bins)')
plt.ylabel('Number of Rides')
plt.title('Number of Rides over Time')
plt.legend()
plt.show()

# 对时间戳进行标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(data['interval'].values.reshape(-1, 1))

# 对y进行标准化
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(data['Number of Rides'].values.reshape(-1, 1))

# 将数据转换为PyTorch张量
X = torch.from_numpy(X_scaled).type(torch.FloatTensor)
Y = torch.from_numpy(Y_scaled).type(torch.FloatTensor)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# 创建线性回归模型
model = EIModel(in_features=1)
loss_function = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)  # 使用 Adam 优化器

# 存储损失函数值和准确率
train_losses = []
test_losses = []

# 训练模型
for epoch in range(Epoches):
    model.train()
    for x, y in zip(X_train, Y_train):
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 计算训练集上的损失和测试集上的损失
    model.eval()
    train_pred = model(X_train)
    train_loss = loss_function(train_pred, Y_train).item()

    test_pred = model(X_test)
    test_loss = loss_function(test_pred, Y_test).item()

    if torch.isnan(train_pred).any() or torch.isinf(train_pred).any():
        print(f"NaN or inf found in train predictions at epoch {epoch+1}")
        continue
    if torch.isnan(test_pred).any() or torch.isinf(test_pred).any():
        print(f"NaN or inf found in test predictions at epoch {epoch+1}")
        continue

    try:
        train_r2 = r2_score(Y_train.detach().numpy(), train_pred.detach().numpy())
        test_r2 = r2_score(Y_test.detach().numpy(), test_pred.detach().numpy())
    except ValueError as e:
        print(f"ValueError at epoch {epoch+1}: {e}")
        train_r2 = None
        test_r2 = None

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # 每个epoch都打印损失和R2
    print(f'Epoch {epoch+1}: Train MSE = {train_loss:.4f}, Test MSE = {test_loss:.4f}')
    if train_r2 is not None and test_r2 is not None:
        print(f'Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')

# 创建一个包含两个子图的图形
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 绘制预测结果图
axes[0].plot(X_test.detach().numpy(), test_pred.detach().numpy(), label='Predicted Values', c='r')
axes[0].scatter(X_test.detach().numpy(), Y_test.detach().numpy(), label='True Values')
axes[0].set_xlabel('Interval (10-Minute Bins)')
axes[0].set_ylabel('Number of Rides')
axes[0].set_title('Prediction vs True Values')
axes[0].legend()

# 绘制损失函数变化曲线
axes[1].plot(range(Epoches), train_losses, label='Train Loss')
axes[1].plot(range(Epoches), test_losses, label='Test Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss Curve')
axes[1].legend()

# 调整子图之间的间距
plt.tight_layout()
plt.show()
