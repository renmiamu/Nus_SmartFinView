import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 加载并准备数据
df = pd.read_csv("dataset/training_dataset.csv")  
df = df.drop(columns=["symbol"], errors="ignore")
df = df.fillna(df.mean(numeric_only=True))  # 用每列的平均值填充缺失值


X = df.drop(columns=["score"]).values  # 特征
y = df["score"].values  # 目标

# 标准化特征（对模型很重要）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转换为张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 2. 定义回归神经网络
class StockRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = StockRegressor(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 3. 训练模型
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 4. 保存模型和标准化器
torch.save(model.state_dict(), "score_model.pt")

import joblib
joblib.dump(scaler, "scaler.pkl")

print("✅ 模型和 scaler 已保存")
