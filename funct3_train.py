import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# 1. 加载训练数据
df_train = pd.read_csv("dataset/training_dataset.csv")
df_train = df_train.drop(columns=["symbol"], errors="ignore")
df_train = df_train.fillna(df_train.mean(numeric_only=True))

X = df_train.drop(columns=["score"]).values
y = df_train["score"].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转换为张量
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

class StockRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)


model = StockRegressor(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []

# 3. 模型训练
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Train Loss: {loss.item():.6f}")

# 4. 绘制训练误差曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve_linear.png")


# 5. 加载测试集并评估
df_test = pd.read_csv("dataset/merged_test_data.csv")
df_test = df_test.drop(columns=["symbol"], errors="ignore")
df_test = df_test.fillna(df_test.mean(numeric_only=True))

X_test = df_test.drop(columns=["score"]).values
y_test = df_test["score"].values
X_test_scaled = scaler.transform(X_test)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy().flatten()

# 6. 输出测试指标
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"📊 Test MSE: {mse:.6f}")
print(f"📊 Test MAE: {mae:.6f}")
print(f"📈 R² Score (拟合度): {r2:.6f}")

# 7. 绘制真实值 vs 预测值曲线
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="True Score", linestyle='-', marker='o', markersize=3)
plt.plot(predictions, label="Predicted Score", linestyle='--', marker='x', markersize=3)
plt.title("True vs Predicted Score on Test Set")
plt.xlabel("Sample Index")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_vs_true.png")

# 8. 保存模型和 scaler
torch.save(model.state_dict(), "score_model_linear.pt")
joblib.dump(scaler, "scaler_linear.pkl")
print("✅ 模型、标准化器和图像文件已保存")
