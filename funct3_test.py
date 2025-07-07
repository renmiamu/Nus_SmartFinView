import pandas as pd
import torch
import torch.nn as nn
import joblib

# 1. 加载测试数据
test_df = pd.read_csv("test_data.csv")
symbols = test_df["symbol"]
X_test = test_df.drop(columns=["symbol"], errors="ignore")
X_test = X_test.fillna(X_test.mean(numeric_only=True))

# 2. 加载 scaler 和标准化
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X_test)

# 3. 定义模型结构
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

# 4. 加载模型并预测
model = StockRegressor(X_scaled.shape[1])
model.load_state_dict(torch.load("score_model.pt"))
model.eval()

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    predicted_scores = model(X_tensor).squeeze().numpy()

# 5. 输出得分
result = pd.DataFrame({
    "symbol": symbols,
    "predicted_score": predicted_scores
})
result.to_csv("predicted_scores.csv", index=False)
print("✅ 打分结果已保存到 predicted_scores.csv")
