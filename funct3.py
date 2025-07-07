import yfinance as yf
import pandas as pd

# 读取符号列表
df = pd.read_excel("train.xlsx")
symbols = df.iloc[:, 0].dropna().unique().tolist()
print(f"共有 {len(symbols)} 个股票代码")

# 抓取历史收益率
returns_data = []
for symbol in symbols:
    try:
        df = yf.download(symbol, start="2021-01-01", end="2024-01-01", progress=False)
        if not df.empty:
            start_price = df["Close"].iloc[0].item()
            end_price = df["Close"].iloc[-1].item()
            cumulative_return = end_price / start_price - 1
            returns_data.append({"symbol": symbol, "return": cumulative_return})
            print(f"{symbol}: {cumulative_return:.2%}")
        else:
            print(f"{symbol}: 无数据")
    except Exception as e:
        print(f"{symbol}: 报错 - {e}")
        continue

# 构建收益率表并打标签
returns_df = pd.DataFrame(returns_data)
threshold = returns_df["return"].quantile(0.8)
returns_df["label"] = (returns_df["return"] >= threshold).astype(int)

# 保存结果
returns_df.to_csv("labeled_returns.csv", index=False)
print("已保存 labeled_returns.csv")
