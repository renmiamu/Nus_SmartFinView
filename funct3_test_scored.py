import yfinance as yf
import pandas as pd

# 读取符号列表
df = pd.read_excel("dataset/test.xlsx")
symbols = df.iloc[:, 0].dropna().unique().tolist()
print(f"共有 {len(symbols)} 个股票代码")

# 抓取收益率
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

# 转为 DataFrame
returns_df = pd.DataFrame(returns_data)

# 添加打分（0~1）
min_ret = returns_df["return"].min()
max_ret = returns_df["return"].max()
returns_df["score"] = (returns_df["return"] - min_ret) / (max_ret - min_ret)

# 保存结果（只保留 symbol, return, score）
returns_df[["symbol", "return", "score"]].to_csv("dataset/test_scored_returns.csv", index=False)
print("✅ 已保存 test_scored_returns.csv（只包含得分）")
