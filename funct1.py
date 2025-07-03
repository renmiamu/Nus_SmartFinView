import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = input("Enter the ticker symbol: ")
df = yf.download(ticker, period='1d', interval='1m')

# 实时延迟价格
latest_price = df['Close'].iloc[-1].item()
print("latest price:", latest_price)

# 涨跌幅
prev_close = yf.Ticker(ticker).info['previousClose']
change_percent = (latest_price - prev_close) / prev_close * 100
print(f"change percent: {change_percent:.2f}%")

# 最新成交量（每分钟）
volume = df['Volume'].iloc[-1].item()
print("最新成交量（每分钟）:", volume)

# 今日总成交量
total_volume = df['Volume'].sum().item()
print("今日总成交量:", total_volume)

# 绘制价格变化趋势图
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label='Close Price')
plt.title(f"{ticker} 今日分钟级价格变化")
plt.xlabel("time")
plt.ylabel("price")
plt.legend()
plt.tight_layout()
plt.show()

