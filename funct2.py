import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = input("Enter the ticker symbol: ")
shares = float(input("Enter the number of shares: "))
buy_price = float(input("Enter the buy price: "))

df = yf.download(ticker, period='1d', interval='1m')
current_price = df['Close'].iloc[-1].item()

cost= shares * buy_price
current_value = shares * current_price
profit = current_value - cost
return_pct = (profit / cost) * 100

