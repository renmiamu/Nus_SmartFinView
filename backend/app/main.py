from fastapi import FastAPI, Query
import pandas as pd
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stock/basic")
def stock_basic(ticker: str):
    df = yf.download(ticker, period='1d', interval='1m')
    if df.empty or 'Close' not in df.columns:
        return {"error": "未找到该股票数据"}

    latest_price = df['Close'].iloc[-1].item()
    prev_close = yf.Ticker(ticker).info['previousClose']
    change_percent = (latest_price - prev_close) / prev_close * 100
    volume = df['Volume'].iloc[-1].item()
    total_volume = df['Volume'].sum().item()
    result={
        "ticker": ticker,
        "latest_price": latest_price,
        "change_percent": change_percent,
        "volume": volume,
        "total_volume": total_volume
    }
    print(result)
    return result

@app.get("/stock/profit")
def stock_profit(ticker: str, shares: float, buy_price: float):
    df = yf.download(ticker, period='1d', interval='1m')
    current_price = df['Close'].iloc[-1].item()
    cost = shares * buy_price
    current_value = shares * current_price
    profit = current_value - cost
    return_pct = (profit / cost) * 100
    return {
        "ticker": ticker,
        "shares": shares,
        "buy_price": buy_price,
        "current_price": current_price,
        "cost": cost,
        "current_value": current_value,
        "profit": profit,
        "return_pct": return_pct
    }