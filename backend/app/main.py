from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import yfinance as yf
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import joblib
import numpy as np
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
from requests import get


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

    time_series = [ts.strftime('%H:%M') for ts in df.index]
    close_series = df['Close'].iloc[:, 0].to_list()  # 如果是 DataFrame

    result={
        "ticker": ticker,
        "latest_price": latest_price,
        "change_percent": change_percent,
        "volume": volume,
        "total_volume": total_volume,
        "time_series": time_series,
        "close_series": close_series,
    }
    print(result)
    return result

class Holding(BaseModel):
    ticker: str
    shares: float
    buy_price: float

@app.post("/stock/profit/batch")
def batch_stock_profit(holdings: list[Holding]):
    results = []

    for h in holdings:
        try:
            df = yf.download(h.ticker, period='1d', interval='1m', progress=False)
            if df.empty or 'Close' not in df.columns:
                raise ValueError(f"无法获取 {h.ticker} 的数据")

            current_price = df['Close'].dropna().iloc[-1]
            cost = h.shares * h.buy_price
            current_value = h.shares * current_price
            profit = current_value - cost
            return_pct = (profit / cost) * 100

            results.append({
                "ticker": h.ticker.upper(),
                "shares": h.shares,
                "buy_price": h.buy_price,
                "current_price": round(float(current_price), 2),
                "cost": round(float(cost), 2),
                "current_value": round(float(current_value), 2),
                "profit": round(float(profit), 2),
                "return_pct": round(float(return_pct), 2)
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{h.ticker} 查询失败: {str(e)}")

    return results



@app.get("/stock/score")
def stock_score(ticker: str = Query(..., description="股票代码")):
    # 加载特征名、scaler
    feature_names = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook", "bookValue", "beta",
        "dividendYield", "earningsGrowth", "revenueGrowth", "totalRevenue",
        "grossMargins", "operatingMargins", "profitMargins", "returnOnAssets", "returnOnEquity"
    ]
    print("🔧 加载 scaler 中...")
    scaler = joblib.load("/Users/hlshen/Desktop/Nus_SmartFinView/scaler.pkl")

    # 定义模型结构
    class StockRegressor(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)

    try:
        info = yf.Ticker(ticker).info
        for key in feature_names[:5]:
            print(f"  - {key}: {info.get(key, None)}")

        # 初始特征提取
        features = [info.get(f, None) for f in feature_names]
        print(features)

        # 加载训练集并计算均值
        df = pd.read_csv("/Users/hlshen/Desktop/Nus_SmartFinView/dataset/training_dataset.csv")
        feature_means = df[feature_names].mean(numeric_only=True).to_dict()

        # 用均值填补空缺或非法值
        features = [
            feature_means[f] if (v is None or not isinstance(v, (int, float)) or pd.isna(v)) else v
            for v, f in zip(features, feature_names)
        ]

        # 标准化
        x_scaled = scaler.transform([features])

        # 转换为 tensor
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        # 加载模型
        model = StockRegressor(input_dim=len(feature_names))
        model.load_state_dict(torch.load("/Users/hlshen/Desktop/Nus_SmartFinView/score_model.pt", map_location='cpu'))
        model.eval()

        # 预测
        with torch.no_grad():
            score = model(x_tensor).item()

        return {
            "ticker": ticker.upper(),
            "score": round(float(score), 4),
            "features": dict(zip(feature_names, features))
        }

    except Exception as e:
        print("❌ 出错：", e)
        raise HTTPException(status_code=400, detail=f"无法获取特征或模型预测失败: {str(e)}")
    

@app.get("/stock/emotion")
def stock_emotion(keyword: str):
    API_KEY = "e33e6d925ce416416e1ee5b44ce8c7b9"
    url = f"https://gnews.io/api/v4/search?q={keyword}&lang=en&max=50&token={API_KEY}"
    try:
        response = get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Google News API 请求失败: {response.text}")
        data = response.json()
        articles = data.get("articles", [])
        news_texts = [article["title"] + ". " + article.get("description", "") for article in articles]
        print(f"抓取到{len(news_texts)}条新闻")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取新闻失败: {str(e)}")

    analyzer = SentimentIntensityAnalyzer()
    results = []
    all_words = []
    for text in news_texts:
        score = analyzer.polarity_scores(text)
        results.append(score)
        words = [w.lower() for w in text.split() if w.isalpha()]
        all_words.extend(words)
    word_freq = Counter(all_words).most_common(30)
    avg_compound = sum(r['compound'] for r in results) / len(results) if results else 0

    if avg_compound >= 0.5:
        level = "Very Positive"
        suggestion = "🔥 极度正面情绪，市场过热，建议保持谨慎"
    elif avg_compound >= 0.15:
        level = "Positive"
        suggestion = "✅ 偏正面情绪，信心增强，可适当关注买入机会"
    elif avg_compound >= -0.15:
        level = "Neutral"
        suggestion = "⚖️ 情绪中性，建议观望，等待更明确信号"
    elif avg_compound >= -0.5:
        level = "Negative"
        suggestion = "⚠️ 市场悲观，宜谨慎观望或小仓位试探"
    else:
        level = "Very Negative"
        suggestion = "❗ 恐慌情绪显著，关注潜在反转机会"
    
    return {
        "keyword": keyword,
        "news_count": len(news_texts),
        "avg_compound": round(avg_compound, 4),
        "emotion_level": level,
        "suggestion": suggestion,
        "top_words": [{"word": w, "count": c} for w, c in word_freq]
    }
