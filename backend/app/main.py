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
import quandl
from sklearn.ensemble import RandomForestRegressor
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime, timedelta
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import traceback
import torch
import joblib
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re


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

# 配置Quandl API（需用户自行注册获取密钥）
quandl.ApiConfig.api_key = "sJpDP9EsiAb6FBCNBmUU"


# 1. 投资组合推荐接口相关模型与实现
class UserPreference(BaseModel):
    risk_tolerance: str  # "low", "medium", "high"
    industry_preference: list[str] = []  # 如 ["Technology", "Healthcare"]
    investment_amount: float
    lookback_period: int = 365  # 历史数据天数


from matplotlib import use as mpl_use
mpl_use('Agg')  # 设置为非 GUI 后端
import matplotlib.pyplot as plt

@app.post("/stock/recommendation", description="根据用户偏好生成投资组合推荐")
def generate_recommendation(preference: UserPreference):
    try:
        industry_stocks = {
            "Technology": ["AAPL", "MSFT", "GOOGL"],
            "Healthcare": ["JNJ", "PFE", "MRNA"],
            "Finance": ["JPM", "BAC", "GS"],
            "Energy": ["XOM", "CVX", "COP"]
        }

        selected_tickers = []
        for industry in preference.industry_preference:
            if industry in industry_stocks:
                selected_tickers.extend(industry_stocks[industry])
        if not selected_tickers:
            selected_tickers = ["AAPL", "MSFT", "AMZN", "JNJ", "XOM", "JPM"]
        selected_tickers = list(set(selected_tickers))

        end_date = datetime.now()
        start_date = end_date - timedelta(days=preference.lookback_period)

        stock_data = yf.download(selected_tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data = stock_data['Adj Close'] if 'Adj Close' in stock_data.columns.get_level_values(0) else stock_data['Close']
        else:
            stock_data = stock_data['Adj Close'] if 'Adj Close' in stock_data.columns else stock_data['Close']

        sp500_data = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=False, progress=False)
        if isinstance(sp500_data.columns, pd.MultiIndex):
            sp500 = sp500_data['Adj Close'] if 'Adj Close' in sp500_data.columns.get_level_values(0) else sp500_data['Close']
        else:
            sp500 = sp500_data['Adj Close'] if 'Adj Close' in sp500_data.columns else sp500_data['Close']

        if stock_data.empty or sp500.empty:
            raise ValueError("获取的数据为空，请检查股票代码或时间范围")

        returns = stock_data.pct_change().dropna()
        if len(returns) < 2:
            raise ValueError("数据不足，无法训练模型")

        X = returns.shift(1).dropna()
        y = returns.iloc[1:]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        latest_returns = returns.iloc[-1:].values
        if latest_returns.shape[1] != len(returns.columns):
            raise ValueError("输入维度不匹配，可能由于数据缺失")

        predicted_returns = model.predict(latest_returns)[0]
        expected_returns_dict = dict(zip(returns.columns, predicted_returns))

        mu = expected_returns.mean_historical_return(stock_data)
        S = risk_models.sample_cov(stock_data)

        ef = EfficientFrontier(mu, S)
        if preference.risk_tolerance == "low":
            weights = ef.min_volatility()
        elif preference.risk_tolerance == "high":
            weights = ef.max_sharpe()
        else:
            weights = ef.efficient_risk(target_volatility=0.30)

        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)

        portfolio_returns = (returns * cleaned_weights).sum(axis=1)
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        cumulative_sp500 = (1 + sp500.pct_change().dropna()).cumprod()

        max_drawdown = (cumulative_portfolio.cummax() - cumulative_portfolio).max()
        sharpe_ratio = performance[2]

        latest_prices = get_latest_prices(stock_data)
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=preference.investment_amount)
        allocation, leftover = da.lp_portfolio()

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_portfolio.index, cumulative_portfolio, label="Recommended Portfolio")
        plt.plot(cumulative_sp500.index, cumulative_sp500, label="S&P500")
        plt.title("Portfolio vs S&P500 Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return {
            "recommended_assets": [
                {"ticker": t, "weight": round(w * 100, 2)}
                for t, w in cleaned_weights.items() if w > 0
            ],
            "discrete_allocation": allocation,
            "leftover_cash": round(leftover, 2),
            "performance_metrics": {
                "sharpe_ratio": round(sharpe_ratio, 2),
                "max_drawdown": round(max_drawdown, 4),
                "expected_annual_return": round(performance[0] * 100, 2),
                "expected_annual_volatility": round(performance[1] * 100, 2)
            },
            "backtest_chart": plot_data
        }

    except Exception as e:
        print("❌ 异常信息：", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")