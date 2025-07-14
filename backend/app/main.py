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
        return {"error": "Stock data not found"}

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
                raise ValueError(f"Can't get the data of {h.ticker} ")

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
            raise HTTPException(status_code=400, detail=f"{h.ticker} Query failed: {str(e)}")

    return results



@app.get("/stock/score")
def stock_score(ticker: str = Query(..., description="Ticker")):
    # 加载特征名、scaler
    feature_names = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook", "bookValue", "beta",
        "dividendYield", "earningsGrowth", "revenueGrowth", "totalRevenue",
        "grossMargins", "operatingMargins", "profitMargins", "returnOnAssets", "returnOnEquity"
    ]
    print("🔧 Loading scaler...")
    scaler = joblib.load("../scaler.pkl")

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
        df = pd.read_csv("../dataset/training_dataset.csv")
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
        model.load_state_dict(torch.load("../score_model.pt", map_location='cpu'))
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
        print("❌ Wrong：", e)
        raise HTTPException(status_code=400, detail=f"Failed to obtain features or model prediction failed: {str(e)}")
    

@app.get("/stock/emotion")
def stock_emotion(keyword: str):
    API_KEY = "e33e6d925ce416416e1ee5b44ce8c7b9"
    url = f"https://gnews.io/api/v4/search?q={keyword}&lang=en&max=50&token={API_KEY}"
    try:
        response = get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Google News API request failed: {response.text}")
        data = response.json()
        articles = data.get("articles", [])
        news_texts = [article["title"] + ". " + article.get("description", "") for article in articles]
        print(f"{len(news_texts)}news have been retrieved")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieved the news: {str(e)}")

    analyzer = SentimentIntensityAnalyzer()
    results = []
    all_words = []
    news_items = []
    # 英文常见停用词（可根据需要扩展）
    stopwords = set([
        'the', 'and', 'for', 'are', 'but', 'not', 'with', 'you', 'all', 'any', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'by', 'is', 'it', 'as', 'be', 'or', 'if', 'from', 'that', 'this', 'so', 'do', 'no', 'up', 'down', 'off', 'over', 'under', 'then', 'than', 'into', 'about', 'after', 'before', 'because', 'between', 'during', 'through', 'while', 'where', 'when', 'which', 'what', 'whose', 'whom', 'been', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'shall', 'their', 'there', 'them', 'these', 'those', 'such', 'very', 'just', 'also', 'more', 'most', 'some', 'other', 'only', 'own', 'same', 'each', 'few', 'both', 'many', 'much', 'every', 'any', 'again', 'against', 'once', 'here', 'why', 'how', 'he', 'she', 'they', 'we', 'i', 'me', 'my', 'mine', 'your', 'yours', 'his', 'hers', 'its', 'ours', 'theirs', 'am', 'were', 'being', 'doing', 'having', 'does', 'did', 'done', 'having', 'have', 'had', 'been', 'was', 'is', 'are', 'were', 'do', 'does', 'did', 'has', 'have', 'had', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could'
    ])
    for article in articles:
        title = article["title"]
        desc = article.get("description", "")
        published = article.get("publishedAt", "")
        text = title + ". " + desc
        score = analyzer.polarity_scores(text)
        results.append(score)
        # 修正：用正则提取英文单词，避免 isalpha() 过滤过严
        words = re.findall(r"[a-zA-Z]+", text.lower())
        # 过滤停用词
        words = [w for w in words if w not in stopwords]
        all_words.extend(words)
        news_items.append({
            "title": title,
            "description": desc,
            "publishedAt": published,
            "compound": score["compound"]
        })
    word_freq = Counter(all_words).most_common(30)
    avg_compound = sum(r['compound'] for r in results) / len(results) if results else 0

    if avg_compound >= 0.5:
        level = "Very Positive"
        suggestion = "🔥 Extremely positive sentiment, overheated market, it is recommended to remain cautious."
    elif avg_compound >= 0.15:
        level = "Positive"
        suggestion = "✅ With a relatively positive sentiment and enhanced confidence, one may appropriately focus on buying opportunities."
    elif avg_compound >= -0.15:
        level = "Neutral"
        suggestion = "⚖️ Neutral sentiment. It is recommended to observe and wait for more definite signals."
    elif avg_compound >= -0.5:
        level = "Negative"
        suggestion = "⚠️ The market is pessimistic. It is advisable to adopt a cautious approach and either wait and observe or take a small-scale trial."
    else:
        level = "Very Negative"
        suggestion = "❗ The panic sentiment is significant. One can focus on potential reversal opportunities."
    
    return {
        "keyword": keyword,
        "news_count": len(news_items),
        "avg_compound": round(avg_compound, 4),
        "emotion_level": level,
        "suggestion": suggestion,
        "top_words": [{"word": w, "count": c} for w, c in word_freq],
        "news_items": news_items
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

@app.post("/stock/recommendation", description="Generate investment portfolio recommendations based on user preferences")
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
            raise ValueError("The obtained data is empty. Please check the stock code or the time range.")

        returns = stock_data.pct_change().dropna()
        if len(returns) < 2:
            raise ValueError("Insufficient data, unable to train the model")

        X = returns.shift(1).dropna()
        y = returns.iloc[1:]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        latest_returns = returns.iloc[-1:].values
        if latest_returns.shape[1] != len(returns.columns):
            raise ValueError("The input dimensions do not match, possibly due to missing data.")

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
        print("❌ Exception：", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")