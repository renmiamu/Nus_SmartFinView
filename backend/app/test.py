from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import yfinance as yf
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import joblib
import numpy as np
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
import traceback
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
    prev_close = yf.Ticker(ticker).info.get('previousClose')
    if prev_close is None:
        return {"error": "无法获取前收盘价"}
    
    change_percent = (latest_price - prev_close) / prev_close * 100
    volume = df['Volume'].iloc[-1].item()
    total_volume = df['Volume'].sum().item()

    time_series = [ts.strftime('%H:%M') for ts in df.index]
    close_series = df['Close'].tolist()

    result = {
        "ticker": ticker,
        "latest_price": latest_price,
        "change_percent": change_percent,
        "volume": volume,
        "total_volume": total_volume,
        "time_series": time_series,
        "close_series": close_series,
    }
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
            return_pct = (profit / cost) * 100 if cost != 0 else 0

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
    feature_names = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook", "bookValue", "beta",
        "dividendYield", "earningsGrowth", "revenueGrowth", "totalRevenue",
        "grossMargins", "operatingMargins", "profitMargins", "returnOnAssets", "returnOnEquity"
    ]
    
    try:
        scaler = joblib.load("scaler.pkl")

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

        info = yf.Ticker(ticker).info

        df = pd.read_csv("dataset/training_dataset.csv")
        feature_means = df[feature_names].mean(numeric_only=True).to_dict()
        features = [
            feature_means[f] if (v is None or not isinstance(v, (int, float)) or pd.isna(v)) else v
            for v, f in zip([info.get(f) for f in feature_names], feature_names)
        ]

        x_scaled = scaler.transform([features])
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        
        model = StockRegressor(input_dim=len(feature_names))
        model.load_state_dict(torch.load("score_model.pt", map_location='cpu'))
        model.eval()
        
        with torch.no_grad():
            score = model(x_tensor).item()

        return {
            "ticker": ticker.upper(),
            "score": round(float(score), 4),
            "features": dict(zip(feature_names, features))
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法获取特征或模型预测失败: {str(e)}")


@app.get("/stock/emotion")
def stock_emotion(keyword: str):
    API_KEY = "e33e6d925ce416416e1ee5b44ce8c7b9"
    url = f"https://gnews.io/api/v4/search?q={keyword}&lang=en&max=50&token={API_KEY}"
    
    try:
        response = get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"新闻API请求失败: {response.text}")
        
        articles = response.json().get("articles", [])
        
        def clean_text(text):
            return re.sub(r'[^\w\s.]', '', text).strip()
        
        news_texts = [clean_text(f"{a['title']}. {a.get('description', '')}") for a in articles]
        analyzer = SentimentIntensityAnalyzer()
        results = [analyzer.polarity_scores(text) for text in news_texts]
        
        avg_compound = sum(r['compound'] for r in results) / len(results) if results else 0
        word_freq = Counter([w.lower() for text in news_texts for w in text.split() if w.isalpha()]).most_common(30)

        if avg_compound >= 0.5:
            level, suggestion = "Very Positive", "🔥 极度正面情绪，市场过热，建议保持谨慎"
        elif avg_compound >= 0.15:
            level, suggestion = "Positive", "✅ 偏正面情绪，信心增强，可适当关注买入机会"
        elif avg_compound >= -0.15:
            level, suggestion = "Neutral", "⚖️ 情绪中性，建议观望，等待更明确信号"
        elif avg_compound >= -0.5:
            level, suggestion = "Negative", "⚠️ 市场悲观，宜谨慎观望或小仓位试探"
        else:
            level, suggestion = "Very Negative", "❗ 恐慌情绪显著，关注潜在反转机会"
        
        return {
            "keyword": keyword,
            "news_count": len(news_texts),
            "avg_compound": round(avg_compound, 4),
            "emotion_level": level,
            "suggestion": suggestion,
            "top_words": [{"word": w, "count": c} for w, c in word_freq]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取新闻失败: {str(e)}")


class UserPreference(BaseModel):
    risk_tolerance: str  # "low", "medium", "high"
    industry_preference: list[str] = []
    investment_amount: float
    lookback_period: int = 365


from matplotlib import use as mpl_use
mpl_use('Agg')
import matplotlib.pyplot as plt


def extract_price_data(df, tickers):
    """
    提取价格数据，兼容单只/多只股票，优先使用Adj Close，无则用Close
    tickers: 股票代码列表（用于单只股票转DataFrame时的列名）
    """
    if df.empty:
        raise ValueError("获取的股票数据为空")
    
    # 处理多列（多只股票）：列是MultiIndex（level0: 指标，level1: 股票代码）
    if isinstance(df.columns, pd.MultiIndex):
        available_indicators = df.columns.get_level_values(0).unique()
        target_indicator = "Adj Close" if "Adj Close" in available_indicators else "Close"
        
        price_data = df[target_indicator]
        price_data = price_data.reindex(columns=tickers).dropna(axis=1, how='all')
        
        if price_data.empty:
            raise ValueError(f"数据中缺少 {target_indicator} 列")
        
        return price_data
    
    # 处理单列（单只股票）：返回的是Series，转为DataFrame
    else:
        if "Adj Close" in df.columns:
            return df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in df.columns:
            return df[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise ValueError("数据中缺少Adj Close或Close列")


@app.post("/stock/recommendation")
def generate_recommendation(preference: UserPreference):
    try:
        if preference.risk_tolerance not in ["low", "medium", "high"]:
            raise HTTPException(status_code=400, detail="风险容忍度必须为 'low'、'medium' 或 'high'")

        industry_stocks = {
            "Technology": ["AAPL", "MSFT", "GOOGL"],
            "Healthcare": ["JNJ", "PFE", "MRNA"],
            "Finance": ["JPM", "BAC", "GS"],
            "Energy": ["XOM", "CVX", "COP"]
        }

        selected_tickers = list(set(
            [t for ind in preference.industry_preference if ind in industry_stocks for t in industry_stocks[ind]]
            or ["AAPL", "MSFT", "AMZN", "JNJ", "XOM", "JPM"]
        ))

        end_date = datetime.now()
        start_date = end_date - timedelta(days=preference.lookback_period)
        
        # 显式设置auto_adjust=False，确保数据格式稳定
        stock_data = yf.download(
            selected_tickers, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False
        )
        sp500_data = yf.download(
            "^GSPC", 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False
        )

        # 提取价格数据
        stock_data = extract_price_data(stock_data, selected_tickers)
        sp500 = extract_price_data(sp500_data, ["^GSPC"])["^GSPC"]  # 转为Series

        # 调试信息
        print(f"股票数据格式: {type(stock_data)}, 形状: {stock_data.shape}")
        print(f"股票数据列名: {list(stock_data.columns)}")
        print(f"标普500数据格式: {type(sp500)}, 长度: {len(sp500)}")

        if stock_data.empty or sp500.empty:
            raise HTTPException(status_code=400, detail="获取股票数据为空，请检查股票代码或时间范围")

        # 计算收益率
        returns = stock_data.pct_change().dropna()
        if len(returns) < 2:
            raise ValueError("数据不足，无法生成组合")

        # 构建模型
        X = returns.shift(1).dropna()
        y = returns.iloc[1:]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        latest_returns = returns.iloc[-1:].values
        if latest_returns.shape[1] != len(returns.columns):
            raise ValueError("输入维度不匹配，可能由于数据缺失")

        predicted_returns = model.predict(latest_returns)[0]
        expected_returns_dict = dict(zip(returns.columns, predicted_returns))

        # 投资组合优化
        mu = expected_returns.mean_historical_return(stock_data)
        S = risk_models.sample_cov(stock_data)

        ef = EfficientFrontier(mu, S)
        if preference.risk_tolerance == "low":
            weights = ef.min_volatility()
        elif preference.risk_tolerance == "high":
            weights = ef.max_sharpe()
        else:
            max_sharpe_volatility = ef.portfolio_performance()[1]  # 获取最大夏普组合的波动率
            weights = ef.efficient_risk(target_volatility=max_sharpe_volatility * 1.2)

        cleaned_weights = ef.clean_weights()
        if not any(cleaned_weights.values()):
            raise ValueError("无法生成有效投资组合，请调整参数")
        
        performance = ef.portfolio_performance(verbose=False)
        latest_prices = get_latest_prices(stock_data)
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=preference.investment_amount)
        allocation, leftover = da.lp_portfolio()

        # 回测图表
        portfolio_returns = (returns * cleaned_weights).sum(axis=1)
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        cumulative_sp500 = (1 + sp500.pct_change().dropna()).cumprod()

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_portfolio.index, cumulative_portfolio, label="推荐组合")
        plt.plot(cumulative_sp500.index, cumulative_sp500, label="标普500")
        plt.title("组合与标普500累计收益对比")
        plt.xlabel("日期")
        plt.ylabel("累计收益")
        plt.legend()
        plt.grid(True)
        
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "recommended_assets": [{"ticker": t, "weight": round(w*100, 2)} for t, w in cleaned_weights.items() if w > 0],
            "discrete_allocation": allocation,
            "leftover_cash": round(leftover, 2),
            "performance_metrics": {
                "sharpe_ratio": round(performance[2], 2),
                "max_drawdown": round((cumulative_portfolio.cummax() - cumulative_portfolio).max(), 4),
                "expected_annual_return": round(performance[0]*100, 2),
                "expected_annual_volatility": round(performance[1]*100, 2)
            },
            "backtest_chart": plot_data
        }

    except Exception as e:
        print(f"推荐接口错误: {str(e)}")
        print(traceback.format_exc())  # 打印完整堆栈信息
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")