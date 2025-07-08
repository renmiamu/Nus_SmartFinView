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


# 初始化FastAPI应用
app = FastAPI(title="SmartFinView API", description="整合投资组合推荐与股票分析功能")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置Quandl API（需用户自行注册获取密钥）
quandl.ApiConfig.api_key = "sJpDP9EsiAb6FBCNBmUU"


# 1. 投资组合推荐接口相关模型与实现
class UserPreference(BaseModel):
    risk_tolerance: str  # "low", "medium", "high"
    industry_preference: list[str] = []  # 如 ["Technology", "Healthcare"]
    investment_amount: float
    lookback_period: int = 365  # 历史数据天数


@app.post("/investment/recommendation", description="根据用户偏好生成投资组合推荐")
def generate_recommendation(preference: UserPreference):
    try:
        # 筛选符合行业偏好的股票池
        industry_stocks = {
            "Technology": ["AAPL", "MSFT", "GOOGL"],
            "Healthcare": ["JNJ", "PFE", "MRNA"],
            "Finance": ["JPM", "BAC", "GS"],
            "Energy": ["XOM", "CVX", "COP"]
        }
        
        # 根据用户偏好筛选股票
        selected_tickers = []
        for industry in preference.industry_preference:
            if industry in industry_stocks:
                selected_tickers.extend(industry_stocks[industry])
        # 若未选行业，默认使用大盘股
        if not selected_tickers:
            selected_tickers = ["AAPL", "MSFT", "AMZN", "JNJ", "XOM", "JPM"]
        selected_tickers = list(set(selected_tickers))  # 去重

        # 获取历史数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=preference.lookback_period)
        
        print(f"开始获取历史数据，股票: {selected_tickers}，时间范围: {start_date} 至 {end_date}")
        
        # 从Yahoo Finance获取价格数据
        stock_data = yf.download(
            selected_tickers, 
            start=start_date, 
            end=end_date,
            auto_adjust=False,  # 显式关闭自动调整
            progress=False
        )
        
        print(f"获取到的股票数据形状: {stock_data.shape}")
        print(f"获取到的股票数据列名: {stock_data.columns}")
        
        # 处理Adj Close列（兼容单只/多只股票的索引结构）
        if isinstance(stock_data.columns, pd.MultiIndex):
            # 多只股票：列是多层索引（如(Adj Close, AAPL)）
            if 'Adj Close' in stock_data.columns.get_level_values(0):
                stock_data = stock_data['Adj Close']  # 提取所有股票的Adj Close
            else:
                # 若没有Adj Close，用Close列替代
                stock_data = stock_data['Close']
                print("警告：未找到Adj Close列，已使用Close列替代")
        else:
            # 单只股票：列是单层索引
            if 'Adj Close' in stock_data.columns:
                stock_data = stock_data['Adj Close']
            else:
                stock_data = stock_data['Close']
                print("警告：未找到Adj Close列，已使用Close列替代")
        
        # 获取S&P500数据
        sp500_data = yf.download(
            "^GSPC", 
            start=start_date, 
            end=end_date,
            auto_adjust=False,
            progress=False
        )
        
        print(f"获取到的S&P500数据形状: {sp500_data.shape}")
        print(f"获取到的S&P500数据列名: {sp500_data.columns}")
        
        if isinstance(sp500_data.columns, pd.MultiIndex):
            sp500 = sp500_data['Adj Close'] if 'Adj Close' in sp500_data.columns.get_level_values(0) else sp500_data['Close']
        else:
            sp500 = sp500_data['Adj Close'] if 'Adj Close' in sp500_data.columns else sp500_data['Close']
        
        # 验证数据有效性
        if stock_data.empty:
            raise ValueError("获取的股票数据为空，请检查股票代码或时间范围")
        if sp500.empty:
            raise ValueError("获取的S&P500数据为空")
        
        print(f"最终股票数据形状: {stock_data.shape}")
        print(f"最终S&P500数据形状: {sp500.shape}")
        
        # 训练回归模型预测预期收益率
        returns = stock_data.pct_change().dropna()
        if len(returns) < 2:
            raise ValueError("数据点不足，无法训练模型，请增加lookback_period")
        
        X = returns.shift(1).dropna()  # 前一天收益率作为特征
        y = returns.iloc[1:]  # 当天收益率作为目标
        
        print(f"训练数据形状 - X: {X.shape}, y: {y.shape}")
        
        # 训练随机森林回归模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 预测未来短期预期收益率
        latest_returns = returns.iloc[-1:].values
        predicted_returns = model.predict(latest_returns)[0]
        expected_returns_dict = dict(zip(returns.columns, predicted_returns))
        
        # 使用Markowitz优化算法生成最优组合
        mu = expected_returns.mean_historical_return(stock_data)
        S = risk_models.sample_cov(stock_data)
        
        print(f"预期收益率形状: {mu.shape}")
        print(f"协方差矩阵形状: {S.shape}")
        
        # 根据风险承受能力调整优化目标
        ef = EfficientFrontier(mu, S)
        if preference.risk_tolerance == "low":
            weights = ef.min_volatility()  # 最小波动率
        elif preference.risk_tolerance == "high":
            weights = ef.max_sharpe()  # 最大夏普比率
        else:
            # 目标函数接收两个参数w和self（优化器实例）
            weights = ef.nonconvex_objective(
                lambda w, self: -self.expected_return(w) / self.volatility(w),  # 使用self调用方法
                initial_guess=np.array([1/len(mu)]*len(mu))
            )
        
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        
        print(f"优化后权重: {cleaned_weights}")
        print(f"组合表现: 预期收益率={performance[0]:.4f}, 波动率={performance[1]:.4f}, 夏普比率={performance[2]:.4f}")
        
        # 回测表现（与S&P500对比）
        portfolio_returns = (returns * cleaned_weights).sum(axis=1)
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        cumulative_sp500 = (1 + sp500.pct_change().dropna()).cumprod()
        
        # 计算风险指标
        max_drawdown = (cumulative_portfolio.cummax() - cumulative_portfolio).max()
        sharpe_ratio = performance[2]
        
        print(f"最大回撤: {max_drawdown:.4f}, 夏普比率: {sharpe_ratio:.4f}")
        
        # 离散分配（根据投资金额分配股票数量）
        latest_prices = get_latest_prices(stock_data)
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=preference.investment_amount)
        allocation, leftover = da.lp_portfolio()
        
        print(f"离散分配结果: {allocation}, 剩余现金: {leftover:.2f}")
        
        # 绘制回测对比图
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_portfolio.index, cumulative_portfolio, label="Recommended Portfolio")
        plt.plot(cumulative_sp500.index, cumulative_sp500, label="S&P500")
        plt.title("Portfolio vs S&P500 Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        
        # 保存图表为base64编码供前端展示
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        
        print("成功生成回测图表")
        
        return {
            "recommended_assets": [
                {"ticker": ticker, "weight": round(weight*100, 2)} 
                for ticker, weight in cleaned_weights.items() if weight > 0
            ],
            "discrete_allocation": allocation,
            "leftover_cash": round(leftover, 2),
            "performance_metrics": {
                "sharpe_ratio": round(sharpe_ratio, 2),
                "max_drawdown": round(max_drawdown, 4),
                "expected_annual_return": round(performance[0]*100, 2),
                "expected_annual_volatility": round(performance[1]*100, 2)
            },
            "backtest_chart": plot_data  # base64编码的图表
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"发生异常: {str(e)}")
        print(f"完整堆栈信息: {error_trace}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


# 2. 股票基础信息接口
@app.get("/stock/basic", description="获取股票实时基础信息（价格、成交量等）")
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
    close_series = df['Close'].to_list()  # 兼容单只股票数据结构

    result = {
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


# 3. 批量收益计算接口相关模型与实现
class Holding(BaseModel):
    ticker: str
    shares: float
    buy_price: float


@app.post("/stock/profit/batch", description="批量计算持仓收益")
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


# 4. 股票评分接口
@app.get("/stock/score", description="基于财务指标的股票评分预测")
def stock_score(ticker: str = Query(..., description="股票代码")):
    # 加载特征名、scaler
    feature_names = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook", "bookValue", "beta",
        "dividendYield", "earningsGrowth", "revenueGrowth", "totalRevenue",
        "grossMargins", "operatingMargins", "profitMargins", "returnOnAssets", "returnOnEquity"
    ]
    print("🔧 加载 scaler 中...")
    scaler = joblib.load(r"C:\software\Nus_SmartFinView\scaler.pkl")

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
        df = pd.read_csv(r"C:\software\Nus_SmartFinView\dataset\training_dataset.csv")
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
        model.load_state_dict(torch.load(r"C:\software\Nus_SmartFinView\score_model.pt", map_location='cpu'))
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


# 5. 股票情绪分析接口
@app.get("/stock/emotion", description="基于Twitter的股票情绪分析")
def stock_emotion(keyword: str):
    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAITI2wEAAAAA2LljYwgcTLpwwatxzyXzYK%2F9Qos%3DsfnzN8gkDp91qqOyCEFQqCNSvuuS0RdXHUfTBcRxb6HeKcYKfe"
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    query = f"{keyword} lang:en -is:retweet"
    try:
        tweets = client.search_recent_tweets(query=query, max_results=50, tweet_fields=["text"])
        tweet_texts = [tweet.text for tweet in tweets.data] if tweets.data else []
        print(f"抓取到{len(tweet_texts)}条推文")
    except tweepy.TooManyRequests:
        raise HTTPException(status_code=429, detail="Twitter API 请求过多，请稍后再试")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取推文失败: {str(e)}")

    analyzer = SentimentIntensityAnalyzer()
    results = []
    all_words = []
    for text in tweet_texts:
        score = analyzer.polarity_scores(text)
        results.append(score)
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', text)]
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
        "tweet_count": len(tweet_texts),
        "avg_compound": round(avg_compound, 4),
        "emotion_level": level,
        "suggestion": suggestion,
        "top_words": [{"word": w, "count": c} for w, c in word_freq]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)