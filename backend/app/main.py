from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import joblib
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
import traceback
from transformers import pipeline
import string


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
    close_series = df['Close'].iloc[:, 0].to_list()  # If it's a DataFrame

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
                raise ValueError(f"Cannot retrieve data for {h.ticker}")

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
            raise HTTPException(status_code=400, detail=f"{h.ticker} query failed: {str(e)}")

    return results



@app.get("/stock/score")
def stock_score(ticker: str = Query(..., description="Ticker symbol")):
    # Load feature names and scaler
    feature_names = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook", "bookValue", "beta",
        "dividendYield", "earningsGrowth", "revenueGrowth", "totalRevenue",
        "grossMargins", "operatingMargins", "profitMargins", "returnOnAssets", "returnOnEquity"
    ]
    print("🔧 Loading scaler...")
    scaler = joblib.load("../scaler.pkl")

    class StockRegressor(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(128, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                
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

        # Initial feature extraction
        features = [info.get(f, None) for f in feature_names]
        print(features)

        # Load training dataset and calculate means
        df = pd.read_csv("../dataset/training_dataset.csv")
        feature_means = df[feature_names].mean(numeric_only=True).to_dict()

        # Fill missing or invalid values with means
        features = [
            feature_means[f] if (v is None or not isinstance(v, (int, float)) or pd.isna(v)) else v
            for v, f in zip(features, feature_names)
        ]

        # Normalization
        x_scaled = scaler.transform([features])

        # Convert to tensor
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        # Load model
        model = StockRegressor(input_dim=len(feature_names))
        model.load_state_dict(torch.load("../score_model.pt", map_location='cpu'))
        model.eval()

        # Prediction
        with torch.no_grad():
            score = model(x_tensor).item()

        return {
            "ticker": ticker.upper(),
            "score": round(float(score), 4),
            "features": dict(zip(feature_names, features))
        }

    except Exception as e:
        print("❌ Error：", e)
        raise HTTPException(status_code=400, detail=f"Failed to retrieve features or model prediction failed: {str(e)}")
    

# Custom VADER lexicon (financial domain examples, can be expanded)
CUSTOM_VADER_LEXICON = {
    'bullish': 2.0,
    'bearish': -2.0,
    'overvalued': -1.5,
    'undervalued': 1.5,
    'upgrade': 1.2,
    'downgrade': -1.2,
    'profit warning': -2.0,
    'record high': 1.8,
    'record low': -1.8,
    'missed estimates': -1.5,
    'beats estimates': 1.5,
    'dividend cut': -1.5,
    'dividend increase': 1.5,
}

def preprocess_text(text):
    # Remove special characters, normalize to lowercase, remove extra spaces
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
        print(f"{len(news_texts)} news articles retrieved")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve news: {str(e)}")

    # Initialize VADER with custom lexicon
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(CUSTOM_VADER_LEXICON)

    # Initialize BERT sentiment analysis pipeline
    bert_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    results = []
    all_words = []
    news_items = []
    stopwords = set([
        's', 'y', 'the', 'and', 'for', 'are', 'but', 'not', 'with', 'you', 'all', 'any', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'by', 'is', 'it', 'as', 'be', 'or', 'if', 'from', 'that', 'this', 'so', 'do', 'no', 'up', 'down', 'off', 'over', 'under', 'then', 'than', 'into', 'about', 'after', 'before', 'because', 'between', 'during', 'through', 'while', 'where', 'when', 'which', 'what', 'whose', 'whom', 'been', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'shall', 'their', 'there', 'them', 'these', 'those', 'such', 'very', 'just', 'also', 'more', 'most', 'some', 'other', 'only', 'own', 'same', 'each', 'few', 'both', 'many', 'much', 'every', 'any', 'again', 'against', 'once', 'here', 'why', 'how', 'he', 'she', 'they', 'we', 'i', 'me', 'my', 'mine', 'your', 'yours', 'his', 'hers', 'its', 'ours', 'theirs', 'am', 'were', 'being', 'doing', 'having', 'does', 'did', 'done', 'having', 'have', 'had', 'been', 'was', 'is', 'are', 'were', 'do', 'does', 'did', 'has', 'have', 'had', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could'
    ])
    for article in articles:
        title = article["title"]
        desc = article.get("description", "")
        published = article.get("publishedAt", "")
        text = title + ". " + desc
        pre_text = preprocess_text(text)
        # VADER score
        vader_score = analyzer.polarity_scores(pre_text)
        # BERT score (returns label and score, label is 1-5 star(s), 5 stars most positive)
        bert_result = bert_sentiment(pre_text[:512])[0]  # BERT max input 512 characters
        bert_label = bert_result['label']
        match = re.search(r"([1-5])", bert_label)
        if match:
            bert_compound = (int(match.group(1)) - 3) / 2  # 1 star = -1, 3 stars = 0, 5 stars = 1
        else:
            bert_compound = 0
        results.append({"vader": vader_score, "bert": bert_compound})
        # Word segmentation and stopword filtering
        words = re.findall(r"[a-zA-Z]+", pre_text)
        words = [w for w in words if w not in stopwords]
        all_words.extend(words)
        news_items.append({
            "title": title,
            "description": desc,
            "publishedAt": published,
            "compound_vader": vader_score["compound"],
            "compound_bert": bert_compound
        })
    word_freq = Counter(all_words).most_common(30)
    avg_compound_vader = sum(r['vader']['compound'] for r in results) / len(results) if results else 0
    avg_compound_bert = sum(r['bert'] for r in results) / len(results) if results else 0

    # Comprehensive sentiment score (can be weighted average, simple average here)
    avg_compound = (avg_compound_vader + avg_compound_bert) / 2

    if avg_compound >= 0.5:
        level = "Very Positive"
        suggestion = "🔥 Extremely positive sentiment, overheated market, it is recommended to remain cautious."
    elif avg_compound >= 0.15:
        level = "Positive"
        suggestion = "✅ With relatively positive sentiment and enhanced confidence, one may appropriately focus on buying opportunities."
    elif avg_compound >= -0.15:
        level = "Neutral"
        suggestion = "⚖️ Neutral sentiment. It is recommended to observe and wait for more definite signals."
    elif avg_compound >= -0.5:
        level = "Negative"
        suggestion = "⚠️ The market is pessimistic. It is advisable to adopt a cautious approach and either wait and observe or take a small-scale trial."
    else:
        level = "Very Negative"
        suggestion = "❗ Significant panic sentiment. One can focus on potential reversal opportunities."
    
    return {
        "keyword": keyword,
        "news_count": len(news_items),
        "avg_compound": round(avg_compound, 4),
        "avg_compound_vader": round(avg_compound_vader, 4),
        "avg_compound_bert": round(avg_compound_bert, 4),
        "emotion_level": level,
        "suggestion": suggestion,
        "top_words": [{"word": w, "count": c} for w, c in word_freq],
        "news_items": news_items
    }

# Configure Quandl API (users need to register for an API key)
quandl.ApiConfig.api_key = "sJpDP9EsiAb6FBCNBmUU"

# Expand industry stock pool - 100 stocks per industry
industry_stocks = {
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "PYPL", "ADBE", "NFLX",
        "INTC", "CSCO", "ORCL", "IBM", "HPQ", "DELL", "CRM", "ACN", "AVGO", "QCOM",
        "TXN", "AMD", "MU", "LUMN", "AMAT", "ADI", "MRVL", "MCHP", "INTU",
        "ANSS", "ADSK", "CDNS", "SNPS", "ZS", "CRWD", "PANW", "MDB", "DBX", 
        "ZM", "TWTR", "PINS", "ETSY", "ROKU", "SPOT", "DDOG", "OKTA", "DOCU", "TEAM",
        "NET", "WDAY",  "SNOW", "ADSK", "ASML", "LRCX", "KLAC", "TER", "GLW",
        "IPGP", "FSLR", "ENPH", "SEDG", "MASI", "CTSH", "INFY", "WIT", 
        "TECH", "AKAM", "FFIV", "JNPR", "ZBRA", "HPE", "STX",  "NTAP", "VRSN",
        "VRSK", "SWKS", "QRVO", "INVE", "OLED", "SYNA", "ON", "ADI", "NXPI",
        "LITE", "FLEX", "JBL", "GOOGL", "GOOG", "BIDU", "NTES", "PDD"
    ],
    "Healthcare": [
        "JNJ", "PFE", "MRNA", "LLY", "ABBV", "MRK", "TMO", "UNH", "ABT", "AMGN",
        "GILD", "VRTX", "REGN", "DHR", "BDX", "ISRG", "MDT", "SYK", "BAX", "HCA",
        "CVS", "ANTM", "CI", "LH", "HOLX", "IDXX", "ILMN", "ALGN", "VRTX", "BIIB",
        "CELG", "REGN", "SGEN", "INCY", "AMLN", "VRTX", "ALXN", "RPRX", "PRGO", "MYL",
        "TEVA", "AGN", "ZTS", "LLY", "ELAN", "ENDP", "MNK", "ZBH", "WAT", "A",
        "DGX", "IQV", "CERN", "MCK", "CAH", "ABC", "WBA", "RAD", "HUM", "CNC",
        "ESRX", "AET", "ANTM", "CI", "HCSG", "UHS", "THC", "LPNT", "PDCO", "MHS",
        "CRL", "WCG", "HRC", "CPSI", "CHSI", "AMMD", "OMCL", "MEDP", "NSTG", "SRDX",
        "PDLI", "IMGN", "AVEO", "ARNA", "AGIO", "AXGN", "BLUE", "CARA", "CLVS", "CYTR",
        "DRRX", "EDIT", "EXEL", "FATE", "FOLD", "GERN", "HCAT", "IMMU", "KURA", "MRTX"
    ],
    "Finance": [
        "JPM", "BAC", "GS", "MS", "C", "WFC", "USB", "PNC", "COF", "AXP",
        "BLK", "SCHW", "CME", "ICE", "MSCI", "SPGI", "MCO", "TRV", "MET", "PRU",
        "AIG", "ALL", "HIG", "PGR", "UNM", "CINF", "AIZ", "LNC", "GL", "CNA",
        "CB", "WRB", "BHF", "BRK-B", "BK", "STT", "TROW", "AMP", "BEN",
        "IVZ", "JEF", "LPLA",  "ALLY", "FITB", "KEY", "MTB", "RF", "HBAN",
        "ZION", "CFG", "TFC",   "FRC", "CMA", "WAL", "FHN",
        "TCBI", "UMBF", "WSBC", "CPF", "CVBF", "FNB", "FISI", "HBCP", 
         "OFIX", "OFG",  "PB",   "PRA", 
        "RF", "RWT", "SBNY",  "SIVB", "TCBI", "TD", "TRMK", 
        "USB", "WABC", "WAL", "WBS", "WFC",  "WTFC", "ZION", "ALLY", "COF"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "OXY", "EOG", "PXD", "MPC", "VLO", "PSX", "HES",
        "KMI", "ET", "PAA", "EPD", "ENB", "TRP", "XEL", "DUK", "SO", "NEE",
        "AEP", "EXC", "CMS", "ES", "ED", "PEG", "SRE", "XRAY", "HAL", "SLB",
        "BKR", "NOV", "OII", "FTI", "HP", "BHI", "NE", "RIG", "VAL", "WLL",
        "APC", "APA", "CHK", "CXO", "DVN", "FANG", "MRO", "NBL", "OXY", "PXD",
        "QEP", "SWN", "WMB", "COG", "EQT", "RRC", "AR", "BTU", "CLR", "CRZO",
        "CVX", "DVN", "EOG", "EQT", "FANG", "HAL", "HES", "HFC", "KMI", "MPC",
        "MRO", "NBL", "NOV", "OXY", "OKE", "PSX", "PXD", "RIG", "SLB", "SO",
        "SWN", "TOT", "TRP", "VLO", "WMB", "XEC", "XOM", "XRAY", "APA", "APC",
        "BHI", "CHK", "COP", "COG", "CVX", "DVN", "EOG", "EQT", "FANG", "HAL"
    ]
}

# 1. Portfolio recommendation API related models and implementation
class UserPreference(BaseModel):
    risk_tolerance: str  # "low", "medium", "high"
    industry_preference: list[str] = []  # e.g. ["Technology", "Healthcare"]
    investment_amount: float
    lookback_period: int = 1825  # Number of days for historical data


from matplotlib import use as mpl_use
mpl_use('Agg')  # Set to non-GUI backend


@app.post("/stock/recommendation", description="Generate portfolio recommendations based on user preferences")
def generate_recommendation(preference: UserPreference):
    try:
        selected_tickers = []
        for industry in preference.industry_preference:
            if industry in industry_stocks:
                selected_tickers.extend(industry_stocks[industry])
                
        if not selected_tickers:
            for industry in ["Technology", "Healthcare", "Finance", "Energy"]:
                selected_tickers.extend(industry_stocks[industry])
            
        # Ensure unique stock list
        selected_tickers = list(set(selected_tickers))

        end_date = datetime.now()
        start_date = end_date - timedelta(days=preference.lookback_period)

        # Download stock data in batches to avoid API request limits
        stock_data_parts = []
        batch_size = 30
        for i in range(0, len(selected_tickers), batch_size):
            batch_tickers = selected_tickers[i:i+batch_size]
            batch_data = yf.download(batch_tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
            if isinstance(batch_data.columns, pd.MultiIndex):
                batch_data = batch_data['Adj Close'] if 'Adj Close' in batch_data.columns.get_level_values(0) else batch_data['Close']
            else:
                batch_data = batch_data['Adj Close'] if 'Adj Close' in batch_data.columns else batch_data['Close']
            stock_data_parts.append(batch_data)
        
        # Merge all batches of data
        stock_data = pd.concat(stock_data_parts, axis=1)
        
        # Download S&P 500 data as benchmark
        sp500_data = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=False, progress=False)
        if isinstance(sp500_data.columns, pd.MultiIndex):
            sp500 = sp500_data['Adj Close'] if 'Adj Close' in sp500_data.columns.get_level_values(0) else sp500_data['Close']
        else:
            sp500 = sp500_data['Adj Close'] if 'Adj Close' in sp500_data.columns else sp500_data['Close']

        if stock_data.empty or sp500.empty:
            raise ValueError("Obtained data is empty, please check stock codes or time range")

        # Data preprocessing - handle missing values
        stock_data = stock_data.dropna(thresh=int(len(stock_data) * 0.5), axis=1)  # Remove columns with too many missing values
        if stock_data.empty:
            raise ValueError("Insufficient valid stock data, please adjust industry preferences or time range")
        stock_data = stock_data.ffill()  # Forward fill missing values
        
        returns = stock_data.pct_change().dropna()
        if len(returns) < 20:  # Increase data point requirement
            raise ValueError("Insufficient data to train the model")

        max_lag = 5  # Increase lag features to 5 days

        def engineer_features(returns_df):
            # Original lag return features
            lag_features = []
            for i in range(1, max_lag+1):
                shifted = returns_df.shift(i)
                shifted.columns = [f"{col}_lag_{i}" for col in returns_df.columns]
                lag_features.append(shifted)
    
            # Add moving average features
            ma_short = returns_df.rolling(window=5).mean()
            ma_short.columns = [f"{col}_ma_5" for col in returns_df.columns]
    
            ma_medium = returns_df.rolling(window=20).mean()
            ma_medium.columns = [f"{col}_ma_20" for col in returns_df.columns]
    
            ma_long = returns_df.rolling(window=60).mean()
            ma_long.columns = [f"{col}_ma_60" for col in returns_df.columns]
    
            # Add volatility features
            volatility = returns_df.rolling(window=20).std()
            volatility.columns = [f"{col}_vol" for col in returns_df.columns]
    
            # Combine all features
            all_features = pd.concat([*lag_features, ma_short, ma_medium, ma_long, volatility], axis=1)
    
            return all_features.dropna()

        # Use improved feature engineering in main function
        X = engineer_features(returns)
        y = returns.loc[X.index]

        model = RandomForestRegressor(
            n_estimators=5000,  # Increase number of trees
            max_depth=20,      # Limit tree depth to prevent overfitting
            min_samples_split=20,  # Increase minimum samples for splitting
            min_samples_leaf=5,    # Increase minimum samples for leaf nodes
            max_features='sqrt',   # Reduce number of features considered per tree
            bootstrap=True,             # Whether to use bootstrap sampling
            oob_score=True,             # Whether to calculate out-of-bag error
            n_jobs=-1,             # Use all CPU cores
            random_state=42
        )

        model.fit(X, y)

        latest_data = returns.iloc[-max_lag-60:]  # Get enough data to calculate all features
        latest_features = engineer_features(latest_data)  # Use the same feature engineering function

        # Ensure prediction features match training features
        if set(latest_features.columns) != set(X.columns):
            # Add missing features (if any) filled with 0
            missing_features = [col for col in X.columns if col not in latest_features.columns]
            for col in missing_features:
                latest_features[col] = 0
    
            # Order features as in training
            latest_features = latest_features[X.columns]

        # Use model to predict returns
        predicted_returns = model.predict(latest_features)[0]
        expected_returns_dict = {}
        for i, ticker in enumerate(returns.columns):
            expected_returns_dict[ticker] = predicted_returns[i]

        # Portfolio optimization
        mu = expected_returns.mean_historical_return(stock_data, frequency=252, compounding=True)
        # Use Ledoit-Wolf shrinkage estimator to handle high-dimensional covariance matrix
        ew_data = stock_data.ewm(span=365).mean()
        # Use Ledoit-Wolf shrinkage to estimate covariance
        S = risk_models.CovarianceShrinkage(ew_data).ledoit_wolf()

        ef = EfficientFrontier(mu, S)  # Maximum 50% weight for individual stocks
        if preference.risk_tolerance == "low":
            weights = ef.min_volatility()
        elif preference.risk_tolerance == "high":
            weights = ef.max_sharpe()
        else:
            weights = ef.efficient_risk(target_volatility=0.1)

        cleaned_weights = ef.clean_weights(cutoff=0.01)  # Remove stocks with too small weights
        performance = ef.portfolio_performance(verbose=False)

        # Calculate portfolio returns and benchmark returns
        portfolio_returns = (returns * cleaned_weights).sum(axis=1)
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        cumulative_sp500 = (1 + sp500.pct_change().dropna()).cumprod()

        # Calculate risk metrics
        max_drawdown = (cumulative_portfolio.cummax() - cumulative_portfolio).max()
        risk_free_daily = 0.03 / 252  # Assume 3% annual risk-free rate, converted to daily rate
        excess_returns = portfolio_returns - risk_free_daily
        annual_sharpe = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))


        # Get latest prices and calculate discrete allocation
        latest_prices = stock_data.iloc[-1]
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=preference.investment_amount)
        allocation, leftover = da.lp_portfolio()

        # Plot backtest chart
        plt.figure(figsize=(12, 8))
        plt.plot(cumulative_portfolio.index, cumulative_portfolio, label="Recommended Portfolio", linewidth=2)
        plt.plot(cumulative_sp500.index, cumulative_sp500, label="S&P500", linewidth=2)
        plt.title("Portfolio vs S&P500 Cumulative Returns", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Cumulative Return", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save chart as PNG and encode as Base64
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plot_data = base64.b64encode(buf.read())
        buf.close()

        return {
            "recommended_assets": [
                {"ticker": t, "weight": round(w * 100, 2)}
                for t, w in cleaned_weights.items() if w > 0
            ],
            "discrete_allocation": allocation,
            "leftover_cash": round(leftover, 2),
            "performance_metrics": {
                "sharpe_ratio": round(annual_sharpe, 2),
                "max_drawdown": round(max_drawdown, 4),
                "expected_annual_return": round(performance[0] * 100, 2),
                "expected_annual_volatility": round(performance[1] * 100, 2)
            },
            "backtest_chart": plot_data
        }

    except Exception as e:
        print("❌ Exception information: ", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")