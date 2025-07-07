import yfinance as yf
import pandas as pd

symbols = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK.B", "JPM", "V",
    "UNH", "XOM", "PG", "MA", "JNJ", "HD", "COST", "BAC", "MRK", "PEP",
    "LLY", "ABBV", "AVGO", "KO", "TMO", "WMT", "CVX", "ORCL", "CRM", "MCD",
    "ACN", "ADBE", "DHR", "NKE", "QCOM", "ABT", "VZ", "MDT", "TXN", "LIN",
    "HON", "BMY", "AMGN", "AMAT", "PFE", "LOW", "SBUX", "MS", "UNP", "T",
    "GE", "UPS", "CAT", "IBM", "NOW", "C", "RTX", "SCHW", "BLK", "BKNG",
    "SPGI", "ISRG", "ADP", "FISV", "ZTS", "MU", "LMT", "INTU", "GILD", "CI",
    "TMUS", "DE", "CB", "MO", "ICE", "REGN", "TGT", "F", "GM", "PYPL",
    "ADI", "BIIB", "CSCO", "NFLX", "PANW", "SNPS", "MAR", "DD", "AZO", "ORLY",
    "EA", "ANET", "KLAC", "HCA", "CDNS", "AON", "EW", "ILMN", "ROP", "DXCM"
]


data = []

for symbol in symbols:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        data.append({
            "symbol": symbol,
            "marketCap": info.get("marketCap", None),
            "trailingPE": info.get("trailingPE", None),
            "forwardPE": info.get("forwardPE", None),
            "priceToBook": info.get("priceToBook", None),
            "bookValue": info.get("bookValue", None),
            "beta": info.get("beta", None),
            "dividendYield": info.get("dividendYield", None),
            "earningsGrowth": info.get("earningsQuarterlyGrowth", None),
            "revenueGrowth": info.get("revenueGrowth", None),
            "totalRevenue": info.get("totalRevenue", None),
            "grossMargins": info.get("grossMargins", None),
            "operatingMargins": info.get("operatingMargins", None),
            "profitMargins": info.get("profitMargins", None),
            "returnOnAssets": info.get("returnOnAssets", None),
            "returnOnEquity": info.get("returnOnEquity", None)
        })
        print(f"{symbol} ✔")
    except Exception as e:
        print(f"{symbol} ✘ Error: {e}")

# 转换为 DataFrame 并保存
df = pd.DataFrame(data)
df.to_csv("test_data.csv", index=False)
print("✅ 测试数据已保存为 test_data.csv")
