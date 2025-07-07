import yfinance as yf
import pandas as pd

symbols = [
    "PLTR", "SQ", "UBER", "LYFT", "RIVN", "LCID", "DASH", "COIN", "HOOD", "SOFI",
    "NET", "OKTA", "ZM", "DOCU", "ROKU", "TWLO", "CRWD", "DDOG", "MDB", "SNOW",
    "SHOP", "ETSY", "MELI", "W", "BILI", "IQ", "JD", "BABA", "PDD", "NTES",
    "TME", "BIDU", "WB", "YMM", "SE", "GDS", "BEKE", "ZTO", "KWEB", "FXI",
    "ARKK", "ARKG", "ARKF", "ARKW", "ICLN", "TAN", "LIT", "BOTZ", "ROBO", "QQQ",
    "SPY", "VOO", "DIA", "IWM", "VT", "EWZ", "RSX", "EFA", "EEM", "TLT",
    "IEF", "HYG", "JNK", "BND", "AGG", "XLF", "XLK", "XLY", "XLC", "XLV",
    "XLE", "XLI", "XLB", "XLRE", "XLU", "FANG", "DVN", "OXY", "MPC", "PSX",
    "EOG", "APA", "MRO", "HES", "HAL", "SLB", "VLO", "CTRA", "AR", "PXD",
    "NEM", "GOLD", "WPM", "AEM", "FNV", "AA", "FCX", "SCCO", "TECK", "RIO"
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
df.to_csv("fundamental_data.csv", index=False)
print("✅ 基本面数据已保存为 fundamental_data.csv")
