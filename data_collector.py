import yfinance as yf
import pandas as pd

TICKER_SYMBOLS = ["AAPL", "MSFT", "GOOG", "META", "NVDA", "TSM", "AVGO", "ADBE", "CSCO", "ORCL", "NFLX", "AMD", "ASML", "AMZN", "TSLA", "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD", "LVMUY", "NSRGY", "2222.SR", "XOM", "CVX", "SHEL", "BP", "VALE", "JPM", "V", "MA", "BAC", "WFC", "BRK.B", "HSBC", "GS", "AXP", "TCEHY", "JNJ", "LLY", "UNH", "MRK", "PFE", "NVO", "AZN", "ROG", "TOYOF", "BA", "GE", "MMM", "CAT", "DDAIF", "SIE"]  # Add your desired ticker symbols here
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"  

for ticker in TICKER_SYMBOLS:
    print(f"Downloading data for {ticker}...")
    # Download historical data for the current ticker
    data = yf.download(ticker, start=START_DATE, end=END_DATE)

    if data.empty:
        print(f"⚠️ No data found for {ticker}, skipping.")
        continue

    # Save the data to a CSV file named after the ticker
    filename = f"{ticker}_historical_data.csv"
    data.to_csv(filename)

    print(f"✅ Data for {ticker} saved to {filename}")