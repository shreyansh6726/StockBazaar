import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# 1. Load the API key from .env
load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_KEY')

# List of companies provided
symbols = [
    "AAPL", "MSFT", "GOOG", "META", "NVDA", "TSM", "AVGO", "ADBE", "CSCO", "ORCL", 
    "NFLX", "AMD", "ASML", "AMZN", "TSLA", "WMT", "COST", "HD", "PG", "KO", 
    "PEP", "MCD", "LVMUY", "NSRGY", "2222.SR", "XOM", "CVX", "SHEL", "BP", "VALE", 
    "JPM", "V", "MA", "BAC", "WFC", "BRK.B", "HSBC", "GS", "AXP", "TCEHY", 
    "JNJ", "LLY", "UNH", "MRK", "PFE", "NVO", "AZN", "ROG", "TOYOF", "BA", 
    "GE", "MMM", "CAT", "DDAIF", "SIE"
]

def show_menu():
    print("\n--- Select a Company to Check Share Price ---")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i:2}. {symbol}", end="\t" if i % 5 != 0 else "\n")
    print("\n")

def get_user_choice():
    while True:
        try:
            choice = int(input(f"Enter the number (1-{len(symbols)}) of the company: "))
            if 1 <= choice <= len(symbols):
                return symbols[choice - 1]
            else:
                print("Please choose a number within the valid range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# 2. Execution
show_menu()
selected_symbol = get_user_choice()

print(f"\nFetching data for {selected_symbol}...")

url = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': selected_symbol,
    'apikey': api_key
}

response = requests.get(url, params=params)
data = response.json()

# 3. Visualization
time_series = data.get('Time Series (Daily)', {})

if time_series:
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.columns = [col.split('. ')[1] for col in df.columns]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label=f'{selected_symbol} Close Price', color='green', linewidth=2)
    plt.title(f'Stock Performance: {selected_symbol}', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
else:
    print("Error: Could not retrieve data. The API might be rate-limited or the key is missing.")