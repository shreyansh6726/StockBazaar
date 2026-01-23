import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta, date

# --- Configuration ---
AVAILABLE_TICKERS = ["AAPL", "MSFT", "GOOG", "META", "NVDA", "TSM", "AVGO", "ADBE", "CSCO", "ORCL", "NFLX", "AMD", "ASML", "AMZN", "TSLA", "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD", "LVMUY", "NSRGY", "2222.SR", "XOM", "CVX", "SHEL", "BP", "VALE", "JPM", "V", "MA", "BAC", "WFC", "BRK.B", "HSBC", "GS", "AXP", "TCEHY", "JNJ", "LLY", "UNH", "MRK", "PFE", "NVO", "AZN", "ROG", "TOYOF", "BA", "GE", "MMM", "CAT", "DDAIF", "SIE"]
LOOKBACK_DAYS = 60
DAYS_IN_RANGES = {'1 day': 1, '1 week': 7, '1 month': 30, '1 year': 365}

def get_filenames(ticker):
    return f'lstm_{ticker}_model.h5', f'minmax_{ticker}_scaler.pkl'

def load_and_preprocess_data(ticker):
    filename = f"{ticker}_historical_data.csv"
    if os.path.exists(filename):
        print(f"📂 Loading local data for {ticker}...")
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        print(f"🌐 Data not found locally. Downloading {ticker}...")
        data = yf.download(ticker, start='2015-01-01', end='2024-01-01', progress=False)
    
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, Y = [], []
    for i in range(LOOKBACK_DAYS, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK_DAYS:i, 0])
        Y.append(scaled_data[i, 0])
        
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y, scaler

def build_and_train_model(X_train, Y_train, model_filename):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOKBACK_DAYS, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0) # Lower epochs for speed
    model.save(model_filename)
    return model

def ensure_model_is_trained(ticker, model_filename, scaler_filename):
    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        model = load_model(model_filename)
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    else:
        X_train, Y_train, scaler = load_and_preprocess_data(ticker)
        model = build_and_train_model(X_train, Y_train, model_filename)
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
        return model, scaler

def get_latest_data(ticker):
    # Try to get latest from yfinance for real-time start point
    data = yf.download(ticker, period="3mo", progress=False)
    return data[['Close']].tail(LOOKBACK_DAYS)

def predict_future(model, scaler, last_sequence_data, num_days_to_predict):
    scaled_sequence = scaler.transform(last_sequence_data)
    current_input = scaled_sequence.reshape(1, LOOKBACK_DAYS, 1)
    future_predictions = []
    
    for _ in range(num_days_to_predict):
        pred = model.predict(current_input, verbose=0)
        future_predictions.append(pred[0, 0])
        current_input = np.append(current_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()