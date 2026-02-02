import os
import numpy as np
import pandas as pd
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from twelvedata import TDClient
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Setup & Config
load_dotenv()
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
td = TDClient(apikey=API_KEY)

app = Flask(__name__)
CORS(app)  # Allows your Vercel frontend to talk to this Render backend

def get_prediction_logic(symbol, tenure):
    # Map tenure to Twelve Data intervals
    config = {
        "1h": {"interval": "1min", "outputsize": 500},
        "1d": {"interval": "1h", "outputsize": 500},
        "1w": {"interval": "1day", "outputsize": 500}
    }
    
    selected_config = config.get(tenure, config["1h"])

    # Fetch Data
    ts = td.time_series(
        symbol=symbol,
        interval=selected_config["interval"],
        outputsize=selected_config["outputsize"],
        timezone="America/New_York"
    ).as_pandas()

    # Preprocessing
    df = ts.sort_index(ascending=True)
    data = df.filter(['close']).values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Sequence Creation
    prediction_days = 60
    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # LSTM Model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Prediction
    real_data = [scaled_data[len(scaled_data) - prediction_days:len(scaled_data), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    return {
        "metadata": {
            "symbol": symbol,
            "tenure": tenure,
            "model": "LSTM"
        },
        "prediction": {
            "last_close": float(df['close'].iloc[-1]),
            "predicted_price": round(float(predicted_price), 2),
            "trend": "UP" if predicted_price > df['close'].iloc[-1] else "DOWN"
        }
    }

# 2. API Routes
@app.route('/predict', methods=['GET'])
def predict():
    # Use request.args to get parameters from the URL
    symbol = request.args.get('symbol', 'AAPL').upper()
    tenure = request.args.get('tenure', '1h').lower()
    
    try:
        result = get_prediction_logic(symbol, tenure)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 3. Entry Point for Render
if __name__ == "__main__":
    # Port is dynamic for cloud deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)