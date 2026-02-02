import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from twelvedata import TDClient
from sklearn.linear_model import LinearRegression

load_dotenv()
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
td = TDClient(apikey=API_KEY)

app = Flask(__name__)
CORS(app)

def get_fast_prediction(symbol, tenure):
    # Mapping tenure to intervals
    config = {
        "1h": {"interval": "1min", "outputsize": 100},
        "1d": {"interval": "1h", "outputsize": 100},
        "1w": {"interval": "1day", "outputsize": 100}
    }
    
    selected = config.get(tenure, config["1h"])

    # 1. Fetch Data (Smaller size for speed)
    ts = td.time_series(
        symbol=symbol,
        interval=selected["interval"],
        outputsize=selected["outputsize"]
    ).as_pandas()

    df = ts.sort_index(ascending=True)
    df['Time_Index'] = np.arange(len(df))
    
    # 2. Linear Regression (Instantly fast)
    X = df[['Time_Index']].values
    y = df['close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the next point in the series
    next_index = np.array([[len(df)]])
    predicted_price = model.predict(next_index)[0]

    return {
        "metadata": {"symbol": symbol, "tenure": tenure, "model": "Fast-Linear"},
        "prediction": {
            "last_close": float(df['close'].iloc[-1]),
            "predicted_price": round(float(predicted_price), 2),
            "trend": "UP" if predicted_price > df['close'].iloc[-1] else "DOWN"
        }
    }

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', 'AAPL').upper()
    tenure = request.args.get('tenure', '1h').lower()
    try:
        result = get_fast_prediction(symbol, tenure)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)