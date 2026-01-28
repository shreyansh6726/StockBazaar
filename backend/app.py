import os
from flask import Flask, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

CORS(app, origins=["https://stock-bazaar-one.vercel.app", "http://localhost:3000"])

API_KEY = os.getenv('ALPHA_VANTAGE_KEY')

@app.route('/api/stock/<symbol>')
def get_stock(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY
    }
    try:
        response = requests.get('https://www.alphavantage.co/query', params=params)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health_check():
    return "Backend is running!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)