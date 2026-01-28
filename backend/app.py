# app.py
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app) # Allows React to talk to Flask

API_KEY = os.getenv('ALPHA_VANTAGE_KEY')
BASE_URL = 'https://www.alphavantage.co/query'

@app.route('/api/stock/<symbol>')
def get_stock(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True, port=5000)