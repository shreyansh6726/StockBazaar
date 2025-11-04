Tesla Stock Price Predictor
A machine learning project that predicts Tesla (TSLA) stock prices using LSTM neural networks trained on 10 years of historical market data (2014-2023).

Overview
This project uses Long Short-Term Memory (LSTM) neural networks to forecast Tesla stock prices. The model analyzes historical price patterns, technical indicators, and market trends to predict future closing prices.

Features
LSTM-based prediction model trained on 10 years of Tesla stock data
Technical indicators including RSI, CCI, MACD, Bollinger Bands, ATR, and moving averages
Model evaluation with RMSE, MAE, R² score, and MAPE metrics
Visualization of predicted vs actual stock prices
Ready for deployment to Android apps using TensorFlow Lite
Dataset
The dataset includes:

Daily stock prices (Open, High, Low, Close, Volume)
Technical indicators (RSI-7, RSI-14, CCI-7, CCI-14)
Moving averages (SMA-50, EMA-50, SMA-100, EMA-100)
MACD, Bollinger Bands, True Range, ATR
Target variable: next day's closing price
Data spans from 2014 to 2023, providing approximately 2,500+ trading days for training and testing.

Installation
Clone the repository:
git clone https://github.com/yourusername/StockBazaar.git
cd StockBazaar
Install dependencies:
pip install -r requirements.txt
Usage
Training the Model
Open implementation.ipynb in Jupyter Notebook
Run all cells to:
Load and preprocess the data
Create sequences for LSTM input (60-day windows)
Train the LSTM model
Generate predictions and visualizations
Save the trained model
Model Architecture
Input Layer: 60-day sequences of features
LSTM Layer 1: 100 units with return_sequences=True
Dropout: 0.2
LSTM Layer 2: 100 units
Dropout: 0.2
Dense Layer: 50 units with ReLU activation
Output Layer: 1 unit (predicted price)
Evaluating the Model
Run analytics.ipynb to calculate:

Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R² Score
Mean Absolute Percentage Error (MAPE)
Model Performance
The model achieves:

RMSE: $34.01 (14.30% of average price)
MAE: $26.89 (11.31% of average price)
R² Score: 0.588 (58.80% variance explained)
MAPE: 12.60%
Percentage Accuracy: 87.40%
Directional Accuracy: 49.90%
Dependencies
pandas >= 1.5.3
numpy >= 1.24.3
matplotlib >= 3.7.1
scikit-learn >= 1.2.2
tensorflow >= 2.10.0
h5py >= 3.8.0
Project Structure
StockBazaar/
├── implementation.ipynb    # Main training notebook
├── analytics.ipynb         # Model evaluation metrics
├── dataset.csv             # Historical Tesla stock data
├── tesla_lstm_model.h5     # Trained LSTM model
├── model_analytics.png     # Analytics visualization
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
└── .gitignore             # Git ignore rules
Note: The trained model (tesla_lstm_model.h5) and analytics visualization (model_analytics.png) are included in this repository. If you want to retrain the model, run implementation.ipynb.

Android Deployment
To deploy the model for Android:

Convert the trained model to TensorFlow Lite:

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('stock_lstm.tflite', 'wb') as f:
    f.write(tflite_model)
Add the .tflite file to your Android app's assets folder

Use TensorFlow Lite Interpreter in your Android app to load and run predictions

Notes
Stock price prediction is inherently uncertain due to market volatility
The model is trained on historical data and may not account for sudden market changes
Use predictions as one of many factors in investment decisions
Always validate predictions with current market conditions
License
This project is licensed under the MIT License - see the LICENSE file for details.

References
LSTM for Time Series Forecasting
TensorFlow Lite Documentation
Yahoo Finance - Historical stock data source
