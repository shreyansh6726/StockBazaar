# 🚗 Tesla Stock Price Predictor  

> **Predict future Tesla (TSLA) stock prices using 10 years of historical data and an advanced LSTM neural network — integrated into a native Android app for real-time forecasting.**

---

## 🧭 Overview  

This project leverages **Machine Learning** and a **decade’s worth of Tesla stock market data (2014–2023)** to forecast future prices.  
At its heart lies an **LSTM (Long Short-Term Memory)** neural network — a model particularly effective for **time-series forecasting** tasks such as stock prediction.  

The trained model is integrated into a **native Android app** using **TensorFlow Lite**, providing real-time and on-the-go analysis with a clean, intuitive user interface.

---

## ✨ Features  

- 🔮 **Predicts Tesla’s upcoming stock price** based on historical trends.  
- 🧠 **Uses LSTM architecture** for accurate sequence forecasting.  
- 📊 **Visualizes predicted vs actual prices** for better insights.  
- 📱 **Native Android app** with TensorFlow Lite integration.  
- 🎨 **Minimalist & modern UI** following a blue/cyan color palette.  

---

## 📂 Dataset  

- **Source:** Tesla historical stock prices (2014–2023)  
- **Format:** CSV file containing  
Date | Open | High | Low | Close | Volume

markdown
Copy code
- **Preprocessing:**  
- Handled missing values  
- Normalized the closing price  
- Split into training/testing datasets  
- Reshaped into 60-day input windows for LSTM  

---

## 🧩 Approach & Methodology  

### 1. Data Collection & Preprocessing  
- Fetched daily **TSLA** data from sources like *Yahoo Finance* or *Alpha Vantage*.  
- Handled missing values and normalized closing prices.  
- Shaped input sequences: each input = last 60 days, output = next day’s price.

### 2. Model Design: LSTM Architecture  
- **Input Layer** → **LSTM Layer(s)** → **Dropout Layer** → **Dense Output Layer**  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Frameworks:** TensorFlow / Keras  

### 3. Prediction & Evaluation  
- Tested the model on **unseen 2023 data**.  
- Evaluated using **Root Mean Square Error (RMSE)**.  
- Visualized predictions vs. actual stock prices.  

### 4. Android Integration  
- Converted trained `.h5` model to `.tflite` for mobile deployment.  
- Integrated **TensorFlow Lite Interpreter** for real-time inference.  
- Designed a **clean, modern UI** in **Android Studio** (Java/Kotlin).  
- Added animated “Start” button, tagline, and intuitive navigation.  

---

## 🧑‍💻 Usage  

### 🔹 Running the Model (Python/Notebook)
1. Open and run `stock_predictor.ipynb`.  
2. The notebook will:
 - Train the model  
 - Generate prediction plots  
 - Export files:  
   - `stock_lstm.h5`  
   - `stock_lstm.tflite`  

### 🔹 Mobile Deployment (Android)
1. Place the `stock_lstm.tflite` file in:  
android-app/assets/

markdown
Copy code
2. Use **TensorFlow Lite Interpreter** to load and run predictions.  
3. Input recent Tesla price data → view prediction → visualize results.  

---

## 🎨 Android UI Design  

**Color Palette:**  
| Element | Color |
|----------|-------|
| Primary | `#5585b5` |
| Accent  | `#53a8b6` |
| Soft Blue | `#79c2d0` |
| Light Cyan | `#bbe4e9` |

**UI Highlights:**  
- Central **animated “Start” button**  
- **Main image asset:** `main.png`  
- **Modern, minimalist layout**  
- **Error/status messages** for clarity  

---

## ⚙️ Required Dependencies  

### 🧠 Python (Model Training)


pandas
numpy
matplotlib
scikit-learn
tensorflow
keras
📱 Android
bash
Copy code
TensorFlow Lite
Android Studio (Java/Kotlin)
Glide / Picasso
Standard Android libraries
📁 Folder Structure
css
Copy code
project/
├── stock_predictor.ipynb
├── data/
│   └── tesla_2014-2023.csv
├── model/
│   └── stock_lstm.tflite
├── android-app/
│   ├── assets/
│   │   └── stock_lstm.tflite
│   └── src/...
├── main.png
└── README.md
🚀 How to Reproduce
Clone this repository

bash
Copy code
```
git clone https://github.com/your-username/tesla-stock-predictor.git
cd tesla-stock-predictor
```
Run the Notebook

Train the LSTM model

Export .tflite model

Set up the Android project

Add model file in assets

Sync dependencies in Android Studio

Build and Run

Deploy on a physical Android device or emulator

📚 References
LSTM for Time Series Forecasting (TensorFlow)

TensorFlow Lite Android Deployment Guide

Tesla Stock Data - Yahoo Finance

🪪 License
This project is licensed under the MIT License — feel free to use, modify, and distribute.
See the LICENSE file for details.

🧠 Built with Passion using TensorFlow + Android Studio
💡 Empowering AI-driven financial insights — anywhere, anytime.
