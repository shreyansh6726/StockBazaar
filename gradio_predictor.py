import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta, date
import base64
from io import BytesIO

# --- 1. CONFIGURATION (MUST MATCH PREVIOUS SCRIPT) ---
AVAILABLE_TICKERS = [
    "AAPL", "MSFT", "GOOG", "META", "NVDA", "TSM", "AVGO", "ADBE", "CSCO", "ORCL",
    "NFLX", "AMD", "ASML", "AMZN", "TSLA", "WMT", "COST", "HD", "PG", "KO",
    "PEP", "MCD", "LVMUY", "NSRGY", "2222.SR", "XOM", "CVX", "SHEL", "BP", "VALE",
    "JPM", "V", "MA", "BAC", "WFC", "BRK.B", "HSBC", "GS", "AXP", "TCEHY",
    "JNJ", "LLY", "UNH", "MRK", "PFE", "NVO", "AZN", "ROG", "TOYOF", "BA",
    "GE", "MMM", "CAT", "DDAIF", "SIE"
]

START_DATE = '2015-01-01'
END_DATE = '2024-01-01'
LOOKBACK_DAYS = 60

DAYS_IN_RANGES = {
    '1 day': 1,
    '1 week': 7,
    '1 month': 30,
    '1 year': 365
}

def get_filenames(ticker):
    """Generates unique filenames for the selected ticker."""
    return f'lstm_{ticker}_model.h5', f'minmax_{ticker}_scaler.pkl'

# --- 2. CORE UTILITY FUNCTIONS (Modified for Gradio) ---

def load_and_preprocess_data(ticker):
    """Downloads, scales, and creates sequences for LSTM training."""
    # Data is downloaded in the training function, only preprocessing here
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check the ticker and dates.")
        
    data = data[['Close']]
    
    # Scaling the Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Creating the Dataset Structure
    X, Y = [], []
    for i in range(LOOKBACK_DAYS, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK_DAYS:i, 0])
        Y.append(scaled_data[i, 0])
        
    X, Y = np.array(X), np.array(Y)
    
    # FIX for the previous dimension error: reshape Y into a column vector
    Y = Y.reshape(-1, 1)

    # Reshaping X for LSTM input: [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y, scaler

def build_and_train_model(X_train, Y_train, model_filename):
    """Builds, trains, and saves the LSTM model."""
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOKBACK_DAYS, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1)) 
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model (simplified training for a cleaner UI experience)
    model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)
    
    model.save(model_filename)
    return model

def get_latest_data(ticker):
    """Fetches the most recent data required for the starting sequence."""
    end_date_fetch = date.today() + timedelta(days=1)
    # Increased buffer to ensure 60 trading days are captured
    start_date_fetch = end_date_fetch - timedelta(days=LOOKBACK_DAYS + 40) 
    
    latest_data = yf.download(ticker, start=start_date_fetch, end=end_date_fetch, progress=False)
    latest_data = latest_data[['Close']]
    
    last_sequence_data = latest_data[-LOOKBACK_DAYS:]
    
    if len(last_sequence_data) < LOOKBACK_DAYS:
        raise ValueError(f"Could not fetch enough recent data ({len(last_sequence_data)} days) to form the starting sequence of {LOOKBACK_DAYS} days.")
        
    return last_sequence_data

def predict_future(model, scaler, last_sequence_data, num_days_to_predict):
    """Generates future price predictions iteratively."""
    
    scaled_sequence = scaler.transform(last_sequence_data)
    current_input = scaled_sequence.reshape(1, LOOKBACK_DAYS, 1)
    future_predictions = []
    
    for _ in range(num_days_to_predict):
        predicted_scaled_price = model.predict(current_input, verbose=0)
        future_predictions.append(predicted_scaled_price[0, 0])
        
        # Update the input sequence for the next prediction
        new_scaled_input = np.append(current_input[:, 1:, :], predicted_scaled_price.reshape(1, 1, 1), axis=1)
        current_input = new_scaled_input
    
    # Inverse transform the predictions to get actual prices
    predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return predicted_prices.flatten()

# --- 3. GRADIO INTERFACE LOGIC ---

def run_prediction(ticker, prediction_range, status_output):
    """
    Main function called by Gradio, handles loading/training and prediction.
    It yields status updates for the loading screen.
    """
    try:
        if not ticker or not prediction_range:
            return None, None, gr.update(visible=False), "Please select both a Company and a Time Range."

        # Get filenames for the selected ticker
        model_file, scaler_file = get_filenames(ticker)
        
        model = None
        scaler = None

        # --- A. Load or Train the Model/Scaler ---
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            yield gr.update(value=f"✅ Loading existing model for **{ticker}**..."), None, gr.update(visible=False)
            model = load_model(model_file)
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
        else:
            # BEAUTIFUL LOADING SCREEN: Status Update for Training
            yield gr.update(value=f"⏳ Model not found. Training new model for **{ticker}** (This may take a minute)...", visible=True), None, gr.update(visible=False)
            
            # Training Logic
            X_train, Y_train, scaler = load_and_preprocess_data(ticker)
            model = build_and_train_model(X_train, Y_train, model_file)
            
            # Save the scaler
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            yield gr.update(value=f"🎉 Training complete! Model saved for **{ticker}**.", visible=True), None, gr.update(visible=False)

        # --- B. Prediction ---
        num_days = DAYS_IN_RANGES[prediction_range]
        
        yield gr.update(value=f"🔮 Generating prediction for the next **{prediction_range}** ({num_days} days)...", visible=True), None, gr.update(visible=False)

        # 1. Get last sequence data
        last_sequence_data = get_latest_data(ticker)

        # 2. Predict
        predicted_prices = predict_future(model, scaler, last_sequence_data, num_days)

        # --- C. Display Results ---
        
        # 1. Tabular Data
        last_date = last_sequence_data.index[-1].date()
        future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]
        
        results_df = pd.DataFrame({
            'Date': future_dates,
            f'Predicted Price ({ticker})': [f"${p:.2f}" for p in predicted_prices]
        })
        
        # 2. Graphical Data
        historical_dates = last_sequence_data.index.date
        historical_prices = last_sequence_data['Close'].values

        # *** FIX START ***

        # 1. Ensure historical prices are pure float data
        # If the historical prices have any internal structure (like sequences or objects), 
        # this will flatten them and convert them to float.
        historical_prices = last_sequence_data['Close'].astype(float).values

        # 2. Filter out NaN values, which can disrupt plotting/array creation
        # We keep only the dates and prices where the price is a valid number.
        valid_indices = ~np.isnan(historical_prices)
        historical_dates = historical_dates[valid_indices]
        historical_prices = historical_prices[valid_indices]

        # *** FIX END ***

        plt.figure(figsize=(12, 5))
        plt.plot(historical_dates, historical_prices, label='Historical Price', color='#1f77b4', linewidth=2)
        
        # Plotting the prediction: starts from the last historical point
        plt.plot([historical_dates[-1]] + future_dates, 
                 [historical_prices[-1]] + list(predicted_prices), 
                 label='Predicted Price', color='#d62728', linestyle='--', linewidth=2)
        
        plt.title(f'{ticker} Stock Price Forecast for {prediction_range}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close() # Close figure to free memory
        
        # Convert buffer to base64 for image display (Gradio friendly)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        img_html = f'<img src="data:image/png;base64,{img_data}" style="width: 100%; height: auto; border-radius: 8px;">'

        # Final yield of results
        yield gr.update(value=f"✅ Prediction complete for **{ticker}**!", visible=True), \
              gr.update(value=img_html, visible=True), \
              gr.update(value=results_df, headers=list(results_df.columns), visible=True)

    except Exception as e:
        yield gr.update(value=f"❌ Prediction failed: {e}", visible=True), \
              gr.update(visible=False), \
              gr.update(visible=False)


# --- 4. GRADIO UI DEFINITION ---

# Custom CSS for the "beautiful, minimalist and decent UI"
custom_css = """
/* Theme and general styling */
:root {
    --primary-color: #007BFF; /* Blue */
    --secondary-color: #28a745; /* Green */
    --text-color: #333;
    --background-color: #f8f9fa; /* Light gray */
    --card-background: #ffffff;
    --border-radius: 8px;
}

body {
    background-color: var(--background-color);
}

.gradio-container {
    max-width: 1200px;
    margin: auto;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    border-radius: var(--border-radius);
}

/* Header Styling */
#app-title {
    color: var(--primary-color);
    text-align: center;
    padding-bottom: 20px;
    font-size: 2.5em;
    font-weight: 700;
}

/* Input and Button Styling */
.gr-button {
    background-color: var(--primary-color) !important;
    color: white !important;
    border-radius: var(--border-radius) !important;
    font-weight: bold;
    transition: background-color 0.3s ease;
}

.gr-button:hover {
    background-color: #0056b3 !important;
}

.gr-select, .gr-dropdown {
    border-radius: var(--border-radius) !important;
}

/* Status/Loading Message Styling (Animation Simulation) */
#status-output {
    text-align: center;
    padding: 15px;
    margin-top: 15px;
    border-radius: var(--border-radius);
    font-size: 1.1em;
    font-weight: 600;
    min-height: 50px; /* Ensure space for the message */
    background-color: #fff3cd; /* Light warning color for training */
    color: #856404;
    border: 1px solid #ffeeba;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Table and Plot Containers */
.result-box {
    margin-top: 25px;
    padding: 20px;
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* Tabular Data Styling */
.gr-table {
    border-radius: var(--border-radius);
}

/* Animation for Training (Simulated via yield, but visual effects can be added here) */
/* This spins the icon in the status box */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

#status-output:contains("Training") {
    /* Subtle pulsing effect during training */
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { background-color: #fef7e0; }
    50% { background-color: #fff3cd; }
    100% { background-color: #fef7e0; }
}

"""

with gr.Blocks(title="LSTM Stock Prediction") as demo:
    
    gr.Markdown("# <span id='app-title'>📈 LSTM Stock Price Predictor</span>")
    gr.Markdown("A minimalist interface for forecasting stock prices using a trained Long Short-Term Memory (LSTM) network.")
    
    # Hidden component to display status/loading screen messages
    status_output = gr.Textbox(
        label="Model Status", 
        value="Select a company and time range to begin.", 
        interactive=False, 
        elem_id="status-output"
    )

    with gr.Row():
        # 1. Company Selection
        ticker_input = gr.Dropdown(
            label="1. Select Company Ticker",
            choices=AVAILABLE_TICKERS,
            value="AAPL",
            interactive=True
        )

        # 3. Time Period Selection
        time_range_input = gr.Dropdown(
            label="2. Select Prediction Time Range",
            choices=list(DAYS_IN_RANGES.keys()),
            value="1 week",
            interactive=True
        )
    
    predict_button = gr.Button("🚀 Generate Forecast", variant="primary")

    # --- 4. Output Display Area ---
    
    # Container for visualization and tabular data
    with gr.Column(elem_classes="result-box"):
        gr.Markdown("## Forecast Results")
        
        # Plot output (HTML for custom styling and base64 image display)
        plot_output = gr.HTML(label="Predicted Stock Price Graph", visible=False)
        
        # Table output
        tabular_output = gr.Dataframe(  # Changed from gr.Dataframe
            label="Predicted Prices (Daily)",
            headers=['Date', 'Predicted Price'],
            row_count=10, 
            col_count=2,
            interactive=False,
            visible=False
        )
    
    # Event Handler
    predict_button.click(
        fn=run_prediction,
        inputs=[ticker_input, time_range_input, status_output],
        outputs=[status_output, plot_output, tabular_output]
    )

if __name__ == "__main__":
    try:
        demo.launch()
    except ImportError:
        print("ERROR: Gradio is not installed. Please run: pip install gradio")
    except Exception as e:
        print(f"An error occurred during launch: {e}")