# prediction_engine.py
import pandas as pd
import numpy as np
import xgboost as xgb
import pandas_ta as ta
import joblib
import shap
from alpaca_trade_api.rest import REST, TimeFrame
import json
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
API_KEY = "AK8GEL37N4WOH9AFQT9W"
SECRET_KEY = "WH3OXGdDSKz7VO1DGhPSiNPOkcZyS4Ce2PZ04zUF"
UNIVERSE = ['NVDA', 'TSLA', 'MSFT', 'SPY', 'GLD']
MODEL_FILE = "trader_ai_model.pkl"
OUTPUT_FILE = "predictions.json"

# --- 1. Load Saved Model & Connect to API ---
print("Initializing Prediction Engine...")
model = joblib.load(MODEL_FILE)
api = REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url='https://paper-api.alpaca.markets')
explainer = shap.TreeExplainer(model)

# --- 2. Check Market Status ---
clock = api.get_clock()
if not clock.is_open:
    print(f"Market is closed. Last open: {clock.timestamp.isoformat()}.")
    exit()

print("Market is open. Fetching live data...")

# --- 3. Fetch & Process Data for Each Ticker ---
all_predictions = {}
for ticker in UNIVERSE:
    print(f"  - Processing {ticker}...")
    # Fetch enough data for feature calculation
    bars = api.get_bars(ticker, TimeFrame.Minute, limit=200, adjustment='raw').df
    if bars.empty:
        continue
    
    # a) Feature Engineering (must match training script)
    bars = bars.resample('15T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
    bars['Price_Range'] = bars['High'] - bars['Low']
    bars['ATR_14'] = ta.atr(bars['High'], bars['Low'], bars['Close'], length=14)
    bars['RSI_14'] = ta.rsi(bars['Close'], length=14)
    df_1h = bars.resample('1h').agg({'Close': 'last'}).dropna()
    bars['RSI_1h'] = ta.rsi(df_1h['Close'], length=14).reindex(bars.index, method='ffill')
    
    # We don't have live sentiment, so we use a neutral value (0)
    bars['SentimentScore'] = 0
    bars['Sentiment_Lag1'] = bars['SentimentScore'].shift(1)
    
    bars.dropna(inplace=True)
    if bars.empty:
        continue
        
    # b) Prepare for Prediction
    latest_row = bars.tail(1)
    final_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_1h', 'ATR_14', 'RSI_14', 'Price_Range', 'SentimentScore', 'Sentiment_Lag1']
    X_latest = latest_row[final_features]

    # c) Generate Prediction & Explanation
    prediction = model.predict(X_latest)[0]
    confidence = model.predict_proba(X_latest)[0][prediction]
    signal = "UP" if prediction == 1 else "DOWN"
    shap_values = explainer.shap_values(X_latest)
    
    contributions = pd.Series(shap_values[0], index=X_latest.columns).sort_values(ascending=False)
    
    key_drivers = []
    for feature, value in contributions.head(3).items():
        direction = "supports" if (value > 0 and prediction == 1) or (value < 0 and prediction == 0) else "opposes"
        key_drivers.append(f"{feature} ({direction})")
        
    all_predictions[ticker] = {
        "signal": signal,
        "confidence": f"{confidence:.2%}",
        "drivers": key_drivers
    }

# --- 4. Save to JSON File ---
print(f"Saving all predictions to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_predictions, f, indent=4)
print("✅ Prediction engine run complete.")