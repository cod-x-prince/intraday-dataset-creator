# run_prediction_engine.py (Final, Consolidated Version)
import pandas as pd
import numpy as np
import xgboost as xgb
import talib as ta
import joblib
from alpaca_trade_api.rest import REST, TimeFrame, APIError
from sklearn.preprocessing import StandardScaler
import json, os, warnings, logging, csv
from eventlet import sleep as eventlet_sleep
from datetime import datetime, timedelta
from dotenv import load_dotenv
from market_structure import map_market_structure
from poi_identifier import find_fvgs, add_fvg_features, find_sd_zones, add_sd_zone_features

warnings.filterwarnings('ignore')
load_dotenv()

# --- Centralized Configuration ---
CONFIG = {
    "api_key": os.getenv("ALPACA_API_KEY"),
    "secret_key": os.getenv("ALPACA_SECRET_KEY"),
    "base_url": 'https://api.alpaca.markets',
    "universe": ['NVDA', 'TSLA', 'MSFT', 'SPY', 'GLD'],
    "model_file": "trader_ai_model.pkl",
    "scaler_file": "feature_scaler.pkl",
    "output_file": "predictions.json",
    "tickers_file": "tickers.txt",
    "log_file": "engine.log",
    "data_frequency": "15Min",
    "scaling_method": "standard",
    "retrain_model": True,
    "api_retry_attempts": 3,
    "api_retry_delay_seconds": 15,
}

FREQUENCY_MAP = { "15Min": {"timeframe": TimeFrame.Minute, "resample": "15T"} }

# --- Finalized Feature List ---
# This list MUST be consistent between training and backtesting
FEATURES_LIST = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'RSI_1h', 'ATRr_14',
    'RSI_14', 'Price_Range', 'MACD', 'MACD_Signal', 'MACD_Hist',
    '4H_Trend_Feature', 'is_in_bullish_fvg', 'is_in_bearish_fvg',
    'is_in_supply', 'is_in_demand'
]

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(CONFIG["log_file"]), logging.StreamHandler()])
    logging.info("Logging initialized.")

def connect_alpaca_api():
    try:
        api = REST(key_id=CONFIG["api_key"], secret_key=CONFIG["secret_key"], base_url=CONFIG["base_url"])
        api.get_account()
        logging.info("Successfully connected to Alpaca API.")
        return api
    except APIError as e:
        logging.error(f"Failed to connect to Alpaca API: {e}")
        return None

def fetch_data(api, ticker, days):
    """
    Fetches historical data from Alpaca in larger, more efficient chunks.
    """
    freq_details = FREQUENCY_MAP[CONFIG["data_frequency"]]
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    all_bars = pd.DataFrame()
    
    # Calculate a more efficient chunk size. 10000 bars is a common API limit.
    # For 15-min data, that's ~104 days. We'll use 180 days to be safe and efficient.
    chunk_days = 180 

    logging.info(f"Fetching {days} days of data for {ticker} in {chunk_days}-day chunks.")
    
    current_start = start_date
    while current_start < end_date:
        # Calculate the end of the current chunk
        next_end_date = min(end_date, current_start + timedelta(days=chunk_days))
        try:
            bars = api.get_bars(
                ticker, 
                freq_details["timeframe"], 
                start=current_start.strftime("%Y-%m-%d"), 
                end=next_end_date.strftime("%Y-%m-%d"), 
                adjustment='raw', 
                feed='iex'
            ).df
            
            if not bars.empty:
                all_bars = pd.concat([all_bars, bars])
                
            logging.info(f"Fetched {len(bars)} bars for {ticker} from {current_start.strftime('%Y-%m-%d')} to {next_end_date.strftime('%Y-%m-%d')}")
            
            # Move to the next chunk
            current_start = next_end_date
            
            # A small sleep to respect rate limits, even with fewer calls
            eventlet_sleep(1) 

        except APIError as e:
            if e.status_code == 429:
                logging.warning("API rate limit hit. Sleeping for 15 seconds.")
                eventlet_sleep(15)
            else:
                logging.error(f"API Error fetching data for {ticker}: {e}")
                break
                
    if not all_bars.empty and all_bars.index.has_duplicates:
        logging.warning(f"Found and removed duplicate timestamps for {ticker}.")
        all_bars = all_bars[~all_bars.index.duplicated(keep='first')]
        
    logging.info(f"Total bars fetched for {ticker}: {len(all_bars)}")
    return all_bars

def preprocess_data(bars, scaler=None, fit_scaler=False):
    logging.info(f"Starting preprocessing. Initial rows: {len(bars)}")
    if bars.empty: return pd.DataFrame(), None
    try:
        if not isinstance(bars.index, pd.DatetimeIndex): bars.index = pd.to_datetime(bars.index)

        bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        bars.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        if bars.empty: return pd.DataFrame(), None

        resample_freq = FREQUENCY_MAP[CONFIG["data_frequency"]]["resample"]
        bars_15m = bars.resample(resample_freq).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

        # --- Market Structure Feature ---
        bars_4h = bars.resample('4H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        if not bars_4h.empty:
            bars_4h = bars_4h[~bars_4h.index.duplicated(keep='first')]
            four_hour_trend = map_market_structure(bars_4h)
            bars_15m['4H_Trend'] = four_hour_trend.reindex(bars_15m.index, method='ffill').fillna("Sideways")
        else: bars_15m['4H_Trend'] = "Sideways"
        trend_map = {"Uptrend": 1, "Downtrend": -1, "Sideways": 0}
        bars_15m['4H_Trend_Feature'] = bars_15m['4H_Trend'].map(trend_map)

        # --- Standard Features ---
        bars_15m[['High', 'Low', 'Close']] = bars_15m[['High', 'Low', 'Close']].astype(float)
        bars_15m['Price_Range'] = bars_15m['High'] - bars_15m['Low']
        bars_15m['ATRr_14'] = ta.ATR(bars_15m['High'], bars_15m['Low'], bars_15m['Close'], timeperiod=14)
        bars_15m['RSI_14'] = ta.RSI(bars_15m['Close'], timeperiod=14)
        macd, macdsignal, macdhist = ta.MACD(bars_15m['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        bars_15m['MACD'], bars_15m['MACD_Signal'], bars_15m['MACD_Hist'] = macd, macdsignal, macdhist

        # --- FVG Feature Calculation ---
        fvgs = find_fvgs(bars_15m)
        bars_15m = add_fvg_features(bars_15m, fvgs)

        # --- Supply/Demand Zone Feature Calculation ---
        sd_zones = find_sd_zones(bars_4h)
        bars_15m = add_sd_zone_features(bars_15m, sd_zones)

        # --- Other Timeframe Features ---
        df_1h = bars_15m['Close'].resample('1H').last()
        rsi_1h_values = ta.RSI(df_1h.dropna(), timeperiod=14) if len(df_1h.dropna()) >= 15 else pd.Series(50, index=df_1h.index)
        bars_15m['RSI_1h'] = rsi_1h_values.reindex(bars_15m.index, method='ffill').bfill()

        bars_15m.dropna(inplace=True)
        if bars_15m.empty: return pd.DataFrame(), None

        X = bars_15m[FEATURES_LIST]

        if fit_scaler:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            return X_scaled, scaler

        if scaler:
             X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
             return X_scaled, scaler

        # This case is for training when no scaler is passed initially
        return X, None

    except Exception as e:
        logging.exception(f"An unexpected error during preprocessing: {e}")
        return pd.DataFrame(), None

def train_model(api):
    logging.info("--- Starting Model Retraining ---")
    all_features = []
    for ticker in CONFIG["universe"]:
        bars = fetch_data(api, ticker, days=365*3)
        if not bars.empty:
            # Step 1: Preprocess to get features
            features, _ = preprocess_data(bars)
            if features.empty:
                logging.warning(f"No features generated for {ticker}, skipping.")
                continue

            # Step 2: Add target variable
            features['target'] = (features['Close'].shift(-1) > features['Close']).astype(int)
            features.dropna(inplace=True)
            all_features.append(features)

    if not all_features:
        logging.error("No data available for training. Aborting.")
        return

    full_dataset = pd.concat(all_features)
    y = full_dataset['target']
    X = full_dataset[FEATURES_LIST] # Ensure we only use the specified features

    # Step 3: Scale the features and save the scaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    joblib.dump(scaler, CONFIG["scaler_file"])
    logging.info(f"Scaler trained and saved to {CONFIG['scaler_file']}")

    # Step 4: Train the model and save it
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, tree_method='gpu_hist')
    model.fit(X_scaled, y)
    joblib.dump(model, CONFIG["model_file"])
    logging.info(f"Model retrained and saved to {CONFIG['model_file']}")

# The functions below are for live prediction and can be ignored for backtesting purposes
def log_prediction_to_csv(prediction_data):
    log_file = "predictions_log.csv"
    header = ['timestamp', 'ticker', 'signal', 'confidence', 'drivers']
    row_data = [
        prediction_data['timestamp'],
        prediction_data['ticker'],
        prediction_data['signal'],
        prediction_data['confidence'],
        ','.join(prediction_data['drivers'])
    ]
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row_data)

def execute_prediction_cycle(ticker):
    logging.info(f"--- WORKER: Starting Prediction Cycle for {ticker} ---")
    api = connect_alpaca_api()
    if not api: return f"API connection failed for {ticker}."
    try:
        model = joblib.load(CONFIG["model_file"])
        scaler = joblib.load(CONFIG["scaler_file"])
    except FileNotFoundError:
        return f"Model/scaler not found for {ticker}."

    feature_importances = pd.Series(model.feature_importances_, index=model.get_booster().feature_names)
    top_drivers = feature_importances.nlargest(3).index.tolist()
    bars = fetch_data(api, ticker, days=90) # Fetch enough data for indicators
    if bars.empty: return f"No data for {ticker}."
    X_processed, _ = preprocess_data(bars, scaler=scaler)
    if X_processed.empty: return f"Not enough data to process for {ticker}."

    latest_row = X_processed.tail(1)
    probabilities = model.predict_proba(latest_row)[0]
    prediction = np.argmax(probabilities)
    confidence = np.max(probabilities)
    signal = "UP" if prediction == 1 else "DOWN"

    prediction_data = {
        "ticker": ticker, "signal": signal, "confidence": f"{confidence:.2%}",
        "drivers": top_drivers, "timestamp": latest_row.index[0].isoformat()
    }
    log_prediction_to_csv(prediction_data)
    logging.info(f"WORKER: Prediction for {ticker}: {signal} with {confidence:.2%} confidence.")
    return prediction_data


if __name__ == "__main__":
    setup_logging()
    api = connect_alpaca_api()
    if api:
        if CONFIG["retrain_model"]:
            train_model(api)
        else:
            # Example of a single prediction run
            # result = execute_prediction_cycle('SPY')
            # print(result)
            logging.info("Model training skipped as per config. To run predictions, use a scheduler or API endpoint.")