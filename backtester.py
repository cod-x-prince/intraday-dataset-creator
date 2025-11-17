# backtester.py (Rebuilt with Fibonacci Filter)
import pandas as pd
import joblib
import logging
import talib as ta
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks
import numpy as np
# Make sure run_prediction_engine has the necessary functions available for import
from run_prediction_engine import connect_alpaca_api, fetch_data, preprocess_data, FEATURES_LIST

BACKTEST_CONFIG = {
    "ticker": "SPY",
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "initial_cash": 10000,
    "commission_rate": 0.002,
    "model_file": "trader_ai_model.pkl",
    "scaler_file": "feature_scaler.pkl"
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_scaler():
    try:
        model = joblib.load(BACKTEST_CONFIG["model_file"])
        scaler = joblib.load(BACKTEST_CONFIG["scaler_file"])
        logging.info("Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        logging.error("Model or scaler not found. Please run the prediction engine to train first.")
        return None, None

def generate_historical_signals(historical_data, model, scaler):
    # Preprocess the data to get the same features the model was trained on
    X_processed, _ = preprocess_data(historical_data, scaler=scaler, fit_scaler=False)
    if X_processed.empty:
        logging.error("Preprocessing failed, no features generated.")
        return pd.DataFrame()

    # --- Add strategy-specific indicators (these were NOT used for model training) ---
    X_processed['SMA200'] = ta.SMA(X_processed['Close'], timeperiod=200)
    X_processed['ADX14'] = ta.ADX(X_processed['High'], X_processed['Low'], X_processed['Close'], timeperiod=14)
    X_processed.dropna(inplace=True)

    # Use the correct features for prediction
    features_for_prediction = X_processed[FEATURES_LIST]
    predictions = model.predict(features_for_prediction)
    X_processed['signal'] = predictions

    logging.info("Historical signals generated successfully.")
    return X_processed

def get_fib_levels(price_series):
    """
    Finds the last significant swing high and low to calculate Fibonacci levels.
    Returns the low, high, and the 50% retracement level.
    """
    # Look at the last ~4 months of daily data (approx 80 trading days) to find a swing
    series = price_series.tail(80)
    
    # Using a larger distance to find more significant swings
    peaks, _ = find_peaks(series, distance=5)
    troughs, _ = find_peaks(-series, distance=5)

    if len(peaks) > 0 and len(troughs) > 0:
        # Get the timestamp of the last peak and trough
        last_high_idx = series.index[peaks[-1]]
        last_low_idx = series.index[troughs[-1]]
        
        # Ensure the swing is recent and in the correct order for a pullback (low followed by high)
        if last_high_idx > last_low_idx:
            last_low_price = series[last_low_idx]
            last_high_price = series[last_high_idx]
            return last_low_price, last_high_price
            
    return None, None

# The final, champion strategy for backtester.py

class TRADER_AI_v1(Strategy):
    """
    Final Champion Version.
    This strategy uses a machine learning model signal confirmed by trend and strength filters.
    It enters on a deep Fibonacci pullback and uses a dynamic ATR-based trailing stop for exits.
    Optimized Parameters as of 2025-09-13.
    """
    # --- Final Optimized Parameters ---
    adx_threshold = 25
    fib_entry_min = 0.50
    fib_entry_max = 0.786
    stop_loss_atr_multiplier = 2.0
    trailing_atr_multiplier = 1.25

    def init(self):
        self.signal = self.I(lambda: self.data.signal, name="ML_Signal")
        self.sma = self.I(lambda: self.data.SMA200, name="SMA200")
        self.adx = self.I(lambda: self.data.ADX14, name="ADX14")
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name="ATR14")

    def next(self):
        price = self.data.Close[-1]
        current_atr = self.atr[-1]

        if self.position:
            current_trade = self.trades[-1]
            new_sl = max(current_trade.sl, price - (current_atr * self.trailing_atr_multiplier))
            current_trade.sl = new_sl
            return

        is_primary_uptrend = price > self.sma[-1]
        is_strong_trend = self.adx[-1] > self.adx_threshold
        is_buy_signal = self.signal[-1] == 1

        if not (is_primary_uptrend and is_strong_trend and is_buy_signal):
            return

        is_in_discount_zone = False
        price_series = pd.Series(self.data.Close)
        last_low, last_high = get_fib_levels(price_series)

        if last_low is not None and last_high is not None and last_high > last_low:
            swing_range = last_high - last_low
            entry_zone_top = last_high - (swing_range * self.fib_entry_min)
            entry_zone_bottom = last_high - (swing_range * self.fib_entry_max)
            if price <= entry_zone_top and price >= entry_zone_bottom:
                is_in_discount_zone = True

        if is_in_discount_zone:
            sl_price = price - (current_atr * self.stop_loss_atr_multiplier)
            self.buy(sl=sl_price, tp=None)

#-------------------------------------------------------------------------------
# Add this new experimental class to backtester.py
#Metric	Champion      |(TRADER_AI_v1)  |	Sweep Filter (v1.1)
# Trades	          |82	           | 15
# Return [%]	      |✅ 66.1%        |20.7%
# Sharpe Ratio 	      |✅ 1.40         |1.14
# Max Drawdown [%]	  |-13.6%          | ✅ -5.15%
# Win Rate [%]	      | 42.7%          |	✅ 53.3%
# Profit Factor	      | 2.13           |	✅ 3.70

#-------------------------------------------------------------------------------

#class TRADER_AI_v1_Sweep(Strategy):
    """
    Experimental Version 1.1: Adds a liquidity sweep filter.
    This strategy tests if confirming a recent bullish sweep
    improves the quality of the champion model's entries.
    """
    # --- Champion Parameters ---
    adx_threshold = 25
    fib_entry_min = 0.50
    fib_entry_max = 0.786
    stop_loss_atr_multiplier = 2.0
    trailing_atr_multiplier = 1.25
    
    # --- New Sweep Filter Parameter ---
    sweep_lookback_period = 10 # How many bars to look back to find a sweep

    def init(self):
        self.signal = self.I(lambda: self.data.signal, name="ML_Signal")
        self.sma = self.I(lambda: self.data.SMA200, name="SMA200")
        self.adx = self.I(lambda: self.data.ADX14, name="ADX14")
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name="ATR14")

    def next(self):
        # --- Standard Exit and Entry Logic (from Champion) ---
        price = self.data.Close[-1]
        current_atr = self.atr[-1]

        if self.position:
            current_trade = self.trades[-1]
            new_sl = max(current_trade.sl, price - (current_atr * self.trailing_atr_multiplier))
            current_trade.sl = new_sl
            return

        is_primary_uptrend = price > self.sma[-1]
        is_strong_trend = self.adx[-1] > self.adx_threshold
        is_buy_signal = self.signal[-1] == 1

        if not (is_primary_uptrend and is_strong_trend and is_buy_signal):
            return

        is_in_discount_zone = False
        price_series = pd.Series(self.data.Close)
        last_low, last_high = get_fib_levels(price_series)

        if last_low is not None and last_high is not None and last_high > last_low:
            swing_range = last_high - last_low
            entry_zone_top = last_high - (swing_range * self.fib_entry_min)
            entry_zone_bottom = last_high - (swing_range * self.fib_entry_max)
            if price <= entry_zone_top and price >= entry_zone_bottom:
                is_in_discount_zone = True
        
        if not is_in_discount_zone:
            return

        # --- NEW SWEEP FILTER ---
        # All conditions are met, now we check for a recent bullish sweep
        # before pulling the trigger.
        
        has_recent_sweep = False
        
        # Ensure we have enough data for the lookback
        if len(self.data.Close) > self.sweep_lookback_period:
            # Get the prices for the lookback period
            lookback_lows = self.data.Low[-self.sweep_lookback_period:]
            
            # Find the index of the absolute lowest point in that period
            # Note: np.argmin returns an index relative to the slice, not the full data
            sweep_candle_local_idx = np.argmin(lookback_lows)
            
            # Get the full data for that specific candle
            sweep_candle_open = self.data.Open[-self.sweep_lookback_period:][sweep_candle_local_idx]
            sweep_candle_close = self.data.Close[-self.sweep_lookback_period:][sweep_candle_local_idx]
            
            # Check if our two conditions for a sweep are met
            is_rejection_candle = sweep_candle_close > sweep_candle_open
            is_recent = (self.sweep_lookback_period - sweep_candle_local_idx) <= 5 # Happened in the last 5 bars
            
            if is_rejection_candle and is_recent:
                has_recent_sweep = True

        # --- FINAL ENTRY ---
        if has_recent_sweep:
            sl_price = price - (current_atr * self.stop_loss_atr_multiplier)
            self.buy(sl=sl_price, tp=None)

if __name__ == "__main__":
    setup_logging()
    model, scaler = load_model_and_scaler()
    api = connect_alpaca_api()

    if model and scaler and api:
        # Fetch a longer history to ensure all indicators (like SMA200) are mature
        raw_data = fetch_data(api, BACKTEST_CONFIG['ticker'], days=365*5)
        
        if not raw_data.empty:
            data_with_signals = generate_historical_signals(raw_data, model, scaler)
            
            # Filter the data for the specific backtest period
            backtest_data = data_with_signals.loc[BACKTEST_CONFIG['start_date']:BACKTEST_CONFIG['end_date']]
            
            if not backtest_data.empty:
                bt = Backtest(
                    backtest_data,
                    TRADER_AI_v1,
                    cash=BACKTEST_CONFIG['initial_cash'],
                    commission=BACKTEST_CONFIG['commission_rate']
                )
                stats = bt.run()
                print("--- Backtest Results ---")
                print(stats)
                print("------------------------")
                bt.plot()
            else:
                logging.error("No data available for the specified backtest date range.")
        else:
            logging.error(f"Failed to fetch data for {BACKTEST_CONFIG['ticker']}.")