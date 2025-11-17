# market_structure.py (Final Corrected Version)
import pandas as pd
from scipy.signal import find_peaks

def get_swing_points(data, order=5):
    """
    Identifies potential swing high and swing low points using a simple peak/trough detection.
    'order' means a peak is a point that's higher than the `order` points on either side of it.
    """
    # Find peaks (swing highs) and troughs (swing lows)
    high_peaks, _ = find_peaks(data['High'], distance=order, prominence=0.01)
    low_peaks, _ = find_peaks(-data['Low'], distance=order, prominence=0.01)
    
    swing_highs = data.iloc[high_peaks]
    swing_lows = data.iloc[low_peaks]
    
    return swing_highs, swing_lows

def map_market_structure(data):
    """
    Analyzes historical data to determine market trend.
    This version is self-cleaning and robust against duplicate index entries.
    """
    # Definitive fix: Clean the incoming data index immediately
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep='first')]
    
    # This function call now works because get_swing_points is defined above it.
    swing_highs, swing_lows = get_swing_points(data)
    
    if swing_highs.empty or swing_lows.empty:
        return pd.Series('Sideways', index=data.index)

    all_swings = pd.concat([
        pd.DataFrame({'price': swing_highs['High'], 'type': 'high'}),
        pd.DataFrame({'price': swing_lows['Low'], 'type': 'low'})
    ]).sort_index()
    
    all_swings = all_swings[~all_swings.index.duplicated(keep='first')]

    trend = "Sideways"
    last_swing_high = None
    last_swing_low = None
    trends = []

    for index, swing in all_swings.iterrows():
        current_price = swing['price']
        swing_type = swing['type']

        if swing_type == 'high':
            if last_swing_high is None or current_price > last_swing_high:
                if trend != "Uptrend" and last_swing_low is not None:
                    trend = "Uptrend" # Change of Character
                last_swing_high = current_price
            
        elif swing_type == 'low':
            if last_swing_low is None or current_price < last_swing_low:
                if trend != "Downtrend" and last_swing_high is not None:
                    trend = "Downtrend" # Change of Character
                last_swing_low = current_price
        
        trends.append(trend)
        
    trend_series = pd.Series(trends, index=all_swings.index)
    return trend_series.reindex(data.index, method='ffill').fillna("Sideways")