# price_action_analyzer.py
import pandas as pd
import numpy as np

# We will reuse our existing functions from other modules
from market_structure import get_swing_points
from poi_identifier import find_fvgs

def add_price_action_features(data):
    """
    Analyzes price action around swing points to identify sweeps and displacements.

    Returns the DataFrame with new features:
    - `last_interaction`: A categorical feature (None, 'Sweep', 'Displacement').
    - `interaction_feature`: A numerical feature (0=None, 1=Sweep, 2=Displacement).
    """
    swing_highs, swing_lows = get_swing_points(data)

    data['last_interaction'] = None

    # Analyze interactions at swing highs
    for idx, row in swing_highs.iterrows():
        level = row['High']
        # Find price action after this swing high was formed
        future_candles = data[data.index > idx]

        for candle_idx, candle in future_candles.iterrows():
            interaction_found = False
            # Check for a sweep
            if candle['High'] > level and candle['Close'] < level:
                data.at[candle_idx, 'last_interaction'] = 'Sweep'
                interaction_found = True
                break # Stop after the first interaction with this level

            # Check for a displacement (body close above + FVG)
            if candle['Close'] > level:
                # Check for an FVG created by this move (using a 3-candle window around the break)
                window_start = data.index.get_loc(candle_idx) - 1
                window_end = data.index.get_loc(candle_idx) + 2
                if window_start >= 0 and window_end <= len(data):
                    three_candle_window = data.iloc[window_start:window_end]
                    fvgs = find_fvgs(three_candle_window)
                    if any(fvg['type'] == 'Bullish' for fvg in fvgs):
                        data.at[candle_idx, 'last_interaction'] = 'Displacement'
                        interaction_found = True
                        break
            if interaction_found:
                break

    # Analyze interactions at swing lows (similar logic)
    for idx, row in swing_lows.iterrows():
        level = row['Low']
        future_candles = data[data.index > idx]

        for candle_idx, candle in future_candles.iterrows():
            interaction_found = False
            # Check for a sweep
            if candle['Low'] < level and candle['Close'] > level:
                data.at[candle_idx, 'last_interaction'] = 'Sweep'
                interaction_found = True
                break

            # Check for a displacement (body close below + FVG)
            if candle['Close'] < level:
                window_start = data.index.get_loc(candle_idx) - 1
                window_end = data.index.get_loc(candle_idx) + 2
                if window_start >= 0 and window_end <= len(data):
                    three_candle_window = data.iloc[window_start:window_end]
                    fvgs = find_fvgs(three_candle_window)
                    if any(fvg['type'] == 'Bearish' for fvg in fvgs):
                        data.at[candle_idx, 'last_interaction'] = 'Displacement'
                        interaction_found = True
                        break
            if interaction_found:
                break

    # Convert the categorical interaction into a numerical feature
    interaction_map = {'Sweep': 1, 'Displacement': 2}
    data['interaction_feature'] = data['last_interaction'].map(interaction_map).fillna(0).astype(int)

    return data.drop(columns=['last_interaction'])