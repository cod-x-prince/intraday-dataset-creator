# poi_identifier.py
import pandas as pd
import numpy as np

# --- Fair Value Gap (FVG) Functions ---
def find_fvgs(data):
    """Identifies Fair Value Gaps (FVGs) from OHLC data."""
    fvgs = []
    for i in range(1, len(data) - 1):
        prev_candle = data.iloc[i-1]
        next_candle = data.iloc[i+1]
        
        # Bullish FVG
        if prev_candle['High'] < next_candle['Low']:
            fvgs.append({'type': 'Bullish', 'start_time': data.index[i], 'top': next_candle['Low'], 'bottom': prev_candle['High']})
        # Bearish FVG
        elif prev_candle['Low'] > next_candle['High']:
            fvgs.append({'type': 'Bearish', 'start_time': data.index[i], 'top': prev_candle['Low'], 'bottom': next_candle['High']})
    return fvgs

def add_fvg_features(data, fvgs):
    """Creates features based on FVG locations."""
    data['is_in_bullish_fvg'] = 0
    data['is_in_bearish_fvg'] = 0
    # ... (rest of FVG feature logic can be refined later) ...
    return data

# --- NEW: Supply and Demand (S/D) Zone Functions ---
def find_sd_zones(data, base_max_candles=5, move_multiplier=3.0):
    """
    Identifies Supply and Demand zones based on basing candles and explosive moves.
    """
    zones = []
    i = 0
    while i < len(data):
        # Look for a basing period
        base_start_index = i
        base_end_index = i
        for j in range(1, base_max_candles):
            if i + j < len(data):
                # Simple base definition: small candles
                if abs(data.iloc[i+j]['Close'] - data.iloc[i+j]['Open']) < abs(data.iloc[i]['Close'] - data.iloc[i]['Open']) * 2:
                    base_end_index = i + j
                else:
                    break
        
        if base_start_index == base_end_index:
            i += 1
            continue
            
        base_candles = data.iloc[base_start_index : base_end_index + 1]
        base_high = base_candles['High'].max()
        base_low = base_candles['Low'].min()
        base_range = base_high - base_low

        if base_range == 0:
            i = base_end_index + 1
            continue

        # Check for move *away* from the base
        if base_end_index + 1 < len(data):
            move_away_candle = data.iloc[base_end_index + 1]
            move_range = abs(move_away_candle['Close'] - move_away_candle['Open'])

            if move_range >= base_range * move_multiplier:
                # Check move *before* the base to determine context
                if base_start_index > 0:
                    move_before_candle = data.iloc[base_start_index - 1]
                    
                    # Rally-Base-Drop (Supply)
                    if move_before_candle['Close'] > move_before_candle['Open'] and move_away_candle['Close'] < move_away_candle['Open']:
                        zones.append({'type': 'Supply', 'start_time': base_candles.index[0], 'top': base_high, 'bottom': base_low})
                    
                    # Drop-Base-Rally (Demand)
                    elif move_before_candle['Close'] < move_before_candle['Open'] and move_away_candle['Close'] > move_away_candle['Open']:
                        zones.append({'type': 'Demand', 'start_time': base_candles.index[0], 'top': base_high, 'bottom': base_low})

        i = base_end_index + 1
    return zones

def add_sd_zone_features(data, zones):
    """
    Creates features based on price's proximity to S/D zones.
    A simple feature: is the current price inside a fresh (unmitigated) zone?
    """
    data['is_in_supply'] = 0
    data['is_in_demand'] = 0
    
    active_zones = zones.copy()

    for i in range(len(data)):
        current_candle = data.iloc[i]
        
        # Check against all currently active zones
        for zone in active_zones[:]: # Iterate over a copy
            if current_candle.name > zone['start_time']:
                is_supply = zone['type'] == 'Supply'
                is_demand = zone['type'] == 'Demand'
                
                # Check if current price touches the zone
                if current_candle['High'] >= zone['bottom'] and current_candle['Low'] <= zone['top']:
                    if is_supply:
                        data.at[current_candle.name, 'is_in_supply'] = 1
                    elif is_demand:
                        data.at[current_candle.name, 'is_in_demand'] = 1
                    
                    # First touch mitigates the zone (for this simple model, we remove it)
                    active_zones.remove(zone)
                
                # Check if zone is invalidated (price closes beyond it)
                elif (is_supply and current_candle['Close'] > zone['top']) or \
                     (is_demand and current_candle['Close'] < zone['bottom']):
                    active_zones.remove(zone)
    return data