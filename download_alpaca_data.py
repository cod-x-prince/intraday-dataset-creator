# download_alpaca_data.py (Updated for Maximum Data)
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
import os

# --- ENTER YOUR ALPACA API KEYS HERE ---
API_KEY = "AK8GEL37N4WOH9AFQT9W"
SECRET_KEY = "WH3OXGdDSKz7VO1DGhPSiNPOkcZyS4Ce2PZ04zUF"

# --- Configuration (Updated) ---
UNIVERSE = ['NVDA', 'TSLA', 'MSFT', 'SPY', 'GLD']
START_DATE = "2016-01-01" # Fetch all data since 2016
END_DATE = "2025-08-29"
OUTPUT_FILE = "raw_alpaca_data_MAX.csv" # New file name

# --- Connect to Alpaca API ---
api = REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url='https://paper-api.alpaca.markets')
print("Successfully connected to Alpaca API.")

# --- Download Data ---
all_data = []
for ticker in UNIVERSE:
    print(f"Fetching all available 15-minute data for {ticker} since 2016...")
    try:
        bars = api.get_bars(
            ticker,
            TimeFrame.Minute,
            start=START_DATE,
            end=END_DATE,
            adjustment='raw'
        ).df

        if bars.empty:
            print(f"  - No data found for {ticker}. Skipping.")
            continue
        
        bars = bars.resample('15T').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

        bars['ticker'] = ticker
        all_data.append(bars)
        print(f"  - Successfully fetched {len(bars)} data points for {ticker}.")

    except Exception as e:
        print(f"  - Error fetching data for {ticker}: {e}")

# --- Combine and Save ---
if all_data:
    final_df = pd.concat(all_data)
    final_df.reset_index(inplace=True)
    final_df.rename(columns={
        'timestamp': 'Datetime', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        'ticker': 'Ticker'
    }, inplace=True)
    
    final_df['Datetime'] = final_df['Datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Success! Massive dataset with {len(final_df)} rows saved to '{OUTPUT_FILE}'")
else:
    print("\n❌ No data was downloaded.")