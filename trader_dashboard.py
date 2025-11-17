# trader_dashboard.py
import streamlit as st
import json
from datetime import datetime
import time

PREDICTIONS_FILE = "predictions.json"

st.set_page_config(page_title="AI Trader Mentor", page_icon="🤖", layout="wide")
st.title("🤖 AI Trader Master Mentor")

last_updated_placeholder = st.empty()
st.markdown("---")
cols = st.columns(5)
tickers = ['NVDA', 'TSLA', 'MSFT', 'SPY', 'GLD']
placeholders = {ticker: col for ticker, col in zip(tickers, cols)}

def display_briefing():
    try:
        with open(PREDICTIONS_FILE, 'r') as f:
            predictions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = {}

    last_updated_placeholder.text(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for ticker, ph in placeholders.items():
        briefing = predictions.get(ticker)
        
        with ph:
            st.subheader(ticker)
            if briefing:
                signal = briefing['signal']
                confidence = briefing['confidence']
                drivers = briefing['drivers']
                
                if signal == "UP":
                    st.metric(label="Prediction", value=signal, delta="↑ Positive")
                else:
                    st.metric(label="Prediction", value=signal, delta="↓ Negative", delta_color="inverse")
                
                st.write(f"**Confidence:** {confidence}")
                st.write("**Key Drivers:**")
                for driver in drivers:
                    st.markdown(f"- {driver}")
            else:
                st.metric(label="Prediction", value="Waiting...")

while True:
    display_briefing()
    time.sleep(15) # Refresh every 15 seconds