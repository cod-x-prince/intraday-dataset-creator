# flask_dashboard/app.py (Final, Celery Beat Version)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, render_template, request, redirect, url_for
import json, redis, eventlet, eventlet.wsgi

app = Flask(__name__)

# --- Paths, Connections, and Presets (Unchanged) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PREDICTIONS_FILE = os.path.join(PROJECT_ROOT, 'predictions.json')
TICKERS_FILE = os.path.join(PROJECT_ROOT, 'tickers.txt')
REDIS_URL = 'redis://localhost:6379/0'
redis_conn = redis.from_url(REDIS_URL)
PRESET_TICKERS = {"Big Tech": "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA", "Major ETFs": "SPY,QQQ,IWM,GLD,SLV"}

def get_tickers():
    try:
        with open(TICKERS_FILE, 'r') as f: tickers = [t.strip() for t in f.read().split(',') if t.strip()]
        return tickers
    except FileNotFoundError:
        default_tickers = PRESET_TICKERS["Big Tech"]
        with open(TICKERS_FILE, 'w') as f: f.write(default_tickers)
        return default_tickers.split(',')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        with open(TICKERS_FILE, 'w') as f: f.write(request.form.get('tickers'))
        return redirect(url_for('home'))
    tickers = get_tickers()
    ticker_string = ','.join(tickers)
    predictions = {}
    try:
        with open(PREDICTIONS_FILE, 'r') as f: predictions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): pass
    display_predictions = {ticker: predictions[ticker] for ticker in tickers if ticker in predictions}
    engine_status = redis_conn.get('engine_status')
    engine_status = engine_status.decode('utf-8') if engine_status else 'stopped'
    return render_template('index.html', title="AI Trading Cockpit", predictions=display_predictions,
                           ticker_string=ticker_string, preset_tickers=PRESET_TICKERS, engine_status=engine_status)

@app.route('/start')
def start_engine():
    """Flips the master switch to 'running'."""
    redis_conn.set('engine_status', 'running')
    return redirect(url_for('home'))

@app.route('/stop')
def stop_engine():
    """Flips the master switch to 'stopped'."""
    redis_conn.set('engine_status', 'stopped')
    return redirect(url_for('home'))

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5000)), app)