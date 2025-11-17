# worker.py (Final, Definitive Non-Blocking Version)
import eventlet
eventlet.monkey_patch()

from celery import Celery
import redis
import logging
import json
from eventlet.tpool import execute # Import the thread pool executor
from run_prediction_engine import execute_prediction_cycle, log_prediction_to_csv, CONFIG

REDIS_URL = 'redis://localhost:6379/0'
redis_conn = redis.from_url(REDIS_URL)

app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)
app.config_from_object('celery_config')

@app.on_after_configure.connect
def setup_initial_state(sender, **kwargs):
    print("WORKER BOOTSTEP: Setting initial engine state to 'stopped'.")
    redis_conn.set('engine_status', 'stopped')

@app.task
def start_engine_loop():
    """
    This task is triggered by Celery Beat. It checks the engine status,
    gets the list of tickers, and then loops through them, running each in a thread.
    """
    if redis_conn.get('engine_status').decode('utf-8') == 'running':
        print("WORKER: Received task from Beat. Engine is 'running', proceeding with cycle.")
        try:
            with open(CONFIG['tickers_file'], 'r') as f:
                tickers = [t.strip() for t in f.read().split(',') if t.strip()]
            
            all_predictions = {}
            # Loop through tickers and call the engine for each one
            for ticker in tickers:
                # --- FIX: Run the blocking function in a separate thread ---
                prediction_result = execute(execute_prediction_cycle, ticker)
                
                if isinstance(prediction_result, dict):
                    all_predictions[ticker] = prediction_result
                    log_prediction_to_csv(prediction_result)

            # Save the final JSON for the dashboard
            with open(CONFIG["output_file"], 'w') as f:
                json.dump(all_predictions, f, indent=4)
            print(f"WORKER: Cycle complete. {len(all_predictions)} predictions saved.")

        except Exception as e:
            print(f"WORKER ERROR: An error occurred during prediction cycle: {e}")
            logging.exception(f"WORKER ERROR: {e}")
    else:
        print("WORKER: Received task from Beat, but engine is 'stopped'. Skipping cycle.")

    return "Task check complete."