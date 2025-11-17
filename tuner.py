# tuner.py
import pandas as pd
import xgboost as xgb
import logging
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from run_prediction_engine import (
    connect_alpaca_api, 
    fetch_data, 
    preprocess_data, 
    CONFIG
)

def setup_logging():
    """Sets up basic logging for the tuner."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_hyperparameter_tuning():
    """
    Performs a grid search to find the best hyperparameters for the XGBoost model.
    """
    logging.info("--- Starting Hyperparameter Tuning ---")
    
    # 1. Load and Prepare Data
    api = connect_alpaca_api()
    if not api:
        logging.critical("Could not connect to API. Exiting.")
        return

    logging.info("Fetching a large dataset for tuning...")
    all_features = []
    for ticker in CONFIG["universe"]:
        logging.info(f"Fetching data for {ticker}...")
        bars = fetch_data(api, ticker, days=365*3)
        if not bars.empty:
            features, _ = preprocess_data(bars, fit_scaler=False)
            features['target'] = (features['Close'].shift(-1) > features['Close']).astype(int)
            features.dropna(inplace=True)
            all_features.append(features)

    # --- FIX: All logic that requires data is now nested inside this check ---
    if all_features:
        full_dataset = pd.concat(all_features)
        y = full_dataset['target']
        X = full_dataset.drop(columns=['target'])
        
        logging.info(f"Tuning on a dataset with {len(X)} samples.")

        # 2. Define the Model and Parameter Grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1]
        }

        tscv = TimeSeriesSplit(n_splits=3)

        # 3. Set up and Run the Grid Search
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss', 
            use_label_encoder=False,
            tree_method='gpu_hist'
        )
        
        logging.info("Starting GPU-accelerated Grid Search... This should be much faster.")
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=tscv,
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)

        # 4. Print the Best Results
        logging.info("--- Tuning Complete ---")
        print("Best parameters found: ", grid_search.best_params_)
        print("Best accuracy found: ", grid_search.best_score_)
    else:
        logging.error("No data could be fetched or processed. Aborting tuning.")


if __name__ == "__main__":
    setup_logging()
    run_hyperparameter_tuning()