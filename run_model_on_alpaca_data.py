# run_model_on_alpaca_data.py (Final Version with Model Saving)
import pandas as pd
import numpy as np
import xgboost as xgb
import pandas_ta as ta
from sklearn.metrics import classification_report, accuracy_score
import joblib # New import for saving the model
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

print("--- Starting AI Trader Pipeline with Alpaca Data ---")

# --- 1. Load Your New, Large Dataset ---
try:
    df = pd.read_csv("raw_alpaca_data_MAX.csv")
except FileNotFoundError:
    print("❌ FATAL ERROR: Make sure 'raw_alpaca_data.csv' is in the folder.")
    exit()

print("Step 1: Cleaning and Preparing Data...")
df['Datetime'] = pd.to_datetime(df['Datetime'])

# --- 2. Feature Engineering ---
print("Step 2: Engineering All Features...")
final_dfs = []
for ticker, group in df.groupby('Ticker'):
    print(f"  - Engineering features for {ticker}...")
    group = group.sort_values('Datetime').reset_index(drop=True)
    group['ATR_14'] = ta.atr(group['High'], group['Low'], group['Close'], length=14)
    group['RSI_14'] = ta.rsi(group['Close'], length=14)
    group['Price_Range'] = group['High'] - group['Low']
    df_1h = group.set_index('Datetime').resample('1h').agg({'Close': 'last'}).dropna()
    group['RSI_1h'] = ta.rsi(df_1h['Close'], length=14).reindex(group.set_index('Datetime').index, method='ffill').values
    final_dfs.append(group)
df = pd.concat(final_dfs)


# --- 3. Create Simple Directional Target ---
print("Step 3: Creating Target Variable...")
df['future_return'] = df.groupby('Ticker')['Close'].transform(lambda x: x.shift(-1) / x - 1)
df['target'] = np.where(df['future_return'] > 0, 1, 0)

# --- 4. Final Cleanup ---
print("Step 4: Performing Final Cleanup...")
df.dropna(inplace=True)
df['target'] = df['target'].astype(int)

if df.empty:
    print("❌ FATAL ERROR: The dataset is empty after feature engineering.")
    exit()

# --- 5. Train the Final Model ---
print("Step 5: Training Final Model...")
y = df['target']
features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'ATR_14', 'RSI_14', 'Price_Range', 'RSI_1h'
]
X = df[features]

split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    use_label_encoder=False, n_estimators=200,
    max_depth=5, learning_rate=0.1, random_state=42
)
model.fit(X_train, y_train)

# --- 6. Final Evaluation ---
print("\n--- ✅ FINAL RESULTS on Multi-Year Alpaca Data ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

# --- 7. NEW: Save the Trained Model ---
MODEL_FILE = "trader_ai_model.pkl"
print(f"\nStep 7: Saving the trained model to '{MODEL_FILE}'...")
joblib.dump(model, MODEL_FILE)
print("✅ Model saved successfully.")