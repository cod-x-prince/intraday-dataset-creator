# run_mentor_model.py (Corrected Sequence)
import pandas as pd
import numpy as np
import xgboost as xgb
import pandas_ta as ta
from sklearn.metrics import classification_report, accuracy_score
import shap
import warnings

warnings.filterwarnings('ignore')

print("--- Starting AI Master Mentor Pipeline ---")

# --- 1. Load Data ---
try:
    df = pd.read_csv("raw_alpaca_data_MAX.csv")
    df_sentiment = pd.read_csv("preprocessed_sentiment.csv")
    print("Step 1: Successfully loaded massive dataset.")
except FileNotFoundError:
    print("❌ FATAL ERROR: Make sure 'raw_alpaca_data_MAX.csv' and 'preprocessed_sentiment.csv' are in the folder.")
    exit()

# --- 2. Initial Data Preparation ---
print("Step 2: Preparing initial data...")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df_sentiment['PublishDate'] = pd.to_datetime(df_sentiment['PublishDate'])
df = df.sort_values(by=['Ticker', 'Datetime']).reset_index(drop=True)

# --- 3. Feature Engineering (Part 1) ---
print("Step 3: Engineering technical features...")
ta_dfs = []
for ticker, group in df.groupby('Ticker'):
    group = group.copy()
    group['Price_Range'] = pd.to_numeric(group['High'], errors='coerce') - pd.to_numeric(group['Low'], errors='coerce')
    group['ATR_14'] = ta.atr(group['High'], group['Low'], group['Close'], length=14)
    group['RSI_14'] = ta.rsi(group['Close'], length=14)
    df_1h = group.set_index('Datetime').resample('1h').agg({'Close': 'last'}).dropna()
    group['RSI_1h'] = ta.rsi(df_1h['Close'], length=14).reindex(group.set_index('Datetime').index, method='ffill').values
    ta_dfs.append(group)
df = pd.concat(ta_dfs)

# --- 4. Merge Sentiment & Create Dependant Features ---
print("Step 4: Merging sentiment and creating lag features...")
df = df.sort_values('Datetime')
df_sentiment = df_sentiment.sort_values('PublishDate')
df = pd.merge_asof(
    df, df_sentiment,
    left_on='Datetime', right_on='PublishDate',
    by='Ticker', direction='backward'
)
df['SentimentScore'].fillna(0, inplace=True)
df['Sentiment_Lag1'] = df.groupby('Ticker')['SentimentScore'].shift(1)

# --- 5. Target Variable ---
print("Step 5: Creating target variable...")
df['future_return'] = df.groupby('Ticker')['Close'].transform(lambda x: x.shift(-1) / x - 1)
df['target'] = np.where(df['future_return'] > 0, 1, 0)

# --- 6. Final Cleanup ---
print("Step 6: Performing final cleanup...")
df.dropna(inplace=True)
df['target'] = df['target'].astype(int)

# --- 7. Train the Model ---
print("Step 7: Training the predictive model...")
y = df['target']
final_features = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'RSI_1h',
    'ATR_14', 'RSI_14', 'Price_Range', 'SentimentScore', 'Sentiment_Lag1'
]
X = df[final_features]
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    use_label_encoder=False, n_estimators=200,
    max_depth=5, learning_rate=0.1, random_state=42
)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"--- Model Trained. Accuracy: {accuracy:.4f} ---")

# --- 8. Intelligence Layer (SHAP Analysis) ---
print("\nStep 8: Generating 'Mentor Note' for the latest data point...")
sample_data = X[df.loc[X.index, 'Ticker'] == 'NVDA'].tail(1)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_data)

prediction = model.predict(sample_data)[0]
confidence = model.predict_proba(sample_data)[0][prediction]
signal = "UP" if prediction == 1 else "DOWN"

print(f"\n--- AI MENTOR PRE-TRADE BRIEFING (for NVDA) ---")
print(f"Prediction: {signal}")
print(f"Confidence: {confidence:.2%}")

feature_names = sample_data.columns
contributions = pd.Series(shap_values[0], index=feature_names).sort_values(ascending=False)

print("\nKey Drivers for this Prediction:")
for feature, value in contributions.head(3).items():
    direction = "supports" if (value > 0 and prediction == 1) or (value < 0 and prediction == 0) else "opposes"
    print(f"  - {feature}: {direction} the prediction.")
print("---------------------------------------------")