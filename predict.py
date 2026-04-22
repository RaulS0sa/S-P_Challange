import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Modern NLP: Use sentence-transformers (easier for recruiters to run than Ollama)
from sentence_transformers import SentenceTransformer
import xgboost as xgb

# -----------------------------
# 1. LOAD DATA & CLEANING
# -----------------------------
# Note: Replace with actual file paths or URLs provided in the email
PRICE_URL = "https://huy302.github.io/price.csv"
NEWS_URL = "https://huy302.github.io/news.csv"

def load_and_setup():
    price = pd.read_csv('price.csv')
    news = pd.read_csv('news.csv')
    
    price['date'] = pd.to_datetime(price['date'])
    news['datetime'] = pd.to_datetime(news['datetime'])
    news['date'] = news['datetime'].dt.normalize() # Just the date for merging
    
    return price, news

price, news = load_and_setup()

# -----------------------------
# 2. ADVANCED FEATURE ENGINEERING (PRICE)
# -----------------------------
def engineer_price_features(df):
    df = df.sort_values(by=['ticker', 'date'])
    
    # Returns and Log Returns (Log returns are more stable for modeling)
    df['ret'] = df.groupby('ticker')['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Technical Indicators
    for t in [5, 10, 21]:
        # Moving Averages
        df[f'ma_{t}'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(t).mean())
        # Relative Volatility
        df[f'vol_{t}'] = df.groupby('ticker')['ret'].transform(lambda x: x.rolling(t).std())
    
    # Momentum (RSI Simple Implementation)
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['rsi'] = df.groupby('ticker')['close'].transform(compute_rsi)
    
    # TARGET: Next day's return (What we want to predict)
    df['target'] = df.groupby('ticker')['ret'].shift(-1)
    
    return df

price = engineer_price_features(price)

# -----------------------------
# 3. NLP FEATURE EXTRACTION (NEWS)
# -----------------------------
print("Encoding news text...")
# Using a lightweight, high-performance model
model_nlp = SentenceTransformer('all-MiniLM-L6-v2')

news['text'] = news['headline'].fillna('') + " " + news['summary'].fillna('')
embeddings = model_nlp.encode(news['text'].tolist(), show_progress_bar=True)

# Reduce dimensions to avoid the "curse of dimensionality"
pca = PCA(n_components=10)
emb_reduced = pca.fit_transform(embeddings)
emb_cols = [f'news_pca_{i}' for i in range(10)]
news_features = pd.DataFrame(emb_reduced, columns=emb_cols)

# Combine and aggregate by ticker and date
news = pd.concat([news.reset_index(drop=True), news_features], axis=1)
agg_news = news.groupby(['ticker', 'date'])[emb_cols].mean().reset_index()

# -----------------------------
# 4. MERGE & PRE-PROCESSING
# -----------------------------
df = price.merge(agg_news, on=['ticker', 'date'], how='left')

# Forward fill news (if no news today, assume sentiment persists) then fill rest with 0
df[emb_cols] = df.groupby('ticker')[emb_cols].ffill().fillna(0)

# Drop rows with NaN targets or initial rolling window NaNs
df = df.dropna().sort_values('date')

# -----------------------------
# 5. MODELING (XGBOOST)
# -----------------------------
# Time-based split: Don't use random_state split for time series!
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

features = [col for col in df.columns if col not in ['date', 'ticker', 'target', 'open', 'high', 'low', 'close']]

X_train, y_train = train[features], train['target']
X_test, y_test = test[features], test['target']

regressor = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.003,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# -----------------------------
# EVALUATION
# -----------------------------
preds = regressor.predict(X_test)

print(f"\nModel Performance:")
print(f"MAE: {mean_absolute_error(y_test, preds):.6f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.6f}")

# Critical Financial Metric: Directional Accuracy
dir_acc = np.mean(np.sign(preds) == np.sign(y_test))
print(f"Directional Accuracy: {dir_acc:.2%}")

# Plot Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(regressor, max_num_features=15)
plt.show()


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# 1. CONVERT REGRESSION TO CLASSIFICATION (The "Signal")
# We care about the direction: Is the return > 0 (Up) or <= 0 (Down)?
y_test_binary = (y_test > 0).astype(int)
preds_binary = (preds > 0).astype(int)

# 2. CALCULATE FANCY SCORES
accuracy = accuracy_score(y_test_binary, preds_binary)
precision = precision_score(y_test_binary, preds_binary)
recall = recall_score(y_test_binary, preds_binary)
f1 = f1_score(y_test_binary, preds_binary)

print("\n" + "="*30)
print("TRADING SIGNAL PERFORMANCE")
print("="*30)
print(f"Accuracy (Hit Rate): {accuracy:.2%}")
print(f"Precision (When model says BUY, how often is it right?): {precision:.2%}")
print(f"Recall (How many upside moves did we catch?): {recall:.2%}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, preds_binary, target_names=['Down/Flat', 'Up']))

# 3. VISUALIZATION SUITE
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Plot A: Confusion Matrix ---
sns.heatmap(confusion_matrix(y_test_binary, preds_binary), annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
axes[0,0].set_title("Confusion Matrix: Up vs Down Move")
axes[0,0].set_xlabel("Predicted")
axes[0,0].set_ylabel("Actual")

# --- Plot B: Predicted vs Actual Returns (Zoomed in) ---
# Taking the last 100 observations for clarity
axes[0,1].plot(y_test.values[-100:], label='Actual Return', alpha=0.7)
axes[0,1].plot(preds[-100:], label='Predicted Return', alpha=0.7, linestyle='--')
axes[0,1].set_title("Actual vs Predicted Returns (Last 100 Days)")
axes[0,1].legend()

# --- Plot C: Cumulative Returns (The "Money" Plot) ---
# This shows: If I traded based on this model, what would happen?
# Strategy: If prediction > 0, go Long. If prediction < 0, go Short (or stay cash).
strategy_returns = np.sign(preds) * y_test
cum_market_returns = (1 + y_test).cumprod()
cum_strategy_returns = (1 + strategy_returns).cumprod()

axes[1,0].plot(cum_market_returns.values, label='Market (Buy & Hold)', color='gray', alpha=0.6)
axes[1,0].plot(cum_strategy_returns.values, label='Model-Based Strategy', color='green', lw=2)
axes[1,0].set_title("Equity Curve: Model vs Market")
axes[1,0].set_ylabel("Cumulative Growth (Multiplier)")
axes[1,0].legend()

# --- Plot D: Feature Importance ---
# Re-using the XGBoost importance plot
xgb.plot_importance(regressor, max_num_features=10, ax=axes[1,1], importance_type='weight')
axes[1,1].set_title("Top 10 Drivers of Prediction")

plt.tight_layout()
plt.show()

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
