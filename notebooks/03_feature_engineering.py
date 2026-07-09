# %% [markdown]
# # 03 — Feature Engineering
# 
# **EPR Mean Reversion Sniper — ML Enhancement Project**
# 
# This notebook:
# 1. Loads the fully labeled dataset (signals mapped to targets).
# 2. Computes the advanced machine learning features (structural, temporal, multi-timeframe).
# 3. Performs feature selection / dimensionality reduction.
# 4. Removes correlated features.
# 5. Saves the final feature matrix (`X`) and targets (`y`) for Model Training.

# %%
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# Project paths
PROJECT_ROOT = Path(r"c:\Users\MYCkey98\Desktop\Organized_Files\05_ML_Projects\EPR-ML")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"

# %% [markdown]
# ## 1. Load the Labeled Full Dataset
# We load the full dataset that contains `target_is_win` and `target_r_multiple`.
# We need to compute features on the full dataset so we can use lagging/rolling windows.

# %%
data_path = PROCESSED_DIR / "XAUUSD_M5_full_with_targets.parquet"
if not os.path.exists(data_path):
    print("Full labeled dataset not found. Please run 01 and 02 first.")
else:
    df = pd.read_parquet(data_path)
    print(f"Loaded dataset: {df.shape}")
    print(f"Total signals found in data: {df['target_is_signal'].sum()}")

# %% [markdown]
# ## 2. Feature Construction
# We already calculated many base features in Phase 1 (EMA diffs, BB width, ADX, Pivots, Temporal encodings).
# Let's generate higher-level derived features.

# %%
def create_features(df):
    """Generates advanced ML features from the base indicators."""
    X = pd.DataFrame(index=df.index)
    
    # --- 1. Momentum & Trend Features ---
    # EMA Slopes (rate of change over last 3 bars)
    X['ema_fast_slope'] = df['ema_fast'].diff(3) / df['ema_fast'].shift(3) * 1000
    X['ema_slow_slope'] = df['ema_slow'].diff(3) / df['ema_slow'].shift(3) * 1000
    X['ema_diff_pct'] = df['ema_diff_pct']  # already computed
    
    # MACD-like histogram (EMA cross momentum)
    X['ema_hist_momentum'] = df['ema_diff'].diff(1)
    
    # ADX components
    X['adx'] = df['adx']
    X['di_diff'] = df['di_diff']
    X['di_diff_slope'] = df['di_diff'].diff(3)
    
    # --- 2. Mean Reversion & Volatility Features ---
    # RSI
    X['rsi'] = df['rsi']
    X['rsi_slope'] = df['rsi'].diff(3)
    
    # Bollinger Bands
    X['bb_width'] = df['bb_width']
    X['bb_width_roc'] = df['bb_width'].pct_change(5) # expansion vs contraction
    X['bb_pct_b'] = df['bb_pct_b']
    
    # ATR & Candle sizes
    X['atr_ratio'] = df['atr_ratio']
    X['body_ratio'] = df['body_ratio']
    X['upper_wick_ratio'] = df['upper_wick_ratio']
    X['lower_wick_ratio'] = df['lower_wick_ratio']
    
    # --- 3. Structural Features (Pivot Points) ---
    X['dist_to_pivot_high_pct'] = (df['dist_to_pivot_high'] / df['close']) * 100
    X['dist_to_pivot_low_pct'] = (df['dist_to_pivot_low'] / df['close']) * 100
    X['swing_range_pct'] = (df['swing_range'] / df['close']) * 100
    X['price_in_swing_pct'] = df['price_in_swing_pct']
    X['bars_since_pivot_high'] = df['bars_since_pivot_high']
    X['bars_since_pivot_low'] = df['bars_since_pivot_low']
    
    # --- 4. Macro Filter Features ---
    X['macro_distance_pct'] = (df['macro_distance'] / df['close']) * 100
    
    # --- 5. PSAR Features ---
    X['psar_distance_pct'] = (df['psar_distance'] / df['close']) * 100
    
    # --- 6. Temporal Features ---
    X['hour_sin'] = df['hour_sin']
    X['hour_cos'] = df['hour_cos']
    X['dow_sin'] = df['dow_sin']
    X['dow_cos'] = df['dow_cos']
    
    # One-hot encode the session
    # (assuming session is strings: 'asian', 'london', 'new_york', 'overlap', 'off_hours')
    sessions = pd.get_dummies(df['session'], prefix='session')
    X = pd.concat([X, sessions], axis=1)
    
    # --- 7. Strategy Condition Proximity ---
    # How close were we to all conditions triggering?
    X['buy_conditions_met'] = df['buy_conditions_met']
    X['sell_conditions_met'] = df['sell_conditions_met']
    
    # Forward-fill missing values (for early data) and drop remaining NaNs
    X = X.ffill().bfill()
    
    return X

# %%
print("Creating advanced features...")
X_full = create_features(df)
print(f"Feature matrix shape: {X_full.shape}")
print(f"Number of features: {len(X_full.columns)}")

# %% [markdown]
# ## 3. Feature Selection & Correlation Analysis
# We want to remove features that are highly collinear (Spearman correlation > 0.95).

# %%
def drop_highly_correlated_features(X, threshold=0.95):
    """Drop features that are highly correlated with each other."""
    print("Calculating correlation matrix...")
    corr_matrix = X.corr(method='spearman').abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    X_reduced = X.drop(columns=to_drop)
    return X_reduced

X_reduced = drop_highly_correlated_features(X_full, threshold=0.90)
print(f"Final feature count: {len(X_reduced.columns)}")

# %% [markdown]
# ## 4. Create Datasets for Different Models
# 
# **Model 1: Signal Classifier**
# - Needs all bars. Target: `is_good_buy`, `is_good_sell`, `no_trade`.
# - Since we want to predict *good* signals, we'll label bars where `target_is_win == 1` and `signal == 1` as Class 1 (Good Buy), `target_is_win == 1` and `signal == -1` as Class 2 (Good Sell), and everything else as Class 0 (No Trade / Bad Trade).
# 
# **Model 2: Trade Quality Regressor**
# - Needs ONLY signal bars (`target_is_signal == 1`).
# - Target: `target_r_multiple`.

# %%
# --- Model 1 Dataset ---
y_class = pd.Series(0, index=df.index, name='target_class')
y_class[(df['signal'] == 1) & (df['target_is_win'] == 1)] = 1  # Good Buy
y_class[(df['signal'] == -1) & (df['target_is_win'] == 1)] = 2 # Good Sell

# Add targets to X
data_m1 = X_reduced.copy()
data_m1['target_class'] = y_class

# --- Model 2 Dataset ---
signal_indices = df[df['target_is_signal'] == 1].index
data_m2 = X_reduced.loc[signal_indices].copy()
data_m2['target_r_multiple'] = df.loc[signal_indices, 'target_r_multiple']
data_m2['signal_direction'] = df.loc[signal_indices, 'signal'] # 1 or -1

# %% [markdown]
# ## 5. Save Datasets

# %%
m1_path = PROCESSED_DIR / "model1_features.parquet"
m2_path = PROCESSED_DIR / "model2_features.parquet"

data_m1.to_parquet(m1_path)
data_m2.to_parquet(m2_path)

print(f"\n✅ Model 1 Dataset saved: {data_m1.shape}")
print(f"✅ Model 2 Dataset saved: {data_m2.shape}")
print("\nReady for Phase 4 (Model Training).")
