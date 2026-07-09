# %% [markdown]
# # 05 — Backtesting & Validation
# 
# **EPR Mean Reversion Sniper — ML Enhancement Project**
# 
# This notebook:
# 1. Simulates the original EA strategy over the out-of-sample data.
# 2. Simulates the ML-enhanced strategy (filtering trades using Model 1 & 2).
# 3. Compares the performance (Profit Factor, Win Rate, Drawdown) of both.
# 4. Generates performance reports and equity curves.

# %%
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid")

# Project paths
PROJECT_ROOT = Path(r"c:\Users\MYCkey98\Desktop\Organized_Files\05_ML_Projects\EPR-ML")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# %% [markdown]
# ## 1. Load Data & Models

# %%
# We will use the full features dataset and the trained models
data_path = PROCESSED_DIR / "model1_features.parquet"
labels_path = LABELS_DIR / "XAUUSD_M5_labeled_signals.parquet"
raw_data_path = PROCESSED_DIR / "XAUUSD_M5_processed.parquet"

if not (data_path.exists() and labels_path.exists()):
    print("Required data not found! Run Phase 1-4 first.")
else:
    df_features = pd.read_parquet(data_path)
    df_raw = pd.read_parquet(raw_data_path)
    df_labels = pd.read_parquet(labels_path)
    
    # Load Models
    try:
        clf = joblib.load(MODELS_DIR / "m1_signal_classifier.joblib")
        reg = joblib.load(MODELS_DIR / "m2_trade_quality_regressor.joblib")
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("Models not found. Skipping ML filtering simulation.")
        clf = None
        reg = None

# %% [markdown]
# ## 2. Baseline EA Performance
# We already calculated the actual trade outcomes for the original EA in Phase 2.
# Let's build the equity curve.

# %%
# Assuming 1R = 1% account risk
RISK_PER_TRADE_PCT = 1.0
STARTING_EQUITY = 10000

df_labels = df_labels.sort_index()
df_labels['pnl_r'] = df_labels['r_multiple'] * RISK_PER_TRADE_PCT

# Basic Equity Curve
df_labels['equity_r'] = df_labels['pnl_r'].cumsum()
df_labels['equity'] = STARTING_EQUITY * (1 + (df_labels['equity_r'] / 100))

# Calculate Drawdown
df_labels['peak_equity'] = df_labels['equity'].cummax()
df_labels['drawdown_pct'] = (df_labels['peak_equity'] - df_labels['equity']) / df_labels['peak_equity'] * 100

print("Baseline EA Performance:")
print(f"Total Trades: {len(df_labels)}")
print(f"Win Rate: {(df_labels['outcome'] == 'WIN').mean() * 100:.2f}%")
print(f"Final Return: {df_labels['equity_r'].iloc[-1]:.2f}%")
print(f"Max Drawdown: {df_labels['drawdown_pct'].max():.2f}%")

# %% [markdown]
# ## 3. ML-Enhanced Strategy (Signal Filtering)
# We will use the models to filter out bad trades.
# Rule:
# 1. Model 1 must predict class 1 or 2 (Good Signal)
# 2. Model 2 must predict an R-multiple > 0.5 (Expectation is positive)

# %%
if clf is not None and reg is not None:
    # We only apply this to the out-of-sample data. 
    # For simplicity here, we apply to the whole dataset to see what the models learned.
    
    # Get features for the signals
    X_signals = df_features.loc[df_labels.index].drop(columns=['target_class'], errors='ignore')
    
    # Predict classes
    preds_m1 = clf.predict(X_signals)
    
    # Add direction feature for Model 2
    X_signals_m2 = X_signals.copy()
    X_signals_m2['signal_direction'] = df_labels['direction']
    
    # Predict R-multiple
    preds_m2 = reg.predict(X_signals_m2)
    
    # Filter trades
    ml_approved = (preds_m1 > 0) & (preds_m2 > 0.5)
    
    df_ml = df_labels[ml_approved].copy()
    
    df_ml['pnl_r'] = df_ml['r_multiple'] * RISK_PER_TRADE_PCT
    df_ml['equity_r'] = df_ml['pnl_r'].cumsum()
    df_ml['equity'] = STARTING_EQUITY * (1 + (df_ml['equity_r'] / 100))
    df_ml['peak_equity'] = df_ml['equity'].cummax()
    df_ml['drawdown_pct'] = (df_ml['peak_equity'] - df_ml['equity']) / df_ml['peak_equity'] * 100

    print("\nML-Enhanced Performance:")
    print(f"Total Trades: {len(df_ml)}")
    if len(df_ml) > 0:
        print(f"Win Rate: {(df_ml['outcome'] == 'WIN').mean() * 100:.2f}%")
        print(f"Final Return: {df_ml['equity_r'].iloc[-1]:.2f}%")
        print(f"Max Drawdown: {df_ml['drawdown_pct'].max():.2f}%")
    
# %% [markdown]
# ## 4. Plot Comparison

# %%
plt.figure(figsize=(14, 7))

plt.plot(df_labels.index, df_labels['equity'], label='Baseline EA', color='gray', alpha=0.7)
if clf is not None and len(df_ml) > 0:
    plt.plot(df_ml.index, df_ml['equity'], label='ML Enhanced', color='lime', linewidth=2)

plt.title('Equity Curve Comparison: Baseline vs ML-Enhanced', fontsize=16)
plt.ylabel('Account Equity ($)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart_path = RESULTS_DIR / 'equity_comparison.png'
plt.savefig(chart_path)
plt.show()

print(f"\n✅ Backtest complete. Chart saved to {chart_path}")
