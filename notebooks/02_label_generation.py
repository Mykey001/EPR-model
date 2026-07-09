# %% [markdown]
# # 02 — Label Generation (Ground Truth)
# 
# **EPR Mean Reversion Sniper — ML Enhancement Project**
# 
# This notebook:
# 1. Loads the processed dataset from Phase 1.
# 2. Identifies all generated BUY and SELL signals.
# 3. For each signal, simulates the trade execution forward in time to determine the outcome.
# 4. Computes exactly where the EA would place Stop Loss (SL) and Take Profit (TP).
# 5. Classifies trades as WIN (hit TP first) or LOSS (hit SL first).
# 6. Saves the labeled dataset for feature engineering and model training.

# %%
import numpy as np
import pandas as pd
import yaml
import os
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

# Project paths
PROJECT_ROOT = Path(r"c:\Users\MYCkey98\Desktop\Organized_Files\05_ML_Projects\EPR-ML")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"

# Load EA-matching configuration
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# %% [markdown]
# ## 1. Load Processed Data
# 
# We load the processed M5 data containing all indicators and the basic signals.

# %%
data_path = PROCESSED_DIR / "XAUUSD_M5_processed.parquet"
df = pd.read_parquet(data_path)

print(f"Loaded processed data: {df.shape}")
print(f"Date range: {df.index[0]} -> {df.index[-1]}")

# Identify signal bars
signal_indices = df[df['signal'] != 0].index
signals_df = df.loc[signal_indices].copy()

buy_signals = signals_df[signals_df['signal'] == 1]
sell_signals = signals_df[signals_df['signal'] == -1]

print(f"\nTotal Signals: {len(signals_df)}")
print(f"  BUY:  {len(buy_signals)}")
print(f"  SELL: {len(sell_signals)}")

# %% [markdown]
# ## 2. Trade Simulation Logic
# 
# We need to simulate the trades forward in time to find the outcome.
# 
# **Trade parameters from config:**
# - `use_swing_sl`: bool
# - `swing_sl_buffer_pips`: float
# - `fixed_sl_pips`: float
# - `risk_reward_ratio`: float
# 
# Note: Since the EA acts on the bar close (i.e. opening of the next bar), our entry price for the simulation is the **OPEN of the next bar**. The SL/TP is evaluated against High/Low of subsequent bars.

# %%
trade_cfg = config['trade_management']
USE_SWING_SL = trade_cfg['use_swing_sl']
SWING_SL_BUFFER_PIPS = trade_cfg['swing_sl_buffer_pips']
FIXED_SL_PIPS = trade_cfg['fixed_sl_pips']
RR_RATIO = trade_cfg['risk_reward_ratio']
PIP_SIZE = 0.1  # For XAUUSD, typically 1 pip = 0.1 or 0.01 depending on broker. Standard MT5 XAUUSD spread=200 points=20 pips. 1 point = 0.01. So 1 pip = 0.1.
# Actually, looking at the MQL5 code: pipSize = _Point * 10
# For XAUUSD with 2 or 3 digits. Let's assume _Point = 0.01. Then pipSize = 0.1.

POINT_SIZE = 0.01
PIP_VALUE = POINT_SIZE * 10

def simulate_trades(df, signal_indices):
    """
    Simulates trades for each signal in signal_indices.
    Returns a DataFrame containing trade outcomes.
    """
    results = []
    
    # Pre-extract numpy arrays for fast forward-scanning
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    times = df.index.values
    
    # We will use integer indices to step forward
    # Getting the integer locations of signal_indices
    # Because signal occurs at bar close t, entry is at t+1 open
    signal_ilocs = np.where(df.index.isin(signal_indices))[0]
    
    for iloc in tqdm(signal_ilocs, desc="Simulating Trades"):
        signal_row = df.iloc[iloc]
        signal_time = times[iloc]
        direction = signal_row['signal']
        
        # Entry is next bar's open
        entry_idx = iloc + 1
        if entry_idx >= len(df):
            continue # Signal at the very end of data
            
        entry_price = opens[entry_idx]
        entry_time = times[entry_idx]
        
        # Calculate SL
        if direction == 1: # BUY
            if USE_SWING_SL and not pd.isna(signal_row['last_pivot_low']):
                sl = signal_row['last_pivot_low'] - (SWING_SL_BUFFER_PIPS * PIP_VALUE)
                # Fallback if SL is invalid (above entry)
                if sl >= entry_price:
                    sl = entry_price - (FIXED_SL_PIPS * PIP_VALUE)
            else:
                sl = entry_price - (FIXED_SL_PIPS * PIP_VALUE)
                
        else: # SELL
            if USE_SWING_SL and not pd.isna(signal_row['last_pivot_high']):
                sl = signal_row['last_pivot_high'] + (SWING_SL_BUFFER_PIPS * PIP_VALUE)
                # Fallback if SL is invalid (below entry)
                if sl <= entry_price:
                    sl = entry_price + (FIXED_SL_PIPS * PIP_VALUE)
            else:
                sl = entry_price + (FIXED_SL_PIPS * PIP_VALUE)
                
        # Calculate TP based on RR
        sl_distance = abs(entry_price - sl)
        tp_distance = sl_distance * RR_RATIO
        
        if direction == 1:
            tp = entry_price + tp_distance
        else:
            tp = entry_price - tp_distance
            
        # Walk forward to find outcome
        outcome = 'TIMEOUT'
        exit_price = np.nan
        exit_time = np.nan
        exit_idx = np.nan
        mfe_price = entry_price
        mae_price = entry_price
        
        for j in range(entry_idx, len(df)):
            curr_high = highs[j]
            curr_low = lows[j]
            
            # Update MFE/MAE
            if direction == 1:
                if curr_high > mfe_price: mfe_price = curr_high
                if curr_low < mae_price: mae_price = curr_low
            else:
                if curr_low < mfe_price: mfe_price = curr_low
                if curr_high > mae_price: mae_price = curr_high
                
            # Check for SL / TP hits
            if direction == 1: # BUY
                # Did we hit SL?
                if curr_low <= sl:
                    # To be perfectly precise, if both SL and TP are hit in same bar, 
                    # we often assume SL is hit first for conservative testing.
                    outcome = 'LOSS'
                    exit_price = sl
                    exit_time = times[j]
                    exit_idx = j
                    break
                # Did we hit TP?
                elif curr_high >= tp:
                    outcome = 'WIN'
                    exit_price = tp
                    exit_time = times[j]
                    exit_idx = j
                    break
            else: # SELL
                if curr_high >= sl:
                    outcome = 'LOSS'
                    exit_price = sl
                    exit_time = times[j]
                    exit_idx = j
                    break
                elif curr_low <= tp:
                    outcome = 'WIN'
                    exit_price = tp
                    exit_time = times[j]
                    exit_idx = j
                    break
        
        if outcome == 'TIMEOUT':
            # Last bar of dataset
            exit_price = closes[-1]
            exit_time = times[-1]
            exit_idx = len(df) - 1
            
        # Calculate actual PnL (R-multiple)
        if direction == 1:
            pnl_pips = (exit_price - entry_price) / PIP_VALUE
        else:
            pnl_pips = (entry_price - exit_price) / PIP_VALUE
            
        r_multiple = pnl_pips / (sl_distance / PIP_VALUE) if sl_distance > 0 else 0
        
        # MFE / MAE in pips
        mfe_pips = abs(mfe_price - entry_price) / PIP_VALUE
        mae_pips = abs(mae_price - entry_price) / PIP_VALUE
        
        results.append({
            'signal_time': signal_time,
            'direction': direction,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'sl_price': sl,
            'tp_price': tp,
            'sl_distance_pips': sl_distance / PIP_VALUE,
            'tp_distance_pips': tp_distance / PIP_VALUE,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'outcome': outcome,
            'pnl_pips': pnl_pips,
            'r_multiple': r_multiple,
            'duration_bars': exit_idx - entry_idx + 1,
            'mfe_pips': mfe_pips,
            'mae_pips': mae_pips
        })
        
    return pd.DataFrame(results)

# %%
# Run the simulation
print(f"Simulating {len(signal_indices)} trades...")
trade_results = simulate_trades(df, signal_indices)

# Merge back with the signal features
trade_results.set_index('signal_time', inplace=True)
labeled_data = signals_df.join(trade_results, how='inner')

print(f"\nSimulation complete. Labeled {len(labeled_data)} trades.")

# %% [markdown]
# ## 3. Results Analysis

# %%
# Basic statistics
total_trades = len(labeled_data)
wins = len(labeled_data[labeled_data['outcome'] == 'WIN'])
losses = len(labeled_data[labeled_data['outcome'] == 'LOSS'])
timeouts = len(labeled_data[labeled_data['outcome'] == 'TIMEOUT'])

win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

print(f"{'='*50}")
print("TRADE SIMULATION RESULTS")
print(f"{'='*50}")
print(f"Total Trades: {total_trades}")
print(f"Wins:         {wins} ({wins/total_trades*100:.1f}%)")
print(f"Losses:       {losses} ({losses/total_trades*100:.1f}%)")
print(f"Timeouts:     {timeouts}")
print(f"\nWin Rate (excluding timeouts): {win_rate:.2f}%")

avg_r = labeled_data['r_multiple'].mean()
print(f"\nAverage R-Multiple: {avg_r:.2f}R")

# Calculate Profit Factor (Sum of win PnL / Absolute sum of loss PnL)
win_pnl = labeled_data[labeled_data['outcome'] == 'WIN']['pnl_pips'].sum()
loss_pnl = abs(labeled_data[labeled_data['outcome'] == 'LOSS']['pnl_pips'].sum())

if loss_pnl > 0:
    profit_factor = win_pnl / loss_pnl
    print(f"Profit Factor:      {profit_factor:.2f}")
else:
    print(f"Profit Factor:      N/A (No losses)")

# Average duration
avg_duration = labeled_data['duration_bars'].mean()
print(f"Average Duration:   {avg_duration:.1f} bars")

# Breakdown by direction
print("\nBreakdown by Direction:")
for drn, name in [(1, 'BUY'), (-1, 'SELL')]:
    subset = labeled_data[labeled_data['direction'] == drn]
    w = len(subset[subset['outcome'] == 'WIN'])
    l = len(subset[subset['outcome'] == 'LOSS'])
    wr = w / (w + l) * 100 if (w + l) > 0 else 0
    print(f"  {name}: {len(subset)} trades, Win Rate: {wr:.2f}%")

# %% [markdown]
# ## 4. Create Regression Target (Trade Quality)
# 
# For Model 2 (Trade Quality Regressor), we need a continuous target that represents the quality of the trade.
# Options:
# - Actual PnL in pips
# - Actual R-multiple (bounded between roughly -1.0 and +2.0)
# - Max Favorable Excursion (MFE) to understand if the trade went our way initially even if it eventually lost.
# 
# Let's create a combined 'Quality Score'.
# For now, `r_multiple` is a great regression target.

# %%
labeled_data['target_r_multiple'] = labeled_data['r_multiple']
labeled_data['target_is_win'] = (labeled_data['outcome'] == 'WIN').astype(int)

# %% [markdown]
# ## 5. Save Labeled Data

# %%
# Save labeled dataset
output_path = LABELS_DIR / "XAUUSD_M5_labeled_signals.parquet"
labeled_data.to_parquet(output_path, engine='pyarrow')
print(f"✅ Labeled dataset saved to: {output_path}")

# Also save full dataset with forward-looking labels mapped back (optional, useful for Model 1 - Signal Classifier)
# Since Model 1 predicts if a bar is a good signal, we can map target_is_win back to df.
df['target_is_win'] = 0
df['target_r_multiple'] = np.nan
df['target_is_signal'] = (df['signal'] != 0).astype(int)

# Map labels to signal bars
df.loc[labeled_data.index, 'target_is_win'] = labeled_data['target_is_win']
df.loc[labeled_data.index, 'target_r_multiple'] = labeled_data['target_r_multiple']

# Note: non-signal bars have target_is_win=0 (they aren't signals, so they aren't winning trades)
full_labeled_path = PROCESSED_DIR / "XAUUSD_M5_full_with_targets.parquet"
df.to_parquet(full_labeled_path, engine='pyarrow')
print(f"✅ Full dataset with targets saved to: {full_labeled_path}")

print("\nReady for Phase 3 (Feature Engineering).")
