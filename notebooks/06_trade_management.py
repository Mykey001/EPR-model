# %% [markdown]
# # 06 — Dynamic Trade Management
# 
# **EPR Mean Reversion Sniper — ML Enhancement Project**
# 
# This notebook:
# 1. Designs the dynamic exit strategies (trailing stops, partial closures).
# 2. Uses Model 2 predictions to dynamically size positions (Kelly Criterion-inspired).
# 3. Compares the fixed 1:2 RR strategy against the dynamic exit strategy.

# %%
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(r"c:\Users\MYCkey98\Desktop\Organized_Files\05_ML_Projects\EPR-ML")
LABELS_DIR = PROJECT_ROOT / "data" / "labels"

# %% [markdown]
# ## 1. Load Simulation Data
# For trade management, we look at the MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion) 
# calculated during the Phase 2 label generation.

# %%
labels_path = LABELS_DIR / "XAUUSD_M5_labeled_signals.parquet"
if not labels_path.exists():
    print("Labeled dataset not found. Please run Phase 2.")
else:
    df_trades = pd.read_parquet(labels_path)
    print(f"Loaded {len(df_trades)} simulated trades.")

# %% [markdown]
# ## 2. Analyze MFE & MAE
# Understanding how far trades go in our favor before reversing is key to dynamic exits.

# %%
if 'mfe_pips' in df_trades.columns:
    plt.figure(figsize=(10, 6))
    
    # We convert pips to R-multiples for standardization
    df_trades['mfe_r'] = df_trades['mfe_pips'] / df_trades['sl_distance_pips']
    df_trades['mae_r'] = df_trades['mae_pips'] / df_trades['sl_distance_pips']
    
    # MFE distribution for Winning vs Losing trades
    wins = df_trades[df_trades['outcome'] == 'WIN']
    losses = df_trades[df_trades['outcome'] == 'LOSS']
    
    plt.hist(wins['mfe_r'], bins=50, alpha=0.5, label='Wins MFE (R)', color='green')
    plt.hist(losses['mfe_r'], bins=50, alpha=0.5, label='Losses MFE (R)', color='red')
    
    plt.title('Max Favorable Excursion (in R-multiples)')
    plt.xlabel('R-Multiple Reached')
    plt.ylabel('Frequency')
    plt.axvline(x=2.0, color='lime', linestyle='--', label='Default TP (2R)')
    plt.legend()
    plt.show()

# %% [markdown]
# ## 3. Define Dynamic Exit Rules
# 
# **Rule 1: Breakeven at 1R**
# - If price reaches +1R in profit, move Stop Loss to Entry.
# - If trade later reverses, outcome = Breakeven (0 PnL) instead of -1R loss.
# 
# **Rule 2: Trailing Stop after 1R**
# - After 1R, trail the stop loss by 1R behind the peak price.

# %%
def simulate_dynamic_exits(trade):
    """
    Simulates dynamic exit logic based on MFE and MAE.
    Note: A true simulation requires bar-by-bar data to know if MFE or MAE hit first.
    For this notebook, we approximate using the final MFE/MAE.
    """
    mfe = trade['mfe_r']
    mae = trade['mae_r']
    original_pnl = trade['r_multiple']
    
    # 1. Did it hit -1R immediately before any profit?
    if original_pnl <= -1.0:
        return -1.0
        
    # 2. Did it reach +1R?
    if mfe >= 1.0:
        # Breakeven activated.
        # If the original trade ended up losing, it means it hit +1R then reversed to SL.
        # With breakeven, it stops out at 0.
        if original_pnl <= -1.0:
            return 0.0
            
        # Trailing stop activated after 1R
        # If MFE was e.g. 1.8, trailing stop was at 0.8.
        # Assuming the trade reversed from MFE, we exit at MFE - 1.0
        trailing_exit = mfe - 1.0
        
        # We cap it at the original TP if we didn't remove the TP limit
        # Let's say we remove the 2R limit and just trail:
        return trailing_exit
        
    # 3. Never reached 1R, just hit SL
    return original_pnl

if 'mfe_r' in df_trades.columns:
    df_trades['dynamic_r_multiple'] = df_trades.apply(simulate_dynamic_exits, axis=1)
    
    # Compare
    orig_sum = df_trades['r_multiple'].sum()
    dyn_sum = df_trades['dynamic_r_multiple'].sum()
    
    print("Trade Management Strategy Comparison:")
    print(f"Fixed 1:2 RR Total R: {orig_sum:.2f} R")
    print(f"Dynamic Trailing Total R: {dyn_sum:.2f} R")
    
    # Win rate change (Breakeven trades are neutral, but reduce losses)
    orig_losses = (df_trades['r_multiple'] <= -1.0).sum()
    dyn_losses = (df_trades['dynamic_r_multiple'] <= -1.0).sum()
    saved = orig_losses - dyn_losses
    print(f"Losses saved by Breakeven rule: {saved} trades")

# %% [markdown]
# ## 4. Dynamic Position Sizing (Confidence-Based)
# We can scale the risk between 0.5% and 2.0% based on Model 2's predicted R-multiple.

# %%
def get_dynamic_risk(predicted_r):
    """Scale risk based on predicted trade quality."""
    if predicted_r < 0: return 0.0     # Don't trade
    if predicted_r < 0.5: return 0.5   # Half risk
    if predicted_r < 1.0: return 1.0   # Standard risk
    if predicted_r < 1.5: return 1.5   # Increased risk
    return 2.0                         # Max risk

print("\nReady for production integration.")
