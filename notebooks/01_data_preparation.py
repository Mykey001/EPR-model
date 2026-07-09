# %% [markdown]
# # 01 — Data Preparation & Indicator Computation
# 
# **EPR Mean Reversion Sniper — ML Enhancement Project**
# 
# This notebook:
# 1. Loads raw OHLCV data (M5 + H1) exported from MetaTrader 5
# 2. Cleans, validates, and aligns the data
# 3. Computes all 7 indicators exactly matching the EA's default parameters
# 4. Detects pivot highs/lows using the EA's exact algorithm
# 5. Maps the H1 macro EMA back to M5 bars
# 6. Saves the processed dataset for the next phase

# %% [markdown]
# ## 1. Imports & Configuration

# %%
import numpy as np
import pandas as pd
import talib
import yaml
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.5f}'.format)

# Project paths
PROJECT_ROOT = Path(r"c:\Users\MYCkey98\Desktop\Organized_Files\05_ML_Projects\EPR-ML")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"

# Load EA-matching configuration
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

print("Configuration loaded:")
for section, params in config.items():
    if isinstance(params, dict):
        print(f"\n  [{section}]")
        for k, v in params.items():
            print(f"    {k}: {v}")

# %% [markdown]
# ## 2. Load Raw Data

# %%
def load_mt5_csv(filepath, has_time=True):
    """Load a MetaTrader 5 exported CSV file.
    
    MT5 exports are tab-separated with columns:
    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
    
    Daily files may omit the <TIME> column.
    """
    df = pd.read_csv(filepath, sep='\t')
    
    # Standardize column names (remove angle brackets)
    df.columns = [c.strip('<>').lower() for c in df.columns]
    
    # Build datetime index
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M:%S')
    else:
        df['datetime'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
    
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time'] if 'time' in df.columns else ['date'], inplace=True, errors='ignore')
    
    # Rename for clarity
    df.rename(columns={
        'tickvol': 'tick_volume',
        'vol': 'real_volume'
    }, inplace=True)
    
    # Ensure proper dtypes
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    
    return df

# %%
# Load M5 data (primary timeframe for signal generation)
m5_file = RAW_DATA_DIR / "XAUUSDm_M5_202307022205_202607091215.csv"
df_m5 = load_mt5_csv(m5_file)

print(f"M5 Data Shape: {df_m5.shape}")
print(f"Date Range: {df_m5.index[0]} → {df_m5.index[-1]}")
print(f"Total Bars: {len(df_m5):,}")
print(f"\nFirst 5 rows:")
df_m5.head()

# %%
# Load H1 data (for macro filter — 200 EMA on H1)
h1_file = RAW_DATA_DIR / "XAUUSDm_H1_202307022200_202607091200.csv"
df_h1 = load_mt5_csv(h1_file)

print(f"H1 Data Shape: {df_h1.shape}")
print(f"Date Range: {df_h1.index[0]} → {df_h1.index[-1]}")
print(f"Total Bars: {len(df_h1):,}")
df_h1.head()

# %%
# Load M15 data (for optional multi-timeframe features later)
m15_file = RAW_DATA_DIR / "XAUUSDm_M15_202307022200_202607091215.csv"
df_m15 = load_mt5_csv(m15_file)

print(f"M15 Data Shape: {df_m15.shape}")
print(f"Date Range: {df_m15.index[0]} → {df_m15.index[-1]}")

# %%
# Load Daily data (for optional macro context later)
daily_file = RAW_DATA_DIR / "XAUUSDm_Daily_202307020000_202607090000.csv"
df_daily = load_mt5_csv(daily_file, has_time=False)

print(f"Daily Data Shape: {df_daily.shape}")
print(f"Date Range: {df_daily.index[0]} → {df_daily.index[-1]}")

# %% [markdown]
# ## 3. Data Validation & Cleaning

# %%
def validate_ohlcv(df, name=""):
    """Run comprehensive data quality checks on OHLCV data."""
    issues = []
    
    # 1. Check for NaN values
    nan_counts = df[['open', 'high', 'low', 'close']].isna().sum()
    if nan_counts.sum() > 0:
        issues.append(f"NaN values found: {nan_counts.to_dict()}")
    
    # 2. Check OHLC consistency: High >= Low, High >= Open/Close, Low <= Open/Close
    bad_hl = (df['high'] < df['low']).sum()
    if bad_hl > 0:
        issues.append(f"{bad_hl} bars where High < Low")
    
    bad_ho = (df['high'] < df['open']).sum()
    bad_hc = (df['high'] < df['close']).sum()
    if bad_ho > 0:
        issues.append(f"{bad_ho} bars where High < Open")
    if bad_hc > 0:
        issues.append(f"{bad_hc} bars where High < Close")
    
    bad_lo = (df['low'] > df['open']).sum()
    bad_lc = (df['low'] > df['close']).sum()
    if bad_lo > 0:
        issues.append(f"{bad_lo} bars where Low > Open")
    if bad_lc > 0:
        issues.append(f"{bad_lc} bars where Low > Close")
    
    # 3. Check for duplicate timestamps
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate timestamps")
    
    # 4. Check for zero or negative prices
    zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
    if zero_prices > 0:
        issues.append(f"{zero_prices} zero/negative price values")
    
    # 5. Check for extreme price jumps (> 5% in single bar)
    pct_change = df['close'].pct_change().abs()
    extreme_jumps = (pct_change > 0.05).sum()
    if extreme_jumps > 0:
        issues.append(f"{extreme_jumps} bars with >5% price jump (may be weekend gaps)")
    
    # Report
    print(f"{'='*60}")
    print(f"Validation Report: {name}")
    print(f"{'='*60}")
    print(f"  Total bars: {len(df):,}")
    print(f"  Date range: {df.index[0]} → {df.index[-1]}")
    print(f"  Price range: {df['low'].min():.2f} → {df['high'].max():.2f}")
    
    if len(issues) == 0:
        print(f"  ✅ All checks passed")
    else:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    
    return len(issues) == 0

# %%
# Validate all datasets
validate_ohlcv(df_m5, "XAUUSD M5")
print()
validate_ohlcv(df_h1, "XAUUSD H1")
print()
validate_ohlcv(df_m15, "XAUUSD M15")
print()
validate_ohlcv(df_daily, "XAUUSD Daily")

# %%
# Remove any duplicate timestamps (keep first occurrence)
for name, df in [("M5", df_m5), ("H1", df_h1), ("M15", df_m15), ("Daily", df_daily)]:
    dups = df.index.duplicated()
    if dups.sum() > 0:
        print(f"{name}: Removing {dups.sum()} duplicate timestamps")
        df = df[~dups]

# Sort by index to ensure chronological order
df_m5 = df_m5.sort_index()
df_h1 = df_h1.sort_index()
df_m15 = df_m15.sort_index()
df_daily = df_daily.sort_index()

print("Data cleaning complete.")
print(f"M5: {len(df_m5):,} bars | H1: {len(df_h1):,} bars | M15: {len(df_m15):,} bars | Daily: {len(df_daily):,} bars")

# %% [markdown]
# ## 4. Compute Indicators (Exactly Matching EA Parameters)
# 
# Each indicator is computed to match the MQL5 EA's exact parameters and calculation method.
# Reference: `docs/indicator_reference.md`

# %% [markdown]
# ### 4.1 EMA Fast (5) & Slow (15)

# %%
# EMA parameters from config
ema_fast_period = config['ema']['fast_period']   # 5
ema_slow_period = config['ema']['slow_period']   # 15

# Compute EMAs using TA-Lib (matches MQL5's MODE_EMA)
df_m5['ema_fast'] = talib.EMA(df_m5['close'].values, timeperiod=ema_fast_period)
df_m5['ema_slow'] = talib.EMA(df_m5['close'].values, timeperiod=ema_slow_period)

# Derived features for later use
df_m5['ema_diff'] = df_m5['ema_fast'] - df_m5['ema_slow']
df_m5['ema_diff_pct'] = (df_m5['ema_diff'] / df_m5['ema_slow']) * 100

print(f"EMA Fast({ema_fast_period}) & Slow({ema_slow_period}) computed.")
print(f"Sample values at last bar:")
print(f"  EMA Fast: {df_m5['ema_fast'].iloc[-1]:.3f}")
print(f"  EMA Slow: {df_m5['ema_slow'].iloc[-1]:.3f}")
print(f"  EMA Diff: {df_m5['ema_diff'].iloc[-1]:.5f}")

# %% [markdown]
# ### 4.2 EMA Crossover Detection

# %%
# Crossover detection matching EA logic:
#   crossOver  = (emaFast[t-1] <= emaSlow[t-1]) && (emaFast[t] > emaSlow[t])
#   crossUnder = (emaFast[t-1] >= emaSlow[t-1]) && (emaFast[t] < emaSlow[t])
# 
# Note: In the EA, bars [1] and [2] are the last two completed bars.
# In our chronological DataFrame, we use current row and .shift(1) for the previous bar.

df_m5['ema_cross_up'] = (
    (df_m5['ema_fast'].shift(1) <= df_m5['ema_slow'].shift(1)) &
    (df_m5['ema_fast'] > df_m5['ema_slow'])
)

df_m5['ema_cross_down'] = (
    (df_m5['ema_fast'].shift(1) >= df_m5['ema_slow'].shift(1)) &
    (df_m5['ema_fast'] < df_m5['ema_slow'])
)

cross_up_count = df_m5['ema_cross_up'].sum()
cross_dn_count = df_m5['ema_cross_down'].sum()
print(f"EMA Crossovers detected:")
print(f"  Bullish crosses (up):   {cross_up_count:,}")
print(f"  Bearish crosses (down): {cross_dn_count:,}")
print(f"  Total: {cross_up_count + cross_dn_count:,} crosses over {len(df_m5):,} bars")

# %% [markdown]
# ### 4.3 Parabolic SAR

# %%
# PSAR parameters from config
psar_start = config['psar']['start']       # 0.02
psar_max   = config['psar']['maximum']     # 0.20
# Note: MQL5 iSAR uses start as both start AND increment
# TA-Lib SAR takes acceleration (=start) and maximum

df_m5['psar'] = talib.SAR(
    df_m5['high'].values,
    df_m5['low'].values,
    acceleration=psar_start,
    maximum=psar_max
)

# PSAR direction: bullish when SAR is below price, bearish when above
df_m5['psar_bullish'] = df_m5['psar'] < df_m5['close']
df_m5['psar_bearish'] = df_m5['psar'] > df_m5['close']
df_m5['psar_distance'] = df_m5['close'] - df_m5['psar']

psar_bull_pct = df_m5['psar_bullish'].mean() * 100
print(f"Parabolic SAR (start={psar_start}, max={psar_max}) computed.")
print(f"  Bullish (SAR below price): {psar_bull_pct:.1f}% of bars")
print(f"  Bearish (SAR above price): {100 - psar_bull_pct:.1f}% of bars")

# %% [markdown]
# ### 4.4 RSI (7-period)

# %%
rsi_period = config['rsi']['period']       # 7
rsi_ob     = config['rsi']['overbought']   # 70
rsi_os     = config['rsi']['oversold']     # 30

df_m5['rsi'] = talib.RSI(df_m5['close'].values, timeperiod=rsi_period)

# Track extreme states
df_m5['rsi_oversold']  = df_m5['rsi'] <= rsi_os
df_m5['rsi_overbought'] = df_m5['rsi'] >= rsi_ob

rsi_os_pct = df_m5['rsi_oversold'].mean() * 100
rsi_ob_pct = df_m5['rsi_overbought'].mean() * 100
print(f"RSI({rsi_period}) computed.")
print(f"  Current: {df_m5['rsi'].iloc[-1]:.2f}")
print(f"  Oversold (≤{rsi_os}):   {rsi_os_pct:.2f}% of bars")
print(f"  Overbought (≥{rsi_ob}): {rsi_ob_pct:.2f}% of bars")

# %% [markdown]
# ### 4.5 Bollinger Bands (20-period, 2.0 StdDev)

# %%
bb_period  = config['bollinger_bands']['period']    # 20
bb_std_dev = config['bollinger_bands']['std_dev']   # 2.0

df_m5['bb_upper'], df_m5['bb_middle'], df_m5['bb_lower'] = talib.BBANDS(
    df_m5['close'].values,
    timeperiod=bb_period,
    nbdevup=bb_std_dev,
    nbdevdn=bb_std_dev,
    matype=0  # 0 = SMA (matches MQL5 iBands default)
)

# Derived BB features
df_m5['bb_width'] = (df_m5['bb_upper'] - df_m5['bb_lower']) / df_m5['bb_middle']
df_m5['bb_pct_b'] = (df_m5['close'] - df_m5['bb_lower']) / (df_m5['bb_upper'] - df_m5['bb_lower'])

# Track band touches
df_m5['bb_below_lower'] = df_m5['low'] <= df_m5['bb_lower']
df_m5['bb_above_upper'] = df_m5['high'] >= df_m5['bb_upper']

bb_lower_touch_pct = df_m5['bb_below_lower'].mean() * 100
bb_upper_touch_pct = df_m5['bb_above_upper'].mean() * 100
print(f"Bollinger Bands ({bb_period}, {bb_std_dev}σ) computed.")
print(f"  Lower band touches: {bb_lower_touch_pct:.2f}% of bars")
print(f"  Upper band touches: {bb_upper_touch_pct:.2f}% of bars")
print(f"  Average band width: {df_m5['bb_width'].mean():.5f}")

# %% [markdown]
# ### 4.6 ADX (14-period)

# %%
adx_period    = config['adx']['period']       # 14
adx_threshold = config['adx']['threshold']    # 30

df_m5['adx'] = talib.ADX(
    df_m5['high'].values,
    df_m5['low'].values,
    df_m5['close'].values,
    timeperiod=adx_period
)

# Also compute +DI and -DI for feature engineering
df_m5['plus_di'] = talib.PLUS_DI(
    df_m5['high'].values,
    df_m5['low'].values,
    df_m5['close'].values,
    timeperiod=adx_period
)

df_m5['minus_di'] = talib.MINUS_DI(
    df_m5['high'].values,
    df_m5['low'].values,
    df_m5['close'].values,
    timeperiod=adx_period
)

df_m5['di_diff'] = df_m5['plus_di'] - df_m5['minus_di']

# Regime classification
df_m5['adx_ok'] = df_m5['adx'] < adx_threshold

adx_ok_pct = df_m5['adx_ok'].mean() * 100
print(f"ADX({adx_period}) computed.")
print(f"  Current ADX: {df_m5['adx'].iloc[-1]:.2f}")
print(f"  Bars with ADX < {adx_threshold} (ranging): {adx_ok_pct:.1f}%")
print(f"  Bars with ADX ≥ {adx_threshold} (trending): {100 - adx_ok_pct:.1f}%")

# %% [markdown]
# ### 4.7 Macro Filter — H1 EMA(200) Mapped to M5

# %%
macro_ema_period = config['macro_filter']['ema_period']  # 200

# Compute 200 EMA on H1 data
df_h1['ema_200'] = talib.EMA(df_h1['close'].values, timeperiod=macro_ema_period)

# Map H1 EMA back to M5 bars using forward-fill (each H1 bar covers 12 M5 bars)
# For each M5 bar, the macro EMA is the last completed H1 bar's EMA value
h1_ema_series = df_h1[['ema_200', 'close']].rename(columns={
    'ema_200': 'macro_ema',
    'close': 'macro_close'
})

# Reindex to M5 timestamps and forward-fill
df_m5['macro_ema'] = h1_ema_series['macro_ema'].reindex(df_m5.index, method='ffill')
df_m5['macro_close'] = h1_ema_series['macro_close'].reindex(df_m5.index, method='ffill')

# Macro direction
df_m5['macro_bullish'] = df_m5['macro_close'] > df_m5['macro_ema']
df_m5['macro_bearish'] = df_m5['macro_close'] < df_m5['macro_ema']
df_m5['macro_distance'] = df_m5['macro_close'] - df_m5['macro_ema']

macro_bull_pct = df_m5['macro_bullish'].mean() * 100
print(f"H1 EMA({macro_ema_period}) macro filter computed and mapped to M5.")
print(f"  Macro bullish (price > EMA): {macro_bull_pct:.1f}%")
print(f"  Macro bearish (price < EMA): {100 - macro_bull_pct:.1f}%")
print(f"  H1 EMA NaN count (warm-up period): {df_m5['macro_ema'].isna().sum()}")

# %% [markdown]
# ## 5. Pivot Point Detection (Swing Highs & Lows)
# 
# This replicates the EA's exact `IsPivotHigh()` / `IsPivotLow()` functions.
# A pivot is confirmed when the candidate bar's high/low is strictly higher/lower
# than all bars within `leftLen` and `rightLen` on either side.

# %%
def detect_pivots(highs, lows, left_len=10, right_len=10):
    """Detect pivot highs and lows matching the EA's exact algorithm.
    
    Parameters:
        highs: numpy array of high prices (chronological order)
        lows: numpy array of low prices (chronological order)
        left_len: number of bars to the left to check
        right_len: number of bars to the right to check
    
    Returns:
        pivot_highs: array of NaN except at pivot high locations (= high price)
        pivot_lows: array of NaN except at pivot low locations (= low price)
    
    Note: Pivots at index i are confirmed only after right_len bars form to the right.
    The last right_len bars of the array will never have pivots.
    """
    n = len(highs)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left_len, n - right_len):
        # --- Pivot High ---
        candidate_h = highs[i]
        is_ph = True
        
        # Check left side (bars before the candidate)
        for j in range(1, left_len + 1):
            if highs[i - j] >= candidate_h:
                is_ph = False
                break
        
        # Check right side (bars after the candidate)
        if is_ph:
            for j in range(1, right_len + 1):
                if highs[i + j] >= candidate_h:
                    is_ph = False
                    break
        
        if is_ph:
            pivot_highs[i] = candidate_h
        
        # --- Pivot Low ---
        candidate_l = lows[i]
        is_pl = True
        
        # Check left side
        for j in range(1, left_len + 1):
            if lows[i - j] <= candidate_l:
                is_pl = False
                break
        
        # Check right side
        if is_pl:
            for j in range(1, right_len + 1):
                if lows[i + j] <= candidate_l:
                    is_pl = False
                    break
        
        if is_pl:
            pivot_lows[i] = candidate_l
    
    return pivot_highs, pivot_lows

# %%
# Detect pivots with EA's default parameters
left_len_h  = config['pivot_points']['left_len_high']    # 10
right_len_h = config['pivot_points']['right_len_high']   # 10
left_len_l  = config['pivot_points']['left_len_low']     # 10
right_len_l = config['pivot_points']['right_len_low']    # 10

print(f"Detecting pivots (Left={left_len_h}, Right={right_len_h})...")
print(f"This may take a moment for {len(df_m5):,} bars...")

pivot_highs, pivot_lows = detect_pivots(
    df_m5['high'].values,
    df_m5['low'].values,
    left_len=left_len_h,
    right_len=right_len_h
)

df_m5['pivot_high'] = pivot_highs
df_m5['pivot_low'] = pivot_lows

# Forward-fill the last known pivot values (this is what the EA does — it stores the last pivot)
df_m5['last_pivot_high'] = df_m5['pivot_high'].ffill()
df_m5['last_pivot_low'] = df_m5['pivot_low'].ffill()

# Compute bars since last pivot
df_m5['is_pivot_high'] = ~df_m5['pivot_high'].isna()
df_m5['is_pivot_low'] = ~df_m5['pivot_low'].isna()

# Bars since last pivot (using cumulative grouping)
df_m5['bars_since_pivot_high'] = df_m5.groupby(df_m5['is_pivot_high'].cumsum()).cumcount()
df_m5['bars_since_pivot_low'] = df_m5.groupby(df_m5['is_pivot_low'].cumsum()).cumcount()

ph_count = df_m5['is_pivot_high'].sum()
pl_count = df_m5['is_pivot_low'].sum()
print(f"\nPivot Detection Results:")
print(f"  Pivot Highs found: {ph_count:,}")
print(f"  Pivot Lows found:  {pl_count:,}")
print(f"  Avg bars between pivot highs: {len(df_m5) / max(ph_count, 1):.1f}")
print(f"  Avg bars between pivot lows:  {len(df_m5) / max(pl_count, 1):.1f}")

# %% [markdown]
# ## 6. Mean Reversion Extreme Lookback
# 
# The EA checks whether price was at an extreme (oversold/overbought) within
# the last `InpExtLookback` bars (default: 5). We compute this as a rolling window.

# %%
ext_lookback = config['mean_reversion']['extreme_lookback']  # 5
rsi_os_level = config['rsi']['oversold']    # 30
rsi_ob_level = config['rsi']['overbought']  # 70

# For each bar, check if any of the previous `ext_lookback` bars (not including current)
# had an oversold or overbought condition
# This matches the EA's loop: for(int k = 1; k <= InpExtLookback; k++)

# Oversold: low <= bb_lower OR rsi <= 30
df_m5['_os_bar'] = ((df_m5['low'] <= df_m5['bb_lower']) | (df_m5['rsi'] <= rsi_os_level)).astype(int)

# Overbought: high >= bb_upper OR rsi >= 70
df_m5['_ob_bar'] = ((df_m5['high'] >= df_m5['bb_upper']) | (df_m5['rsi'] >= rsi_ob_level)).astype(int)

# Rolling sum over lookback window, shifted by 1 (EA starts at k=1, not k=0)
# If any bar in the window had the condition, the sum will be > 0
df_m5['was_oversold'] = df_m5['_os_bar'].shift(1).rolling(window=ext_lookback, min_periods=1).sum() > 0
df_m5['was_overbought'] = df_m5['_ob_bar'].shift(1).rolling(window=ext_lookback, min_periods=1).sum() > 0

# Clean up temp columns
df_m5.drop(columns=['_os_bar', '_ob_bar'], inplace=True)

was_os_pct = df_m5['was_oversold'].mean() * 100
was_ob_pct = df_m5['was_overbought'].mean() * 100
print(f"Mean Reversion Extreme Lookback ({ext_lookback} bars):")
print(f"  Recently oversold:   {was_os_pct:.2f}% of bars")
print(f"  Recently overbought: {was_ob_pct:.2f}% of bars")

# %% [markdown]
# ## 7. Composite Signal Generation
# 
# Combine all conditions exactly as the EA does:
# ```
# buySig  = crossOver  && psarBull && wasExtOS && adxOk && macroOkBull
# sellSig = crossUnder && psarBear && wasExtOB && adxOk && macroOkBear
# ```

# %%
# BUY signal: all 5 conditions must be true
df_m5['buy_signal'] = (
    df_m5['ema_cross_up'] &
    df_m5['psar_bullish'] &
    df_m5['was_oversold'] &
    df_m5['adx_ok'] &
    df_m5['macro_bullish']
)

# SELL signal: all 5 conditions must be true
df_m5['sell_signal'] = (
    df_m5['ema_cross_down'] &
    df_m5['psar_bearish'] &
    df_m5['was_overbought'] &
    df_m5['adx_ok'] &
    df_m5['macro_bearish']
)

# Combined signal column: 1 = buy, -1 = sell, 0 = no signal
df_m5['signal'] = 0
df_m5.loc[df_m5['buy_signal'], 'signal'] = 1
df_m5.loc[df_m5['sell_signal'], 'signal'] = -1

buy_count = df_m5['buy_signal'].sum()
sell_count = df_m5['sell_signal'].sum()
total_bars = len(df_m5)

print(f"{'='*60}")
print(f"SIGNAL GENERATION RESULTS")
print(f"{'='*60}")
print(f"  Total M5 bars:  {total_bars:,}")
print(f"  BUY signals:    {buy_count:,} ({buy_count/total_bars*100:.3f}%)")
print(f"  SELL signals:   {sell_count:,} ({sell_count/total_bars*100:.3f}%)")
print(f"  Total signals:  {buy_count + sell_count:,} ({(buy_count + sell_count)/total_bars*100:.3f}%)")
print(f"  No signal bars: {total_bars - buy_count - sell_count:,} ({(total_bars - buy_count - sell_count)/total_bars*100:.2f}%)")

# %% [markdown]
# ### 7.1 Condition Breakdown — Why Signals Are Rare

# %%
# Show how many bars pass each individual condition
conditions = {
    'EMA Cross Up': df_m5['ema_cross_up'],
    'EMA Cross Down': df_m5['ema_cross_down'],
    'PSAR Bullish': df_m5['psar_bullish'],
    'PSAR Bearish': df_m5['psar_bearish'],
    'Was Oversold': df_m5['was_oversold'],
    'Was Overbought': df_m5['was_overbought'],
    'ADX OK (< 30)': df_m5['adx_ok'],
    'Macro Bullish': df_m5['macro_bullish'],
    'Macro Bearish': df_m5['macro_bearish'],
}

print(f"{'Condition':<25} {'True Bars':>12} {'Percentage':>12}")
print(f"{'-'*50}")
for name, cond in conditions.items():
    true_count = cond.sum()
    print(f"{name:<25} {true_count:>12,} {true_count/total_bars*100:>11.2f}%")

print(f"\n{'─'*50}")
print(f"{'BUY SIGNAL (all 5 AND)': <25} {buy_count:>12,} {buy_count/total_bars*100:>11.3f}%")
print(f"{'SELL SIGNAL (all 5 AND)': <25} {sell_count:>12,} {sell_count/total_bars*100:>11.3f}%")

# %% [markdown]
# ### 7.2 Condition Proximity — How Many Conditions Met at Each Bar

# %%
# Count how many buy conditions are met at each bar (0-5)
df_m5['buy_conditions_met'] = (
    df_m5['ema_cross_up'].astype(int) +
    df_m5['psar_bullish'].astype(int) +
    df_m5['was_oversold'].astype(int) +
    df_m5['adx_ok'].astype(int) +
    df_m5['macro_bullish'].astype(int)
)

df_m5['sell_conditions_met'] = (
    df_m5['ema_cross_down'].astype(int) +
    df_m5['psar_bearish'].astype(int) +
    df_m5['was_overbought'].astype(int) +
    df_m5['adx_ok'].astype(int) +
    df_m5['macro_bearish'].astype(int)
)

print("Buy Conditions Met Distribution:")
print(df_m5['buy_conditions_met'].value_counts().sort_index().to_string())
print(f"\nSell Conditions Met Distribution:")
print(df_m5['sell_conditions_met'].value_counts().sort_index().to_string())

# %% [markdown]
# ## 8. Additional Features (ATR, Candle Patterns, Temporal)

# %%
# ATR (Average True Range) — for volatility normalization
df_m5['atr_14'] = talib.ATR(df_m5['high'].values, df_m5['low'].values, 
                              df_m5['close'].values, timeperiod=14)
df_m5['atr_ratio'] = df_m5['atr_14'] / df_m5['atr_14'].rolling(100).mean()

# Candle body and wick analysis
df_m5['body_size'] = (df_m5['close'] - df_m5['open']).abs()
df_m5['candle_range'] = df_m5['high'] - df_m5['low']
df_m5['body_ratio'] = df_m5['body_size'] / df_m5['candle_range'].replace(0, np.nan)
df_m5['upper_wick'] = df_m5['high'] - df_m5[['open', 'close']].max(axis=1)
df_m5['lower_wick'] = df_m5[['open', 'close']].min(axis=1) - df_m5['low']
df_m5['upper_wick_ratio'] = df_m5['upper_wick'] / df_m5['candle_range'].replace(0, np.nan)
df_m5['lower_wick_ratio'] = df_m5['lower_wick'] / df_m5['candle_range'].replace(0, np.nan)

# Temporal features (cyclical encoding)
df_m5['hour'] = df_m5.index.hour
df_m5['minute'] = df_m5.index.minute
df_m5['day_of_week'] = df_m5.index.dayofweek  # 0=Monday, 4=Friday

# Cyclical encoding for hour and day
df_m5['hour_sin'] = np.sin(2 * np.pi * df_m5['hour'] / 24)
df_m5['hour_cos'] = np.cos(2 * np.pi * df_m5['hour'] / 24)
df_m5['dow_sin'] = np.sin(2 * np.pi * df_m5['day_of_week'] / 5)
df_m5['dow_cos'] = np.cos(2 * np.pi * df_m5['day_of_week'] / 5)

# Trading session classification
def classify_session(hour):
    """Classify into major trading sessions (server time)."""
    if 0 <= hour < 8:
        return 'asian'
    elif 8 <= hour < 13:
        return 'london'
    elif 13 <= hour < 17:
        return 'overlap'  # London + New York overlap
    elif 17 <= hour < 22:
        return 'new_york'
    else:
        return 'off_hours'

df_m5['session'] = df_m5['hour'].apply(classify_session)

# Structural features
df_m5['dist_to_pivot_high'] = df_m5['close'] - df_m5['last_pivot_high']
df_m5['dist_to_pivot_low'] = df_m5['close'] - df_m5['last_pivot_low']
df_m5['swing_range'] = df_m5['last_pivot_high'] - df_m5['last_pivot_low']
df_m5['price_in_swing_pct'] = np.where(
    df_m5['swing_range'] > 0,
    (df_m5['close'] - df_m5['last_pivot_low']) / df_m5['swing_range'],
    0.5
)

print("Additional features computed:")
print(f"  ATR(14), ATR ratio")
print(f"  Candle body/wick analysis (6 features)")
print(f"  Temporal features: hour, day, session (cyclical encoded)")
print(f"  Structural features: distance to pivots, swing range, price position")

# %% [markdown]
# ## 9. Dataset Summary & Quality Check

# %%
# Final summary
print(f"{'='*70}")
print(f"PROCESSED DATASET SUMMARY")
print(f"{'='*70}")
print(f"  Shape: {df_m5.shape}")
print(f"  Date range: {df_m5.index[0]} → {df_m5.index[-1]}")
print(f"  Memory usage: {df_m5.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\n  Column count by category:")

# Group columns by category
categories = {
    'OHLCV': ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread'],
    'EMA': [c for c in df_m5.columns if 'ema' in c and 'macro' not in c],
    'PSAR': [c for c in df_m5.columns if 'psar' in c],
    'RSI': [c for c in df_m5.columns if 'rsi' in c],
    'BB': [c for c in df_m5.columns if 'bb_' in c],
    'ADX': [c for c in df_m5.columns if 'adx' in c or 'di_' in c or 'plus_di' in c or 'minus_di' in c],
    'Macro': [c for c in df_m5.columns if 'macro' in c],
    'Pivots': [c for c in df_m5.columns if 'pivot' in c or 'swing' in c],
    'Signals': ['buy_signal', 'sell_signal', 'signal', 'buy_conditions_met', 'sell_conditions_met',
                'was_oversold', 'was_overbought'],
    'Market Structure': ['atr_14', 'atr_ratio', 'body_size', 'candle_range', 'body_ratio',
                         'upper_wick', 'lower_wick', 'upper_wick_ratio', 'lower_wick_ratio'],
    'Temporal': ['hour', 'minute', 'day_of_week', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'session'],
    'Structural': ['dist_to_pivot_high', 'dist_to_pivot_low', 'swing_range', 'price_in_swing_pct'],
}

for cat, cols in categories.items():
    existing = [c for c in cols if c in df_m5.columns]
    print(f"    {cat}: {len(existing)} columns")

# %%
# NaN report (important — indicators have warm-up periods)
nan_report = df_m5.isna().sum()
nan_cols = nan_report[nan_report > 0].sort_values(ascending=False)
print(f"\nColumns with NaN values (indicator warm-up):")
print(f"{'Column':<30} {'NaN Count':>10} {'% of Data':>10}")
print(f"{'-'*50}")
for col, count in nan_cols.items():
    print(f"{col:<30} {count:>10,} {count/len(df_m5)*100:>9.2f}%")

# Drop rows where critical indicators are NaN (warm-up period)
# The macro EMA (200 H1) has the longest warm-up
df_clean = df_m5.dropna(subset=['ema_fast', 'ema_slow', 'psar', 'rsi', 'bb_upper', 'adx', 'macro_ema'])
rows_dropped = len(df_m5) - len(df_clean)
print(f"\nRows dropped due to indicator warm-up: {rows_dropped:,}")
print(f"Clean dataset: {len(df_clean):,} bars")

# %% [markdown]
# ## 10. Save Processed Data

# %%
# Save full processed M5 dataset
output_path = PROCESSED_DIR / "XAUUSD_M5_processed.parquet"
df_clean.to_parquet(output_path, engine='pyarrow')
print(f"✅ Processed M5 data saved to: {output_path}")
print(f"   Shape: {df_clean.shape}")
print(f"   File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

# Save H1 data with macro EMA (for reference/validation)
h1_output = PROCESSED_DIR / "XAUUSD_H1_processed.parquet"
df_h1.to_parquet(h1_output, engine='pyarrow')
print(f"✅ Processed H1 data saved to: {h1_output}")

# Save signal-only subset for quick analysis
df_signals = df_clean[df_clean['signal'] != 0].copy()
signals_path = PROCESSED_DIR / "XAUUSD_M5_signals_only.parquet"
df_signals.to_parquet(signals_path, engine='pyarrow')
print(f"✅ Signals-only subset saved: {len(df_signals)} signals")

# %% [markdown]
# ## 11. Visual Sanity Check — Plot Indicators on a Sample Window

# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Pick a sample window with at least one signal if possible
signal_indices = df_clean[df_clean['signal'] != 0].index
if len(signal_indices) > 0:
    # Find a signal in the middle of the dataset
    mid_signal = signal_indices[len(signal_indices) // 2]
    # Get 200 bars around it
    signal_loc = df_clean.index.get_loc(mid_signal)
    start_idx = max(0, signal_loc - 100)
    end_idx = min(len(df_clean), signal_loc + 100)
    sample = df_clean.iloc[start_idx:end_idx]
else:
    # No signals found — just use a sample window
    sample = df_clean.iloc[1000:1200]

fig, axes = plt.subplots(5, 1, figsize=(18, 20), sharex=True, 
                          gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
fig.suptitle(f'EPR Mean Reversion Sniper — Indicator Verification\n{sample.index[0]} to {sample.index[-1]}', 
             fontsize=14, fontweight='bold')

# --- Panel 1: Price + EMAs + Bollinger Bands + PSAR + Pivots + Signals ---
ax1 = axes[0]
ax1.plot(sample.index, sample['close'], color='white', linewidth=0.8, label='Close', zorder=2)
ax1.plot(sample.index, sample['ema_fast'], color='#00BFFF', linewidth=1, label=f'EMA({ema_fast_period})', alpha=0.9)
ax1.plot(sample.index, sample['ema_slow'], color='#FF6347', linewidth=1, label=f'EMA({ema_slow_period})', alpha=0.9)
ax1.fill_between(sample.index, sample['bb_upper'], sample['bb_lower'], alpha=0.1, color='#888888', label='BB Band')
ax1.plot(sample.index, sample['bb_upper'], color='#888888', linewidth=0.5, linestyle='--')
ax1.plot(sample.index, sample['bb_lower'], color='#888888', linewidth=0.5, linestyle='--')

# PSAR dots
psar_bull = sample[sample['psar_bullish']]
psar_bear = sample[sample['psar_bearish']]
ax1.scatter(psar_bull.index, psar_bull['psar'], color='#00FF7F', s=3, alpha=0.6, zorder=1)
ax1.scatter(psar_bear.index, psar_bear['psar'], color='#FF4444', s=3, alpha=0.6, zorder=1)

# Pivot points
ph = sample[sample['is_pivot_high']]
pl = sample[sample['is_pivot_low']]
ax1.scatter(ph.index, ph['pivot_high'], marker='v', color='red', s=80, zorder=5, label='Pivot High')
ax1.scatter(pl.index, pl['pivot_low'], marker='^', color='lime', s=80, zorder=5, label='Pivot Low')

# Trade signals
buys = sample[sample['buy_signal']]
sells = sample[sample['sell_signal']]
ax1.scatter(buys.index, buys['close'], marker='^', color='#00FF00', s=200, zorder=6, 
            edgecolors='white', linewidths=1.5, label='BUY Signal')
ax1.scatter(sells.index, sells['close'], marker='v', color='#FF0000', s=200, zorder=6,
            edgecolors='white', linewidths=1.5, label='SELL Signal')

ax1.set_ylabel('Price')
ax1.legend(loc='upper left', fontsize=8, ncol=4)
ax1.set_facecolor('#1a1a2e')
ax1.grid(True, alpha=0.2)

# --- Panel 2: RSI ---
ax2 = axes[1]
ax2.plot(sample.index, sample['rsi'], color='#FFD700', linewidth=1)
ax2.axhline(y=rsi_ob, color='red', linestyle='--', alpha=0.5, label=f'OB ({rsi_ob})')
ax2.axhline(y=rsi_os, color='green', linestyle='--', alpha=0.5, label=f'OS ({rsi_os})')
ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.3)
ax2.fill_between(sample.index, rsi_ob, 100, alpha=0.1, color='red')
ax2.fill_between(sample.index, 0, rsi_os, alpha=0.1, color='green')
ax2.set_ylabel(f'RSI({rsi_period})')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper right', fontsize=8)
ax2.set_facecolor('#1a1a2e')
ax2.grid(True, alpha=0.2)

# --- Panel 3: ADX ---
ax3 = axes[2]
ax3.plot(sample.index, sample['adx'], color='#FF69B4', linewidth=1, label='ADX')
ax3.plot(sample.index, sample['plus_di'], color='#00FF7F', linewidth=0.7, alpha=0.7, label='+DI')
ax3.plot(sample.index, sample['minus_di'], color='#FF4444', linewidth=0.7, alpha=0.7, label='-DI')
ax3.axhline(y=adx_threshold, color='yellow', linestyle='--', alpha=0.5, label=f'Threshold ({adx_threshold})')
ax3.set_ylabel(f'ADX({adx_period})')
ax3.legend(loc='upper right', fontsize=8)
ax3.set_facecolor('#1a1a2e')
ax3.grid(True, alpha=0.2)

# --- Panel 4: EMA Difference ---
ax4 = axes[3]
colors = ['#00FF7F' if v > 0 else '#FF4444' for v in sample['ema_diff']]
ax4.bar(sample.index, sample['ema_diff'], color=colors, width=0.003, alpha=0.7)
ax4.axhline(y=0, color='white', linewidth=0.5)
ax4.set_ylabel('EMA Diff')
ax4.set_facecolor('#1a1a2e')
ax4.grid(True, alpha=0.2)

# --- Panel 5: Conditions Met ---
ax5 = axes[4]
ax5.bar(sample.index, sample['buy_conditions_met'], color='#00FF7F', alpha=0.5, width=0.003, label='Buy Conds')
ax5.bar(sample.index, -sample['sell_conditions_met'], color='#FF4444', alpha=0.5, width=0.003, label='Sell Conds')
ax5.axhline(y=5, color='lime', linestyle='--', alpha=0.3, linewidth=0.5)
ax5.axhline(y=-5, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
ax5.set_ylabel('Conditions Met')
ax5.set_xlabel('Time')
ax5.legend(loc='upper right', fontsize=8)
ax5.set_facecolor('#1a1a2e')
ax5.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(str(PROCESSED_DIR / 'indicator_verification_chart.png'), dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("✅ Verification chart saved to data/processed/indicator_verification_chart.png")

# %% [markdown]
# ## 12. Final Statistics for Next Phase

# %%
print(f"\n{'='*70}")
print(f"PHASE 1 COMPLETE — READY FOR PHASE 2 (LABEL GENERATION)")
print(f"{'='*70}")
print(f"\nDataset: {output_path}")
print(f"Shape:   {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
print(f"Period:  {df_clean.index[0]} → {df_clean.index[-1]}")
print(f"\nSignal Summary:")
print(f"  BUY signals:  {df_clean['buy_signal'].sum():,}")
print(f"  SELL signals: {df_clean['sell_signal'].sum():,}")
print(f"  Total:        {(df_clean['signal'] != 0).sum():,}")
print(f"\nIndicator Columns: {len([c for c in df_clean.columns if c not in ['open','high','low','close','tick_volume','real_volume','spread']])}")
print(f"\nNext step: Run 02_label_generation.ipynb to simulate trades")
print(f"  → For each signal, walk forward to determine WIN/LOSS/TIMEOUT")
print(f"  → Compute SL, TP, trade duration, MFE, MAE")
