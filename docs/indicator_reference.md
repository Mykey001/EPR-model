# Indicator Reference — Exact Computation Details

This document specifies the exact mathematical computation for each indicator used by the
EPR Mean Reversion Sniper strategy. These formulas must be replicated identically in Python
to ensure the ML labels match the EA's behavior.

---

## 1. Exponential Moving Average (EMA)

**Used for:** Fast EMA (5), Slow EMA (15), Macro EMA (200)

### Formula

```
multiplier = 2 / (period + 1)
EMA[t] = Close[t] × multiplier + EMA[t-1] × (1 - multiplier)
```

**Initialization:** The first EMA value is typically an SMA (Simple Moving Average) of the first `period` bars.

### Python (pandas):
```python
ema = df['close'].ewm(span=period, adjust=False).mean()
```

### Python (ta-lib):
```python
import talib
ema = talib.EMA(df['close'].values, timeperiod=period)
```

---

## 2. Parabolic SAR (Stop and Reverse)

**Parameters:** Start=0.02, Increment=0.02, Maximum=0.20

### Algorithm

The PSAR is a state machine with two states: uptrend and downtrend.

**Uptrend state:**
```
SAR[t] = SAR[t-1] + AF × (EP - SAR[t-1])
```
Where:
- `AF` = Acceleration Factor, starts at `Start`, increments by `Increment` each time a new high is made, capped at `Maximum`
- `EP` = Extreme Point, the highest high seen during the current uptrend

**Downtrend state:**
```
SAR[t] = SAR[t-1] + AF × (EP - SAR[t-1])
```
Where:
- `EP` = the lowest low seen during the current downtrend

**Reversal:** When price crosses the SAR level, the trend flips, AF resets to `Start`, and EP resets.

### Python (ta-lib):
```python
sar = talib.SAR(df['high'].values, df['low'].values, 
                acceleration=0.02, maximum=0.20)
```

> **Note:** The MQL5 `iSAR()` function takes `start` and `max` but NOT `increment` directly — the increment equals the start value. Ensure Python implementation matches.

---

## 3. Relative Strength Index (RSI)

**Parameters:** Period=7, OB=70, OS=30

### Formula

```
RS = Average Gain over period / Average Loss over period
RSI = 100 - (100 / (1 + RS))
```

**Smoothing method (Wilder's):**
```
AvgGain[t] = (AvgGain[t-1] × (period-1) + currentGain) / period
AvgLoss[t] = (AvgLoss[t-1] × (period-1) + currentLoss) / period
```

### Python (ta-lib):
```python
rsi = talib.RSI(df['close'].values, timeperiod=7)
```

---

## 4. Bollinger Bands

**Parameters:** Period=20, StdDev Multiplier=2.0, Price=Close

### Formula

```
Middle Band = SMA(Close, 20)
Upper Band  = Middle Band + 2.0 × StdDev(Close, 20)
Lower Band  = Middle Band - 2.0 × StdDev(Close, 20)
```

Where `StdDev` is the **population** standard deviation (ddof=0).

### Python (ta-lib):
```python
upper, middle, lower = talib.BBANDS(df['close'].values, 
                                     timeperiod=20, 
                                     nbdevup=2.0, 
                                     nbdevdn=2.0, 
                                     matype=0)  # 0 = SMA
```

### Python (pandas):
```python
middle = df['close'].rolling(20).mean()
std = df['close'].rolling(20).std(ddof=0)
upper = middle + 2.0 * std
lower = middle - 2.0 * std
```

---

## 5. Average Directional Index (ADX)

**Parameters:** Period=14

### Formula

The ADX is derived from the Directional Movement system:

**Step 1 — Directional Movement:**
```
+DM = High[t] - High[t-1]   if > 0 and > (Low[t-1] - Low[t]),  else 0
-DM = Low[t-1] - Low[t]     if > 0 and > (High[t] - High[t-1]), else 0
```

**Step 2 — True Range:**
```
TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
```

**Step 3 — Smoothed values (Wilder's smoothing, 14-period):**
```
Smoothed +DM = Previous +DM - (Previous +DM / 14) + Current +DM
Smoothed -DM = Previous -DM - (Previous -DM / 14) + Current -DM
Smoothed TR  = Previous TR  - (Previous TR / 14)  + Current TR
```

**Step 4 — Directional Indicators:**
```
+DI = 100 × (Smoothed +DM / Smoothed TR)
-DI = 100 × (Smoothed -DM / Smoothed TR)
```

**Step 5 — DX and ADX:**
```
DX  = 100 × |+DI - -DI| / (+DI + -DI)
ADX = Wilder's smoothed average of DX over 14 periods
```

### Python (ta-lib):
```python
adx = talib.ADX(df['high'].values, df['low'].values, 
                df['close'].values, timeperiod=14)
```

---

## 6. Pivot Point Detection (Custom)

**Parameters:** leftLen=10, rightLen=10

### Algorithm

```python
def is_pivot_high(highs, offset, left_len, right_len):
    candidate = highs[offset]
    for i in range(1, left_len + 1):
        if highs[offset + i] >= candidate:
            return False
    for i in range(1, right_len + 1):
        if highs[offset - i] >= candidate:
            return False
    return True

def is_pivot_low(lows, offset, left_len, right_len):
    candidate = lows[offset]
    for i in range(1, left_len + 1):
        if lows[offset + i] <= candidate:
            return False
    for i in range(1, right_len + 1):
        if lows[offset - i] <= candidate:
            return False
    return True
```

**Important indexing note:** In the EA, arrays are set as series (index 0 = most recent bar). The pivot is checked at index `rightLen`, meaning bar index 10 is the candidate, with bars 0–9 to its right and bars 11–20 to its left.

In a pandas DataFrame (chronological order, index 0 = oldest), the equivalent is to check bar `len(df) - 1 - rightLen` and look both directions.

### Vectorized Python:
```python
def detect_pivots(df, left_len=10, right_len=10):
    highs = df['high'].values
    lows = df['low'].values
    n = len(highs)
    
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left_len, n - right_len):
        # Pivot High
        is_ph = True
        for j in range(1, left_len + 1):
            if highs[i - j] >= highs[i]:
                is_ph = False
                break
        if is_ph:
            for j in range(1, right_len + 1):
                if highs[i + j] >= highs[i]:
                    is_ph = False
                    break
        if is_ph:
            pivot_highs[i] = highs[i]
        
        # Pivot Low
        is_pl = True
        for j in range(1, left_len + 1):
            if lows[i - j] <= lows[i]:
                is_pl = False
                break
        if is_pl:
            for j in range(1, right_len + 1):
                if lows[i + j] <= lows[i]:
                    is_pl = False
                    break
        if is_pl:
            pivot_lows[i] = lows[i]
    
    return pivot_highs, pivot_lows
```

---

## 7. EMA Crossover Detection

### Formula

```python
cross_over  = (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
cross_under = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)
```

Note: In the EA, bars [1] and [2] are used (both completed). In pandas with chronological order, this translates to checking the last row and the row before it against their shifted values.

---

## 8. Multi-Timeframe EMA (Macro Filter)

The H1 200-EMA is calculated on the H1 timeframe and then compared against the H1 close price.

### Python approach:
```python
# Resample to H1 if working with lower timeframe data
h1_close = df['close'].resample('1h').last()
h1_ema200 = h1_close.ewm(span=200, adjust=False).mean()

# Map back to original timeframe
df['macro_ema'] = h1_ema200.reindex(df.index, method='ffill')
df['macro_close'] = h1_close.reindex(df.index, method='ffill')
```
