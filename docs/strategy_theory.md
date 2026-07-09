# EPR Mean Reversion Sniper — Complete Strategy Theory

## 1. Strategy Identity

| Field | Value |
|-------|-------|
| **Name** | Combined HL-EPR Mean Reversion Sniper EA |
| **Version** | 2.0 (converted from PineScript) |
| **Type** | Mean Reversion with Momentum Confirmation |
| **Execution** | Bar-close logic (once per completed candle) |
| **Origin** | PineScript → MQL5 conversion |

### Core Thesis

> Price stretches to a statistical extreme (oversold/overbought) → momentum indicators confirm the reversal has begun → enter the trade in the direction of the macro trend → place stop loss at structural swing level → target a fixed risk-reward multiple.

---

## 2. Indicator Stack

The strategy uses **7 indicators** across **4 functional layers**:

### Layer 1: Structure Detection — Pivot Points

**Purpose:** Identify structural swing highs and lows for stop-loss placement.

**Algorithm:** A Pivot High at bar index `offset` is confirmed when:
- `high[offset]` is **strictly greater** than `high[offset+1]` through `high[offset+leftLen]` (left side)
- `high[offset]` is **strictly greater** than `high[offset-1]` through `high[offset-rightLen]` (right side)

Pivot Low is the mirror (strictly less than all surrounding lows).

**Default Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `InpLeftLenH` | 10 | Bars to the left for pivot high |
| `InpRightLenH` | 10 | Bars to the right for pivot high |
| `InpLeftLenL` | 10 | Bars to the left for pivot low |
| `InpRightLenL` | 10 | Bars to the right for pivot low |

**Confirmation Delay:** Pivots are only confirmed after `rightLen` bars have formed to the right. With default settings, this means a 10-bar lag.

**Usage:** The most recent confirmed pivot high/low is stored and used to calculate the swing-based stop loss when a signal triggers.

---

### Layer 2: Entry Trigger — EMA Crossover

**Purpose:** Detect the moment momentum shifts direction, signaling the mean reversion has begun.

**Indicators:**
| Indicator | Period | Price Source |
|-----------|--------|-------------|
| Fast EMA | 5 | Close |
| Slow EMA | 15 | Close |

**Crossover Detection (on completed bars [1] and [2]):**

```
Bullish Cross (crossOver):  emaFast[2] <= emaSlow[2]  AND  emaFast[1] > emaSlow[1]
Bearish Cross (crossUnder): emaFast[2] >= emaSlow[2]  AND  emaFast[1] < emaSlow[1]
```

**Role:** This is the **primary trigger**. No trade signal fires without an EMA crossover. All other conditions are confirmations/filters.

---

### Layer 3: Directional Confirmation — Parabolic SAR

**Purpose:** Confirm that the trend direction agrees with the trade.

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Start | 0.02 |
| Increment | 0.02 |
| Maximum | 0.20 |

**Logic:**
```
psarBull = SAR[1] < Close[1]    → SAR dots below price → bullish
psarBear = SAR[1] > Close[1]    → SAR dots above price → bearish
```

**Role:** Prevents buying when SAR indicates downtrend, and selling when SAR indicates uptrend.

---

### Layer 4A: Mean Reversion Qualifier — RSI + Bollinger Bands

**Purpose:** Confirm that price was recently at a statistical extreme (the "reversion from" condition).

#### RSI Parameters:
| Parameter | Value |
|-----------|-------|
| Period | 7 |
| Overbought | 70 |
| Oversold | 30 |

#### Bollinger Bands Parameters:
| Parameter | Value |
|-----------|-------|
| Period | 20 |
| StdDev Multiplier | 2.0 |
| Price Source | Close |

#### Lookback Window:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `InpExtLookback` | 5 | Number of bars to search for extreme condition |

#### Extreme Detection Logic:

The EA scans bars `[1]` through `[InpExtLookback]` (default: bars 1–5):

**Oversold extreme (for BUY signals):**
```
wasExtOS = true  IF  any bar k in [1..5] has:
    low[k] <= bbLower[k]   OR   RSI[k] <= 30
```

**Overbought extreme (for SELL signals):**
```
wasExtOB = true  IF  any bar k in [1..5] has:
    high[k] >= bbUpper[k]   OR   RSI[k] >= 70
```

**Key insight:** The extreme does NOT need to be active right now. It just needs to have occurred within the lookback window. This allows the entry to happen on the "bounce back" after the extreme.

---

### Layer 4B: Regime Filter — ADX

**Purpose:** Block trades when the market is in a strong trend (mean reversion fails in trends).

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Period | 14 |
| Threshold | 30 |

**Logic:**
```
adxOk = ADX[1] < 30    (or ADX filter is disabled)
```

| ADX Range | Interpretation | Trade Allowed? |
|-----------|---------------|---------------|
| 0–20 | No/weak trend | ✅ Ideal |
| 20–30 | Moderate trend | ✅ Acceptable |
| 30+ | Strong trend | ❌ Blocked |

---

### Layer 4C: Macro Bias — Higher Timeframe EMA

**Purpose:** Ensure trades align with the larger market direction.

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Timeframe | H1 |
| EMA Period | 200 |

**Logic:**
```
macroOkBull = H1_Close > H1_EMA200    → Buys allowed
macroOkBear = H1_Close < H1_EMA200    → Sells allowed
```

**Note:** The macro close uses bar index [0] (current, possibly forming) while the EMA uses bar index [1] (last completed). This gives the freshest directional read but uses unconfirmed data.

---

## 3. Complete Signal Logic

### BUY Signal — ALL conditions must be TRUE:

| # | Condition | Indicator | Check |
|---|-----------|-----------|-------|
| 1 | Momentum shift up | EMA Cross | Fast EMA crossed above Slow EMA |
| 2 | Direction confirms up | PSAR | SAR below price |
| 3 | Was oversold recently | RSI/BB | RSI ≤ 30 or price ≤ lower BB within last 5 bars |
| 4 | Not trending strongly | ADX | ADX < 30 |
| 5 | Macro trend is bullish | H1 EMA200 | H1 close above H1 200-EMA |

### SELL Signal — ALL conditions must be TRUE:

| # | Condition | Indicator | Check |
|---|-----------|-----------|-------|
| 1 | Momentum shift down | EMA Cross | Fast EMA crossed below Slow EMA |
| 2 | Direction confirms down | PSAR | SAR above price |
| 3 | Was overbought recently | RSI/BB | RSI ≥ 70 or price ≥ upper BB within last 5 bars |
| 4 | Not trending strongly | ADX | ADX < 30 |
| 5 | Macro trend is bearish | H1 EMA200 | H1 close below H1 200-EMA |

### Position Guard:
- Only ONE buy position allowed at a time (per magic number + symbol)
- Only ONE sell position allowed at a time
- Hedging IS allowed (simultaneous buy + sell)

---

## 4. Risk Management

### Stop Loss

**Primary method — Swing Point SL (default ON):**

| Trade | SL Formula |
|-------|-----------|
| BUY | `SL = lastPivotLow - (bufferPips × pipSize)` |
| SELL | `SL = lastPivotHigh + (bufferPips × pipSize)` |

Default buffer = 5 pips. `pipSize = _Point × 10`.

**Fallback — Fixed Pip SL:**

Used when swing SL is disabled or when the swing level is stale (on the wrong side of entry):

| Trade | SL Formula |
|-------|-----------|
| BUY | `SL = entry - (50 × pipSize)` |
| SELL | `SL = entry + (50 × pipSize)` |

### Take Profit

TP is always derived from the SL distance:

```
slDistance = |entry - SL|
tpDistance = slDistance × riskRewardRatio    (default: 2.0)

BUY TP  = entry + tpDistance
SELL TP = entry - tpDistance
```

Default risk-reward = 1:2 (risk 1 unit to gain 2 units).

### Position Sizing

Fixed lot size (default 0.1). No dynamic risk percentage calculation.

---

## 5. What the Strategy Does NOT Have

| Missing Feature | Impact |
|----------------|--------|
| No trailing stop | Cannot lock in profits during favorable moves |
| No breakeven logic | Cannot move SL to breakeven after partial profit |
| No partial close | Cannot scale out of winning positions |
| No session filter | Trades during low-liquidity hours |
| No spread filter | May enter during wide-spread conditions |
| No news filter | Trades through high-impact news events |
| No dynamic position sizing | Cannot adapt risk to account equity or signal quality |
| No early exit logic | Relies entirely on SL/TP — no indicator-based exit |

These gaps represent the **ML enhancement opportunities** for this project.

---

## 6. Default Parameter Summary

```yaml
# Pivot Points
pivot_left_high: 10
pivot_right_high: 10
pivot_left_low: 10
pivot_right_low: 10

# EMA
ema_fast: 5
ema_slow: 15

# Parabolic SAR
psar_start: 0.02
psar_increment: 0.02
psar_max: 0.20

# RSI
rsi_period: 7
rsi_overbought: 70
rsi_oversold: 30

# Bollinger Bands
bb_period: 20
bb_stddev: 2.0

# ADX
adx_period: 14
adx_threshold: 30

# Macro Filter
macro_timeframe: H1
macro_ema_period: 200

# Mean Reversion Lookback
extreme_lookback: 5

# Trade Management
lot_size: 0.1
use_swing_sl: true
swing_sl_buffer_pips: 5.0
risk_reward_ratio: 2.0
fixed_sl_pips: 50.0
```
