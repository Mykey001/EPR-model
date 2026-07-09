# Signal Logic — Formal Specification

This document provides a formal, unambiguous specification of the signal logic
that the ML model must learn to replicate (and then improve upon).

---

## Signal State Machine

```
On each new bar:
    1. Detect pivots (10-bar confirmed)
    2. Store latest pivot high/low prices
    3. Compute all indicators on completed bars
    4. Check BUY/SELL conditions
    5. If signal AND no existing position in that direction → OPEN TRADE
```

---

## BUY Signal — Formal Definition

```python
BUY_SIGNAL = (
    # Trigger: EMA crossover on bars [t-1] and [t-2]
    (ema_fast[t-2] <= ema_slow[t-2]) and (ema_fast[t-1] > ema_slow[t-1])
    
    # Confirmation: PSAR bullish
    and (psar[t-1] < close[t-1])
    
    # Mean reversion: was oversold within lookback
    and any(
        (low[t-k] <= bb_lower[t-k]) or (rsi[t-k] <= 30)
        for k in range(1, 6)  # bars t-1 through t-5
    )
    
    # Regime filter: not trending
    and (adx[t-1] < 30)  # or ADX filter disabled
    
    # Macro filter: bullish bias on H1
    and (h1_close[current] > h1_ema200[t-1])  # or macro filter disabled
    
    # Position guard
    and (no_existing_buy_position)
)
```

---

## SELL Signal — Formal Definition

```python
SELL_SIGNAL = (
    # Trigger: EMA crossunder on bars [t-1] and [t-2]
    (ema_fast[t-2] >= ema_slow[t-2]) and (ema_fast[t-1] < ema_slow[t-1])
    
    # Confirmation: PSAR bearish
    and (psar[t-1] > close[t-1])
    
    # Mean reversion: was overbought within lookback
    and any(
        (high[t-k] >= bb_upper[t-k]) or (rsi[t-k] >= 70)
        for k in range(1, 6)  # bars t-1 through t-5
    )
    
    # Regime filter: not trending
    and (adx[t-1] < 30)  # or ADX filter disabled
    
    # Macro filter: bearish bias on H1
    and (h1_close[current] < h1_ema200[t-1])  # or macro filter disabled
    
    # Position guard
    and (no_existing_sell_position)
)
```

---

## Stop Loss Rules

### Swing SL (default):
```python
if direction == BUY:
    sl = last_pivot_low - (5.0 * pip_size)
    if sl >= entry_price:  # stale pivot
        sl = entry_price - (50.0 * pip_size)  # fallback

if direction == SELL:
    sl = last_pivot_high + (5.0 * pip_size)
    if sl <= entry_price:  # stale pivot
        sl = entry_price + (50.0 * pip_size)  # fallback
```

### Fixed SL (when swing SL disabled):
```python
if direction == BUY:
    sl = entry_price - (50.0 * pip_size)
if direction == SELL:
    sl = entry_price + (50.0 * pip_size)
```

---

## Take Profit Rules

```python
sl_distance = abs(entry_price - sl_price)
tp_distance = sl_distance * risk_reward_ratio  # default 2.0

if direction == BUY:
    tp = entry_price + tp_distance
if direction == SELL:
    tp = entry_price - tp_distance
```

---

## Trade Outcome Classification

For ML labeling purposes, each trade resolves as one of:

| Outcome | Definition |
|---------|-----------|
| **WIN** | Price reached TP before SL |
| **LOSS** | Price reached SL before TP |
| **TIMEOUT** | Neither SL nor TP hit within analysis window (edge case) |

---

## Individual Condition Flags (for Feature Engineering)

Each sub-condition should be tracked as an independent binary feature:

| Feature | Type | Description |
|---------|------|-------------|
| `ema_cross_up` | bool | Fast EMA crossed above slow EMA |
| `ema_cross_down` | bool | Fast EMA crossed below slow EMA |
| `psar_bullish` | bool | SAR below price |
| `psar_bearish` | bool | SAR above price |
| `was_oversold` | bool | OS extreme within lookback |
| `was_overbought` | bool | OB extreme within lookback |
| `adx_ok` | bool | ADX below threshold |
| `macro_bullish` | bool | H1 close above H1 EMA200 |
| `macro_bearish` | bool | H1 close below H1 EMA200 |
| `buy_signal` | bool | All buy conditions met |
| `sell_signal` | bool | All sell conditions met |

Additionally, the continuous values should be stored:

| Feature | Type | Description |
|---------|------|-------------|
| `ema_fast` | float | Current fast EMA value |
| `ema_slow` | float | Current slow EMA value |
| `ema_diff` | float | `ema_fast - ema_slow` |
| `ema_diff_pct` | float | `(ema_fast - ema_slow) / ema_slow × 100` |
| `psar_value` | float | Current PSAR value |
| `psar_distance` | float | `close - psar` (positive = bullish) |
| `rsi_value` | float | Current RSI value |
| `bb_upper` | float | Upper Bollinger Band |
| `bb_lower` | float | Lower Bollinger Band |
| `bb_width` | float | `(upper - lower) / middle` |
| `bb_pct_b` | float | `(close - lower) / (upper - lower)` |
| `adx_value` | float | Current ADX value |
| `macro_ema` | float | H1 200-EMA value |
| `macro_distance` | float | `h1_close - h1_ema200` |
| `last_pivot_high` | float | Most recent confirmed pivot high price |
| `last_pivot_low` | float | Most recent confirmed pivot low price |
| `bars_since_pivot_high` | int | Bars since last pivot high was confirmed |
| `bars_since_pivot_low` | int | Bars since last pivot low was confirmed |
