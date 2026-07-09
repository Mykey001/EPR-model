# EPR-ML Implementation Plan — Mean Reversion Sniper ML Enhancement

## Goal

Build a machine learning system that:
1. **Replicates** the EPR Mean Reversion Sniper signal logic in Python (ground truth)
2. **Predicts** entry signals with higher accuracy (fewer false signals, better timing)
3. **Manages trades dynamically** (adaptive SL/TP, early exits, position sizing based on signal confidence)

---

## What I Need From You (Data Requirements)

> [!IMPORTANT]
> **This is the critical first step.** I need you to export raw OHLCV data from MetaTrader 5 and place the CSV files in `EPR-ML/data/raw/`. Without this data, nothing else can proceed.

### Required Data Export #1: Primary Timeframe (M5)

| Field | Requirement |
|-------|-------------|
| **Symbol** | XAUUSD (Gold) — or whichever symbol you run this EA on |
| **Timeframe** | M5 (5-minute) |
| **History** | **Minimum 2 years** (Jan 2023 → present). 3+ years is better for walk-forward validation |
| **Columns** | `Date, Time, Open, High, Low, Close, Tick Volume` |
| **Format** | CSV file |
| **Filename** | `XAUUSD_M5.csv` |

### Required Data Export #2: Macro Timeframe (H1)

| Field | Requirement |
|-------|-------------|
| **Symbol** | Same symbol as above |
| **Timeframe** | H1 (1-hour) |
| **History** | Same date range as M5 |
| **Columns** | `Date, Time, Open, High, Low, Close, Tick Volume` |
| **Format** | CSV file |
| **Filename** | `XAUUSD_H1.csv` |

### How to Export from MT5

1. Open MetaTrader 5
2. Go to **File → Open Data Folder** (note the path)
3. Or: Open the **Symbols** window (Ctrl+U), find your symbol
4. Use the **History Center** or a script — I can provide an MQL5 export script if needed
5. Alternatively: **Tools → History Center → Select symbol → Select timeframe → Export**

### Optional But Valuable

| Data | Why It Helps |
|------|-------------|
| **MT5 Strategy Tester report** (HTML or CSV) from backtesting the EA on the same data range | Gives us exact trade entries/exits to validate our Python label generation matches the EA |
| **M15 data** for the same range | Allows multi-timeframe feature engineering |
| **D1 data** for the same range | Additional macro context features |

---

## User Review Required

> [!IMPORTANT]
> **Which symbol do you run this EA on?** The data export must match the symbol. If you run it on multiple symbols, we start with one and expand later.

> [!IMPORTANT]
> **Which timeframe do you run the EA on?** The EA uses `PERIOD_CURRENT` which means it adapts to whatever chart you attach it to. I'm assuming **M5** based on your other projects — please confirm.

> [!WARNING]
> **Do you have TA-Lib installed?** TA-Lib requires a C library and can be tricky on Windows. If not installed, I'll use a pure-Python fallback (pandas-ta) — same results, slightly slower. Let me know your current Python environment setup.

---

## Phase 1: Data Preparation & Indicator Computation

**Notebook:** `01_data_preparation.ipynb`

### What happens:
1. Load raw OHLCV CSVs (M5 + H1)
2. Clean & validate data (check for gaps, weekends, duplicates)
3. Compute **all 7 indicators** exactly matching the EA's parameters:
   - EMA(5), EMA(15) on M5
   - Parabolic SAR (0.02 / 0.02 / 0.20) on M5
   - RSI(7) on M5
   - Bollinger Bands (20, 2.0) on M5
   - ADX(14) on M5
   - EMA(200) on H1, mapped back to M5 bars
4. Compute pivot highs/lows with the exact 10L/10R algorithm
5. Save processed dataset to `data/processed/`

### Validation checkpoint:
- Spot-check 10+ random bars against MT5 chart values
- Indicator values must match MT5 within floating-point tolerance (< 0.01%)

---

## Phase 2: Label Generation (Ground Truth)

**Notebook:** `02_label_generation.ipynb`

### What happens:
1. Replicate the **exact** EA signal logic in Python:
   - EMA crossover detection
   - PSAR direction check
   - RSI/BB extreme lookback scan
   - ADX regime filter
   - Macro EMA filter
2. Generate binary labels: `buy_signal`, `sell_signal` for every bar
3. For each signal, simulate the trade forward:
   - Compute SL (swing point or fixed fallback)
   - Compute TP (SL distance × RR ratio)
   - Walk forward bar-by-bar to determine: **WIN** (hit TP), **LOSS** (hit SL), or **TIMEOUT**
4. Record trade metadata: entry price, SL, TP, duration, max favorable excursion (MFE), max adverse excursion (MAE)
5. Save labeled dataset to `data/labels/`

### Validation checkpoint:
- If MT5 backtest report is available: compare our Python signals against the EA's actual trades
- Signal count, win rate, and average trade duration should closely match

---

## Phase 3: Feature Engineering

**Notebook:** `03_feature_engineering.ipynb`

### Feature Categories:

#### A. Strategy-Native Features (from the EA's own indicators)
| Feature | Description |
|---------|-------------|
| `ema_diff` | Fast EMA − Slow EMA |
| `ema_diff_pct` | EMA difference as % of price |
| `ema_slope_fast` | Rate of change of fast EMA |
| `ema_slope_slow` | Rate of change of slow EMA |
| `psar_distance` | Close − SAR (positive = bullish) |
| `psar_distance_pct` | SAR distance as % of price |
| `rsi_value` | Raw RSI(7) |
| `rsi_divergence` | Price making new low but RSI not (and vice versa) |
| `bb_pct_b` | %B = (Close − Lower) / (Upper − Lower) |
| `bb_width` | Band width normalized |
| `adx_value` | Raw ADX(14) |
| `plus_di` | +DI value |
| `minus_di` | −DI value |
| `di_diff` | +DI − (−DI) |

#### B. Structural Features (from pivot points)
| Feature | Description |
|---------|-------------|
| `dist_to_last_pivot_high` | Price distance to last swing high |
| `dist_to_last_pivot_low` | Price distance to last swing low |
| `bars_since_pivot_high` | Bars elapsed since last pivot high |
| `bars_since_pivot_low` | Bars elapsed since last pivot low |
| `swing_range` | Last pivot high − last pivot low |
| `price_in_swing_pct` | Where price sits within the swing range |

#### C. Market Microstructure Features
| Feature | Description |
|---------|-------------|
| `atr_14` | Average True Range (volatility) |
| `atr_ratio` | Current ATR / rolling mean ATR (vol expansion/contraction) |
| `body_size` | |Close − Open| / ATR |
| `upper_wick_ratio` | Upper wick / total range |
| `lower_wick_ratio` | Lower wick / total range |
| `volume_ratio` | Current volume / rolling avg volume |

#### D. Multi-Timeframe Features
| Feature | Description |
|---------|-------------|
| `h1_trend_strength` | H1 close distance from H1 EMA200, normalized |
| `h1_rsi` | RSI on H1 timeframe |
| `h1_atr` | ATR on H1 timeframe |
| `h1_candle_direction` | H1 candle body direction |

#### E. Temporal Features
| Feature | Description |
|---------|-------------|
| `hour_of_day` | 0–23, cyclically encoded (sin/cos) |
| `day_of_week` | 0–4, cyclically encoded |
| `session` | London / New York / Asian / overlap (one-hot) |

#### F. Condition Proximity Features (ML Enhancement)
| Feature | Description |
|---------|-------------|
| `conditions_met_buy` | Count of how many of the 5 buy conditions are currently true (0–5) |
| `conditions_met_sell` | Count of how many of the 5 sell conditions are currently true (0–5) |
| `bars_since_last_cross` | How many bars since the last EMA cross |
| `bars_since_extreme` | How many bars since last RSI/BB extreme |

### Feature Selection:
- Remove highly correlated features (Spearman > 0.95)
- Use permutation importance from initial XGBoost model
- Target: ~30–50 final features

---

## Phase 4: Model Training

**Notebook:** `04_model_training.ipynb`

### Model Architecture — Three Models, Three Jobs:

```
┌──────────────────────────────────────────────────────────────┐
│                    MODEL 1: Signal Classifier                │
│  "Should we enter a trade right now?"                        │
│  Input: All features at bar t                                │
│  Output: P(buy), P(sell), P(no_trade)                        │
│  Type: XGBoost / LightGBM multi-class classifier            │
│  Training: On all bars, with class weights for imbalance     │
└──────────────────────────────────────────────────────────────┘
                           │
                    (if signal fires)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                MODEL 2: Trade Quality Regressor              │
│  "How good is this particular trade setup?"                  │
│  Input: Features at signal bar + signal metadata             │
│  Output: P(win), Expected R-multiple, Optimal RR ratio       │
│  Type: XGBoost regressor / classifier                        │
│  Training: Only on signal bars, target = trade outcome        │
└──────────────────────────────────────────────────────────────┘
                           │
                    (during trade)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              MODEL 3: Dynamic Exit Manager                   │
│  "Should we exit/adjust this trade now?"                     │
│  Input: Features at bar t + trade state (unrealized PnL,     │
│         bars in trade, current SL/TP distance)               │
│  Output: P(exit_now), Suggested SL adjustment                │
│  Type: XGBoost classifier + rule-based trailing logic        │
│  Training: On in-trade bars, target = optimal exit timing     │
└──────────────────────────────────────────────────────────────┘
```

### Training Strategy:
1. **Walk-Forward Validation** (not random train/test split — time series data!)
   - 5 folds, each with expanding training window
   - No future data leakage
2. **Class Imbalance Handling** (signals are rare ~1% of bars)
   - SMOTE oversampling on training set only
   - Class weights in XGBoost
   - Focal loss consideration
3. **Hyperparameter Tuning**
   - Optuna Bayesian optimization
   - Cross-validated on walk-forward windows
4. **Threshold Calibration**
   - Precision-recall tradeoff analysis
   - Select threshold that maximizes profit factor, not just accuracy

### Key Metrics (NOT just accuracy):
| Metric | Why It Matters |
|--------|---------------|
| **Precision** | % of predicted signals that are actually profitable |
| **Recall** | % of profitable setups that the model catches |
| **Profit Factor** | Gross profit / gross loss (must be > 1.5) |
| **Sharpe Ratio** | Risk-adjusted returns |
| **Max Drawdown** | Worst peak-to-trough equity decline |
| **Win Rate** | % of trades that hit TP |
| **Average R-multiple** | Average return per unit of risk |

---

## Phase 5: Backtesting & Validation

**Notebook:** `05_backtesting.ipynb`

### What happens:
1. **Walk-forward backtest** across out-of-sample periods
2. Apply realistic trading conditions:
   - Spread simulation (average spread for the symbol)
   - Slippage modeling
   - No look-ahead bias verification
3. Compare three strategies head-to-head:
   - **Baseline:** Original EA logic (Python replica)
   - **ML Signal:** ML Model 1 replacing the EA's signal logic
   - **Full ML:** ML Models 1+2+3 (signal + quality filter + dynamic exits)
4. Generate performance report:
   - Equity curve
   - Drawdown chart
   - Monthly returns heatmap
   - Trade distribution analysis
   - Feature importance visualization

### Success Criteria:
| Metric | Baseline Target | ML Target |
|--------|----------------|-----------|
| Profit Factor | > 1.0 | > 1.5 |
| Win Rate | Match EA | > 55% |
| Max Drawdown | Match EA | < EA's DD |
| Sharpe Ratio | Baseline | > Baseline × 1.3 |
| Signal Quality | 100% EA match | Higher precision |

---

## Phase 6: Trade Management Model

**Notebook:** `06_trade_management.ipynb`

### Dynamic Exit Strategies (the ML enhancements the EA lacks):
1. **Trailing Stop** — ML-driven, adapts to volatility regime
2. **Breakeven Move** — Move SL to entry after trade reaches 1R profit
3. **Partial Close** — Take 50% at 1R, let rest run to 2R+ with trailing SL
4. **Early Exit** — Close trade if indicator conditions deteriorate (e.g., ADX spikes above 40 mid-trade)
5. **Dynamic Position Sizing** — Scale lot size based on Model 2's confidence score

---

## Execution Order

| Step | Phase | Blocked By | Estimated Effort |
|------|-------|-----------|-----------------|
| 1 | **Data Export** (YOU) | Nothing | 30 min |
| 2 | Phase 1: Data Prep | Step 1 | 1 notebook session |
| 3 | Phase 2: Label Gen | Step 2 | 1 notebook session |
| 4 | Phase 3: Features | Step 3 | 1 notebook session |
| 5 | Phase 4: Training | Step 4 | 1–2 notebook sessions |
| 6 | Phase 5: Backtest | Step 5 | 1 notebook session |
| 7 | Phase 6: Trade Mgmt | Step 6 | 1 notebook session |

---

## Project Folder Structure (Completed ✅)

```
EPR-ML/
├── README.md                     ✅
├── requirements.txt              ✅
├── configs/
│   └── default_config.yaml       ✅
├── docs/
│   ├── strategy_theory.md        ✅
│   ├── indicator_reference.md    ✅
│   └── signal_logic.md           ✅
├── data/
│   ├── raw/                      ✅ (awaiting your CSV exports)
│   ├── processed/                ✅
│   └── labels/                   ✅
├── notebooks/                    ✅ (to be created per phase)
├── src/                          ✅
├── models/                       ✅
├── results/                      ✅
└── original_ea/                  ✅
```

---

## Open Questions

> [!IMPORTANT]
> 1. **Which symbol?** XAUUSD or another? This determines data export.
> 2. **Which primary timeframe?** M5, M15, or other? I'm assuming M5.
> 3. **Do you have a Strategy Tester backtest report** for this EA? If so, export it — invaluable for validation.
> 4. **Python environment** — Do you have a working Python setup with Jupyter? Do you have TA-Lib installed? If not, I'll handle alternatives.
> 5. **Do you want the final model deployed back to MQL5** as a new EA, or kept as a Python-only analysis/signal tool?
