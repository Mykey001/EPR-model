# EPR-ML: Mean Reversion Sniper — Machine Learning Enhancement

> ML-powered prediction system for the Combined HL-EPR Mean Reversion Sniper EA v2.0

## Project Overview

This project builds a machine learning pipeline that learns the signal logic, risk management, and trade management of the EPR Mean Reversion Sniper strategy, then enhances it with:

- **Predictive entry signals** — anticipate setups before all 5 conditions align
- **Dynamic exit management** — ML-driven trailing stops, partial closes, and early exits
- **Adaptive risk sizing** — position sizing based on predicted signal quality

## Project Structure

```
EPR-ML/
├── README.md                          # This file
├── docs/                              # Strategy documentation & theory
│   ├── strategy_theory.md             # Complete strategy analysis
│   ├── indicator_reference.md         # Indicator calculations & parameters
│   └── signal_logic.md               # Signal generation rules
├── data/
│   ├── raw/                           # Raw OHLCV data exports
│   ├── processed/                     # Feature-engineered datasets
│   └── labels/                        # Generated trade labels (entry/exit/SL/TP)
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Data loading, cleaning, indicator computation
│   ├── 02_label_generation.ipynb      # Replicate EA logic to generate ground-truth labels
│   ├── 03_feature_engineering.ipynb   # Feature creation & selection
│   ├── 04_model_training.ipynb        # Model training & hyperparameter tuning
│   ├── 05_backtesting.ipynb           # Walk-forward validation & performance metrics
│   └── 06_trade_management.ipynb      # Dynamic exit & risk management model
├── src/
│   ├── __init__.py
│   ├── indicators.py                  # Indicator computation functions
│   ├── labeler.py                     # EA logic replication for label generation
│   ├── features.py                    # Feature engineering pipeline
│   ├── models.py                      # Model definitions
│   ├── backtester.py                  # Backtesting engine
│   └── utils.py                       # Shared utilities
├── models/                            # Saved trained models
│   └── .gitkeep
├── results/                           # Backtest results, charts, reports
│   └── .gitkeep
├── configs/                           # Configuration files
│   └── default_config.yaml            # Default parameters matching EA defaults
├── original_ea/                       # Original EA source for reference
│   └── CombinedHL_EPR_SwingSL_EA_v2.mq5
└── requirements.txt                   # Python dependencies
```

## Status

| Phase | Status |
|-------|--------|
| Project Setup | ✅ Complete |
| Documentation | ✅ Complete |
| Data Collection | ⏳ Pending |
| Feature Engineering | ⏳ Pending |
| Model Training | ⏳ Pending |
| Backtesting | ⏳ Pending |
| Trade Management | ⏳ Pending |
# EPR-model
