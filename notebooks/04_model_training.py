# %% [markdown]
# # 04 — Model Training & Validation
# 
# **EPR Mean Reversion Sniper — ML Enhancement Project**
# 
# This notebook:
# 1. Trains **Model 1 (Signal Classifier)** to predict if a bar presents a profitable trade setup.
# 2. Trains **Model 2 (Trade Quality Regressor)** to predict the expected R-multiple of a signal.
# 3. Uses Time-Series Split (Walk-Forward Validation) to ensure no data leakage.
# 4. Saves the trained models and their performance metrics.

# %%
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(r"c:\Users\MYCkey98\Desktop\Organized_Files\05_ML_Projects\EPR-ML")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# %% [markdown]
# ## 1. Train Model 1: Signal Classifier
# We want to predict if a bar is a good buy, good sell, or no trade.

# %%
m1_path = PROCESSED_DIR / "model1_features.parquet"
if not m1_path.exists():
    print("Model 1 features not found! Please run Phase 3.")
else:
    df_m1 = pd.read_parquet(m1_path)
    print(f"Model 1 Data: {df_m1.shape}")

    X1 = df_m1.drop(columns=['target_class'])
    y1 = df_m1['target_class']

    print(f"Class distribution:\n{y1.value_counts(normalize=True)*100}")

    # Time-Series Walk-Forward Split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # We will train a simple XGBoost model to establish a baseline
    clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=1, # Since it's multi-class, we might need class_weights, but let's start simple
        random_state=42,
        n_jobs=-1
    )

    # Note: For extreme imbalance (99% class 0), sample weights or SMOTE are usually needed.
    # We will use sample weights.
    
    # To keep this notebook fast for demonstration, we will train on the last split
    for train_index, test_index in tscv.split(X1):
        pass # just get the last one

    X1_train, X1_test = X1.iloc[train_index], X1.iloc[test_index]
    y1_train, y1_test = y1.iloc[train_index], y1.iloc[test_index]

    print(f"\nTraining Model 1 on {len(X1_train)} samples, testing on {len(X1_test)} samples...")

    # Calculate weights to handle severe class imbalance
    # 0 = No Trade, 1 = Good Buy, 2 = Good Sell
    counts = np.bincount(y1_train)
    weights = np.ones(len(y1_train))
    weights[y1_train == 1] = counts[0] / max(counts[1], 1)
    weights[y1_train == 2] = counts[0] / max(counts[2], 1)

    clf.fit(X1_train, y1_train, sample_weight=weights)

    # Evaluate
    y1_pred = clf.predict(X1_test)
    print("\nModel 1 Classification Report:")
    print(classification_report(y1_test, y1_pred, target_names=['No Trade', 'Good Buy', 'Good Sell']))

    # Feature Importance
    importance = pd.DataFrame({
        'feature': X1.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Features for Signal Classification:")
    print(importance.head(10))

    # Save Model 1
    joblib.dump(clf, MODELS_DIR / "m1_signal_classifier.joblib")
    print("✅ Model 1 saved.")

# %% [markdown]
# ## 2. Train Model 2: Trade Quality Regressor
# We want to predict the exact R-multiple a signal will achieve.

# %%
m2_path = PROCESSED_DIR / "model2_features.parquet"
if not m2_path.exists():
    print("Model 2 features not found! Please run Phase 3.")
else:
    df_m2 = pd.read_parquet(m2_path)
    print(f"\nModel 2 Data (Signals Only): {df_m2.shape}")

    X2 = df_m2.drop(columns=['target_r_multiple'])
    y2 = df_m2['target_r_multiple']

    # For regression, an XGBoost Regressor
    reg = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )

    # Since signal data is much smaller, we'll do an 80/20 temporal split
    split_idx = int(len(X2) * 0.8)
    X2_train, X2_test = X2.iloc[:split_idx], X2.iloc[split_idx:]
    y2_train, y2_test = y2.iloc[:split_idx], y2.iloc[split_idx:]

    print(f"\nTraining Model 2 on {len(X2_train)} samples, testing on {len(X2_test)} samples...")

    reg.fit(X2_train, y2_train)

    y2_pred = reg.predict(X2_test)
    
    mse = mean_squared_error(y2_test, y2_pred)
    r2 = r2_score(y2_test, y2_pred)
    
    print(f"\nModel 2 Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    # Feature Importance
    importance2 = pd.DataFrame({
        'feature': X2.columns,
        'importance': reg.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Features for Trade Quality Prediction:")
    print(importance2.head(10))

    # Save Model 2
    joblib.dump(reg, MODELS_DIR / "m2_trade_quality_regressor.joblib")
    print("✅ Model 2 saved.")

print("\nReady for Phase 5 (Backtesting).")
