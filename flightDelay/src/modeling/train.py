#!/usr/bin/env python3
"""
flightDelay/src/modeling/train.py

Trains a 5‑class delay classifier on historical flight-path data.
Loads HISTORICAL_POINTS, bins continuous delays into classes,
derives temporal features, splits into train/val/test (70/15/15),
prints precision/recall/F1/support, and saves the VotingClassifier.
"""

import os
import sys
import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Optional LGBM
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# Configuration imports
from src.data_processing.config import (
    HISTORICAL_POINTS,
    SCHED_DATETIME_COL,
    FLIGHT_DELAY_COL,
    FDPP_FEATURES,
    CATEGORICAL_MODEL_FEATURES,
    TEMPORAL_MODEL_FEATURES,
    DELAY_THRESHOLDS,
    NUM_CLASSES,
    MODELS_DIR,
    TRAINED_MODEL_FILE
)


def categorize_delay(delay: float) -> int:
    """
    Map a continuous delay (minutes) into an integer class 0…NUM_CLASSES‑1
    using the configured DELAY_THRESHOLDS.
    """
    for cls, (low, high) in enumerate(zip(DELAY_THRESHOLDS[:-1], DELAY_THRESHOLDS[1:])):
        if low < delay <= high:
            return cls
    return NUM_CLASSES - 1


def load_data() -> pd.DataFrame:
    """
    Loads the historical points CSV, applies delay binning,
    and derives Month/DayOfWeek/Hour features.
    """
    if not os.path.exists(HISTORICAL_POINTS):
        print(f"Error: Historical data not found at {HISTORICAL_POINTS}")
        sys.exit(1)

    df = pd.read_csv(HISTORICAL_POINTS, parse_dates=[SCHED_DATETIME_COL])
    # Bin into classes
    df['delay_category'] = df[FLIGHT_DELAY_COL].apply(categorize_delay)
    # Temporal features
    df['Month']     = df[SCHED_DATETIME_COL].dt.month
    df['DayOfWeek'] = df[SCHED_DATETIME_COL].dt.dayofweek
    df['Hour']      = df[SCHED_DATETIME_COL].dt.hour

    return df


def build_pipeline() -> VotingClassifier:
    """
    Constructs a soft‑voting ensemble of GBT, RFC, (optional LGBM),
    each wrapped in a preprocessing pipeline.
    """
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',   StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, FDPP_FEATURES + TEMPORAL_MODEL_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_MODEL_FEATURES),
    ], remainder='drop')

    estimators = [
        ('gbt', Pipeline([('pre', preprocessor),
                          ('clf', GradientBoostingClassifier(random_state=42))])),
        ('rfc', Pipeline([('pre', preprocessor),
                          ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))]))
    ]
    if LGBM_AVAILABLE:
        estimators.append((
            'lgbm',
            Pipeline([('pre', preprocessor),
                      ('clf', LGBMClassifier(random_state=42, n_jobs=-1))])
        ))

    return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)


def run_training():
    """
    Full training pipeline:
     1. Load and prepare data
     2. Split into train/val/test (70/15/15)
     3. Fit voting classifier on train
     4. Print classification reports on val & test
     5. Save the model
    """
    # 1) Load
    df = load_data()
    n = len(df)

    # 2) Split indices
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    idx_train = perm[:train_end]
    idx_val   = perm[train_end:val_end]
    idx_test  = perm[val_end:]

    splits = [
        ('Train',      df.iloc[idx_train]),
        ('Validation', df.iloc[idx_val]),
        ('Test',       df.iloc[idx_test])
    ]

    feature_cols = FDPP_FEATURES + CATEGORICAL_MODEL_FEATURES + TEMPORAL_MODEL_FEATURES

    # 3) Train
    X_train = splits[0][1][feature_cols]
    y_train = splits[0][1]['delay_category']
    clf = build_pipeline()
    print("Fitting VotingClassifier on Train split…")
    clf.fit(X_train, y_train)

    # 4) Evaluate
    for name, split_df in splits[1:]:
        print(f"\n=== {name} Classification Report ===")
        X_split = split_df[feature_cols]
        y_split = split_df['delay_category']
        preds   = clf.predict(X_split)
        print(classification_report(y_split, preds, digits=4))

    # 5) Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, TRAINED_MODEL_FILE)
    print(f"\nModel saved to {TRAINED_MODEL_FILE}")


if __name__ == "__main__":
    run_training()
