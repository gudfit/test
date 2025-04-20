import os
import sys
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm

from src.data_processing.config import (
    FUTURE_PREPARED,
    FINAL_PREDICTIONS,
    TRAINED_MODEL_FILE,
    SCHED_DATETIME_COL,
    TAIL_NUM_COL,
    PFD_COL,
    FDPP_FEATURES,
    CATEGORICAL_MODEL_FEATURES,
    TEMPORAL_MODEL_FEATURES
)

# Column name for the predicted class
PREDICTION_COL = 'Predicted_Delay_Class'


def load_model(model_path: str = None):
    """
    Load the trained VotingClassifier from disk.
    """
    path = model_path or TRAINED_MODEL_FILE
    if not os.path.exists(path):
        print(f"Error: model file not found at {path}")
        sys.exit(1)
    model = joblib.load(path)
    print(f"Loaded model from {path}")
    return model


def load_future_data():
    """
    Load the prepared future dataset, parse datetimes, and derive temporal features.
    """
    if not os.path.exists(FUTURE_PREPARED):
        print(f"Error: future data file not found at {FUTURE_PREPARED}")
        sys.exit(1)

    df = pd.read_csv(FUTURE_PREPARED, parse_dates=[SCHED_DATETIME_COL])
    if df.empty:
        print("Warning: future dataset is empty.")
        return df

    # Derive temporal features matching training
    df['Month']     = df[SCHED_DATETIME_COL].dt.month
    df['DayOfWeek'] = df[SCHED_DATETIME_COL].dt.dayofweek
    df['Hour']      = df[SCHED_DATETIME_COL].dt.hour

    # Ensure PFD exists
    if PFD_COL not in df.columns:
        df[PFD_COL] = 0.0

    return df
def iterative_prediction(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    For each tail number group, predict the delay class using the features:
      FDPP_FEATURES + CATEGORICAL_MODEL_FEATURES + TEMPORAL_MODEL_FEATURES.
    Always uses the original PFD from the dataset.
    """
    if df.empty:
        return df
    feature_cols = FDPP_FEATURES + CATEGORICAL_MODEL_FEATURES + TEMPORAL_MODEL_FEATURES
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing feature columns for prediction: {missing}")
        sys.exit(1)
    predictions = {}
    # Group by tail number
    for tail, group in tqdm(df.groupby(TAIL_NUM_COL), desc="Predicting paths"):
        # Iterate in chronological order
        for idx in group.sort_values(SCHED_DATETIME_COL).index:
            row = df.loc[[idx], feature_cols].copy()
            # Always use the original precomputed PFD value
            row[PFD_COL] = df.at[idx, PFD_COL]
            # Predict class label
            pred_class = int(model.predict(row)[0])
            predictions[idx] = pred_class
    # Assign predictions back to the DataFrame
    df[PREDICTION_COL] = pd.Series(predictions)
    return df


def save_predictions(df: pd.DataFrame):
    """Save the DataFrame with predictions to FINAL_PREDICTIONS CSV"""
    out_dir = os.path.dirname(FINAL_PREDICTIONS)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(FINAL_PREDICTIONS, index=False)
    print(f"Saved {len(df)} predictions to {FINAL_PREDICTIONS}")


def run_prediction():
    """Full prediction pipeline: load model, load data, predict, save"""
    print("Loading classifier…")
    model = load_model()
    print("Loading future dataset…")
    df_future = load_future_data()
    print("Running prediction…")
    df_pred = iterative_prediction(df_future, model)
    print("Saving prediction results…")
    save_predictions(df_pred)
    print("Prediction pipeline completed successfully.")


if __name__ == "__main__":
    run_prediction()
