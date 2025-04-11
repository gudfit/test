# src/1_dataPreprocessing/weatherPredictionProcessing/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import importlib
import os
import sys

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_notebook_style(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """Applies feature engineering and notebook-style preprocessing."""
    logging.info("Starting notebook-style preprocessing with feature engineering...")
    target = config.TARGET_VARIABLE

    # --- 1. Feature Engineering ---
    try:
        # Import the feature_engineering module directly from its implementation path
        # to avoid circular imports
        fe_path = os.path.join(os.path.dirname(__file__), "feature_engineering.py")
        spec = importlib.util.spec_from_file_location("feature_engineering", fe_path)
        feature_engineering = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_engineering)
        
        # Now use the feature_engineering module
        df_engineered = feature_engineering.engineer_features(df.copy())
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}", exc_info=True)
        return pd.DataFrame(), pd.Series(), []

    # --- 2. Select Features and Target ---
    # Use the MODEL_FEATURES list *updated* by engineer_features
    # Also ensure base categoricals and target are present for subsequent steps
    features_to_use_in_model = [f for f in config.MODEL_FEATURES if f in df_engineered.columns]
    base_cats_needed = [f for f in config.CATEGORICAL_FEATURES_BASE if f in df_engineered.columns]
    cols_to_keep = list(set(features_to_use_in_model + [target] + base_cats_needed))

    missing_features = set(config.MODEL_FEATURES) - set(features_to_use_in_model)
    if missing_features:
        logging.warning(f"Engineered features configured but not found in DataFrame after engineering: {missing_features}")

    if target not in df_engineered.columns:
        logging.error(f"Target variable '{target}' not found after engineering.")
        return pd.DataFrame(), pd.Series(), []
    if not features_to_use_in_model:
         logging.error("No usable features found after engineering.")
         return pd.DataFrame(), pd.Series(), []

    # Select only necessary columns
    df_model = df_engineered[cols_to_keep].copy()
    logging.debug(f"Shape after initial column selection: {df_model.shape}")

    # --- 3. Handle Target ---
    df_model[target] = pd.to_numeric(df_model[target], errors='coerce').fillna(0)
    df_model[target] = df_model[target].clip(lower=0)

    # --- 4. Drop rows with NaNs in the *selected feature* columns ---
    # Use the list determined from config.MODEL_FEATURES, excluding the target itself
    features_to_check_na = [f for f in features_to_use_in_model if f in df_model.columns]
    if not features_to_check_na:
        logging.error("No features available to check for NaNs.")
        return pd.DataFrame(), pd.Series(), []

    rows_before = len(df_model)
    df_model = df_model.dropna(subset=features_to_check_na)
    rows_after = len(df_model)
    if rows_before > rows_after:
         logging.warning(f"Dropped {rows_before - rows_after} rows due to NaNs in selected feature set.")

    if df_model.empty:
        logging.warning("DataFrame is empty after dropping NaNs.")
        return pd.DataFrame(), pd.Series(), []
    logging.debug(f"Shape after dropping NaNs: {df_model.shape}")


    # --- 5. One-hot encode BASE categoricals ---
    categorical_to_encode = [f for f in config.CATEGORICAL_FEATURES_BASE if f in df_model.columns]
    logging.info(f"Applying one-hot encoding to base categoricals: {categorical_to_encode}")
    if categorical_to_encode: # Only run if there are categoricals to encode
        try:
            df_model = pd.get_dummies(df_model, columns=categorical_to_encode, drop_first=True, dummy_na=False)
            logging.debug(f"Shape after OHE: {df_model.shape}")
        except Exception as e:
            logging.error(f"Error during one-hot encoding: {e}", exc_info=True)
            return pd.DataFrame(), pd.Series(), []
    else:
        logging.warning("No base categorical features found to encode.")


    # --- 6. FINAL SELECTION: Ensure only numeric features remain in X ---
    if target not in df_model.columns:
        logging.error("Target column lost during preprocessing.")
        return pd.DataFrame(), pd.Series(), []

    X_potential = df_model.drop(target, axis=1)
    y = df_model[target]

    # Select only columns with numeric dtypes (int or float)
    numeric_cols = X_potential.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols = X_potential.select_dtypes(exclude=np.number).columns.tolist()

    if non_numeric_cols:
        logging.warning(f"Non-numeric columns found in feature set just before final selection, will be DROPPED: {non_numeric_cols}")

    # Keep only numeric columns for the final feature set X
    X = X_potential[numeric_cols]
    final_feature_columns = numeric_cols # This is the definitive list of features used

    if X.empty or len(final_feature_columns) == 0:
        logging.error("Feature set X is empty after selecting only numeric columns.")
        return pd.DataFrame(), pd.Series(), []

    logging.info(f"Preprocessing complete. Final Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, final_feature_columns
