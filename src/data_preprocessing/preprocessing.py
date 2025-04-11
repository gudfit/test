# src/data_preprocessing/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Keep sklearn imports if using the pipeline approach elsewhere
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
import logging

from src import config
# Import feature engineering module
from src.data_preprocessing import feature_engineering # Ensure this module exists and has engineer_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Keep other preprocessing functions if desired for alternative pipelines ---
# Example placeholder:
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Placeholder for other cleaning logic if needed """
    logging.warning("Standard clean_data function not fully implemented for this pipeline.")
    # Add any generic cleaning steps needed before feature engineering if required
    return df

def create_preprocessing_pipeline(numerical_features: list, categorical_features: list):
    """ Placeholder for sklearn pipeline creation if needed """
    logging.warning("Standard create_preprocessing_pipeline function not implemented for this pipeline.")
    return None

def split_data(df: pd.DataFrame):
     """ Placeholder for standard splitting if needed """
     logging.warning("Standard split_data function not implemented for this pipeline.")
     # Implement standard splitting based on config if required elsewhere
     X=df[config.MODEL_FEATURES] # This line would need config.MODEL_FEATURES to be relevant
     y=df[config.TARGET_VARIABLE]
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE)
     return X_train, X_test, y_train, y_test


# --- Notebook Style Preprocessing Function (Corrected) ---
def preprocess_notebook_style(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """Applies feature engineering and notebook-style preprocessing."""
    logging.info("Starting notebook-style preprocessing with feature engineering...")
    target = config.TARGET_VARIABLE

    # --- 1. Feature Engineering ---
    try:
        # This call assumes feature_engineering.py has the engineer_features function
        # which also updates config.MODEL_FEATURES dynamically
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
