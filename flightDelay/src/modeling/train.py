# flightDelay/src/modeling/train.py

import pandas as pd
import joblib
import sys
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
# Optional: LightGBM
try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Info: lightgbm package not found. VotingRegressor will use GBR and RFR only.")

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Added for robust NaN handling in features

# --- Path Setup ---
# Best practice: Try relative import first, then adjust path if run directly
try:
    # Assumes train.py is in src/modeling/ and config.py is in src/data_processing/
    from ..data_processing import config
except ImportError:
    # Handle case where the script might be run directly or paths are different
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir) # Up one level to src/
    project_dir = os.path.dirname(src_dir) # Up another level to flightDelay/ potentially
    # Add src directory to path to find data_processing module
    sys.path.insert(0, src_dir)
    try:
        from data_processing import config
    except ImportError:
        print("CRITICAL: Cannot find 'config.py'.")
        print(f"Attempted to add '{src_dir}' to sys.path.")
        print(f"Current sys.path: {sys.path}")
        sys.exit(1)


def load_training_data():
    """Loads the historical data prepared for training."""
    print(f"Loading training data from: {config.HISTORICAL_POINTS}")
    try:
        df = pd.read_csv(config.HISTORICAL_POINTS, parse_dates=[config.SCHED_DATETIME_COL])
        print(f"Loaded {len(df)} historical points for training.")

        # --- Subsampling (Add this block for memory testing) ---
        sample_fraction = 0.1 # Start with a small fraction, e.g., 10%
        if sample_fraction < 1.0:
            print(f"!!! SUBSAMPLING TRAINING DATA TO {sample_fraction*100:.1f}% FOR MEMORY TESTING !!!")
            df = df.sample(frac=sample_fraction, random_state=42).copy()
            print(f"Using {len(df)} rows after subsampling.")
        # --- End Subsampling ---

        if df.empty:
            print("Error: Training data is empty after loading (or subsampling).")
            sys.exit(1)
        # ... (validation of columns) ...
        return df
    except Exception as e:
        print(f"Error loading or subsampling training data: {e}")
        sys.exit(1)

    if not config.HISTORICAL_POINTS.exists():
        print(f"Error: Training data file not found at {config.HISTORICAL_POINTS}")
        print("Please ensure the data processing steps have run successfully.")
        sys.exit(1)
    try:
        df = pd.read_csv(config.HISTORICAL_POINTS, parse_dates=[config.SCHED_DATETIME_COL])
        print(f"Loaded {len(df)} historical points for training.")
        if df.empty:
            print("Error: Loaded training data is empty.")
            sys.exit(1)
        # Validate necessary columns exist
        required_cols_for_train = config.FDPP_FEATURES + config.CATEGORICAL_MODEL_FEATURES + [config.SCHED_DATETIME_COL, config.FLIGHT_DELAY_COL]
        missing_cols = [col for col in required_cols_for_train if col not in df.columns]
        if missing_cols:
            print(f"Error: Training data is missing required columns: {missing_cols}")
            sys.exit(1)
        return df
    except Exception as e:
        print(f"Error loading or validating training data: {e}")
        sys.exit(1)

def train_model(df):
    """
    Trains the Voting Regressor model based on FDPP-ML features and Table 1,
    using base models (GBR, RFR, optionally LGBM) with default parameters
    as specified in the paper. Includes preprocessing within the pipeline.
    """
    print("Starting model training (Voting Regressor with GBR, RFR [+LGBM])...")

    # --- Feature Engineering / Selection ---
    fdpp_features = config.FDPP_FEATURES
    categorical_features = [f for f in config.CATEGORICAL_MODEL_FEATURES if f in df.columns]

    # Derive temporal features
    try:
        df['Month'] = df[config.SCHED_DATETIME_COL].dt.month
        df['DayOfWeek'] = df[config.SCHED_DATETIME_COL].dt.dayofweek
        df['Hour'] = df[config.SCHED_DATETIME_COL].dt.hour
        temporal_features = config.TEMPORAL_MODEL_FEATURES
    except AttributeError as e:
        print(f"Error extracting temporal features from '{config.SCHED_DATETIME_COL}': {e}")
        print("Ensure this column was parsed correctly as datetime.")
        sys.exit(1)

    numerical_features = fdpp_features # FTD, PFD

    target = config.FLIGHT_DELAY_COL

    all_feature_names = numerical_features + categorical_features + temporal_features
    print(f"Using base features for model: {all_feature_names}")

    X = df[all_feature_names].copy()
    y = df[target].copy()

    # --- Preprocessing Pipeline ---
    # Numerical features: Impute NaNs (median) THEN Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle NaNs first
        ('scaler', StandardScaler())
    ])

    # Categorical features: Impute NaNs (most frequent) THEN One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Handle NaNs first
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            # Apply numeric transformer to numerical + derived temporal features
            ('num', numeric_transformer, numerical_features + temporal_features),
            # Apply categorical transformer to categorical features
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any other columns (if any) - safer than 'drop' initially
    )

    # --- Define Base Models (with DEFAULT parameters) ---
    gbr = GradientBoostingRegressor(random_state=42)
    rfr = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Create Pipelines for each base model (Preprocessor + Estimator)
    # The same preprocessor instance is used in each pipeline
    pipeline_gbr = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', gbr)])
    pipeline_rfr = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rfr)])

    base_estimators_list = [
        ('gbr', pipeline_gbr),
        ('rfr', pipeline_rfr),
    ]

    # Conditionally add LGBM pipeline if available
    if LGBM_AVAILABLE:
        print("Adding LGBMRegressor as a base estimator.")
        lgbm = LGBMRegressor(random_state=42, n_jobs=-1)
        pipeline_lgbm = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', lgbm)])
        base_estimators_list.append(('lgbm', pipeline_lgbm))

    # --- Create Voting Regressor ---
    voting_regressor = VotingRegressor(
        estimators=base_estimators_list,
        n_jobs=1 # Parallelize fitting of base estimators if possible
    )

    # --- Train the Voting Regressor ---
    print(f"Fitting the Voting Regressor with base models: {[name for name, _ in base_estimators_list]}...")
    try:
        voting_regressor.fit(X, y)
        print("Voting Regressor training complete.")
    except Exception as e:
        print(f"ERROR during model fitting: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed error
        sys.exit(1)

    # --- Save the trained Voting Regressor model ---
    model_filename = config.TRAINED_MODEL_FILE
    print(f"Saving trained Voting Regressor model to {model_filename}")
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(voting_regressor, model_filename)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1) # Exit if model cannot be saved

    return voting_regressor

def run_training():
    """Orchestrates loading data and training the model."""
    df_train = load_training_data()
    trained_model = train_model(df_train)
    print("Training pipeline finished.")

if __name__ == "__main__":
    run_training()
