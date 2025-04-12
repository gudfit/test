# flightDelay/src/modeling/predict.py

import pandas as pd
import numpy as np
import joblib
import sys
import os
from tqdm import tqdm

# --- Path Setup ---
try:
    from ..data_processing import config
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    sys.path.insert(0, src_dir)
    try:
        from data_processing import config
    except ImportError:
        print("CRITICAL: Cannot find 'config.py'.")
        print(f"Attempted to add '{src_dir}' to sys.path.")
        print(f"Current sys.path: {sys.path}")
        sys.exit(1)

def load_model(model_path=config.TRAINED_MODEL_FILE):
    """Loads the pre-trained Voting Regressor model pipeline."""
    print(f"Loading trained model from: {model_path}")
    if not model_path.exists():
        print(f"Error: Trained model file not found at {model_path}")
        print("Please ensure the training step has run successfully.")
        sys.exit(1)
    try:
        model = joblib.load(model_path)
        print("Voting Regressor model loaded successfully.")
        # Optional: Print model details or steps if needed for verification
        # print(model)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def load_future_data():
    """Loads the prepared future data set (includes last historical points)."""
    print(f"Loading future data from: {config.FUTURE_PREPARED}")
    if not config.FUTURE_PREPARED.exists():
        print(f"Error: Future data file not found at {config.FUTURE_PREPARED}")
        print("Please ensure the data processing steps (incl. partition_prepare) have run.")
        sys.exit(1)
    try:
        df = pd.read_csv(config.FUTURE_PREPARED, parse_dates=[config.SCHED_DATETIME_COL])
        # Ensure sorted for correct path processing
        df = df.sort_values(by=[config.TAIL_NUM_COL, config.SCHED_DATETIME_COL]).reset_index(drop=True)
        print(f"Loaded {len(df)} future points for prediction.")
        if df.empty:
            print("Warning: Loaded future data is empty.")
            # Allow continuing, prediction loop will just do nothing
        # Initialize prediction column if it doesn't exist
        if config.PREDICTED_DELAY_COL not in df.columns:
            df[config.PREDICTED_DELAY_COL] = np.nan
        return df
    except Exception as e:
        print(f"Error loading future data: {e}")
        sys.exit(1)

def iterative_prediction(df, model):
    """
    Performs iterative prediction using the loaded Voting Regressor pipeline.
    Updates PFD based on previous predictions within the same flight path.
    """
    print("Starting iterative prediction...")
    if df.empty:
        print("Future data is empty. No predictions to make.")
        return df

    # --- Define Features expected by the model's preprocessor ---
    # These are the raw features BEFORE preprocessing
    fdpp_features = config.FDPP_FEATURES
    categorical_features = config.CATEGORICAL_MODEL_FEATURES
    temporal_features = config.TEMPORAL_MODEL_FEATURES
    numerical_features = fdpp_features

    all_feature_names = numerical_features + categorical_features + temporal_features

    # --- Prepare features needed for prediction ---
    print("Preparing features in future data...")
    # Derive temporal features
    try:
        if 'Month' in temporal_features: df['Month'] = df[config.SCHED_DATETIME_COL].dt.month
        if 'DayOfWeek' in temporal_features: df['DayOfWeek'] = df[config.SCHED_DATETIME_COL].dt.dayofweek
        if 'Hour' in temporal_features: df['Hour'] = df[config.SCHED_DATETIME_COL].dt.hour
    except AttributeError as e:
        print(f"Error extracting temporal features from '{config.SCHED_DATETIME_COL}' in future data: {e}")
        sys.exit(1)

    # Check if all required raw feature columns exist
    missing_features = [f for f in all_feature_names if f not in df.columns]
    if missing_features:
         print(f"Error: Features required for prediction are missing from future data: {missing_features}")
         sys.exit(1)

    # --- Iteration Logic ---
    # Group by Tail Number to process paths sequentially
    grouped = df.groupby(config.TAIL_NUM_COL)
    # Use a dictionary to store predictions efficiently, then assign back
    predictions_dict = {}

    print(f"Iterating through {len(grouped)} flight paths (Tail Numbers)...")
    for tail_num, path_df in tqdm(grouped, desc="Predicting Paths"):
        last_predicted_delay = None # Reset for each new path
        # Iterate through each point (flight leg) in the path by index
        for index in path_df.index:
            row_data = path_df.loc[[index], all_feature_names].copy() # Get features as a DataFrame row
            current_pfd = path_df.loc[index, config.PFD_COL] # Get initial PFD for this point

            # Use the previously predicted delay as PFD if available for this path
            if last_predicted_delay is not None:
                current_pfd = last_predicted_delay

            # Update the PFD value in the feature row DataFrame before prediction
            row_data[config.PFD_COL] = current_pfd

            # Predict delay using the loaded pipeline (handles preprocessing)
            try:
                # The pipeline expects a DataFrame with original features
                predicted_delay = model.predict(row_data)[0]
                # Ensure prediction is a sensible numerical value (e.g., not NaN/inf)
                if not np.isfinite(predicted_delay):
                    print(f"Warning: Non-finite prediction ({predicted_delay}) for index {index}. Setting to 0.")
                    predicted_delay = 0.0
            except Exception as pred_e:
                 print(f"\nError predicting for index {index}, TailNum {tail_num}: {pred_e}")
                 # Consider logging problematic features: print(f"Features: {row_data.iloc[0].to_dict()}")
                 predicted_delay = 0.0 # Default on error

            # Store the prediction using the original index
            predictions_dict[index] = predicted_delay

            # Update last_predicted_delay for the next iteration within THIS path
            last_predicted_delay = predicted_delay

    # --- Assign predictions back to the main DataFrame ---
    if predictions_dict:
        print(f"Assigning {len(predictions_dict)} predictions...")
        pred_series = pd.Series(predictions_dict)
        # Assign based on index, overwriting existing NaNs or values
        df[config.PREDICTED_DELAY_COL] = pred_series
        # Fill any remaining NaNs ONLY if PREDICTED_DELAY_COL was newly created or had NaNs
        # If prediction failed for some rows, they will remain NaN unless filled here.
        nan_preds_after = df[config.PREDICTED_DELAY_COL].isnull().sum()
        if nan_preds_after > 0:
             print(f"Warning: {nan_preds_after} points were not predicted (remained NaN). Filling with 0.")
             df[config.PREDICTED_DELAY_COL].fillna(0.0, inplace=True)
        print(f"Predicted Delay stats:\n{df[config.PREDICTED_DELAY_COL].describe()}")
    else:
         print("Warning: No predictions were generated.")


    print("Iterative prediction complete.")
    return df


def save_predictions(df, file_path):
    """Saves the dataframe with predictions."""
    print(f"Saving final predictions to: {file_path}")
    if df is None or df.empty:
        print("Dataframe is empty. Skipping save.")
        return
    try:
        df_copy = df.copy()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Format datetime back to string for saving if needed
        if config.SCHED_DATETIME_COL in df_copy.columns:
             if pd.api.types.is_datetime64_any_dtype(df_copy[config.SCHED_DATETIME_COL]):
                  df_copy[config.SCHED_DATETIME_COL] = df_copy[config.SCHED_DATETIME_COL].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Select relevant columns for final output (optional, but good practice)
        output_columns = [
             config.TAIL_NUM_COL, config.SCHED_DATETIME_COL, config.ORIENTATION_COL,
             config.ORIGIN_COL, config.DEST_COL, config.CARRIER_CODE_COL, config.FLIGHT_NUM_COL,
             config.FTD_COL, config.PFD_COL, # The features used
             config.FLIGHT_DELAY_COL, # Original actual/assigned delay (for comparison if available)
             config.PREDICTED_DELAY_COL # The final prediction
        ]
        # Keep only columns that actually exist in the dataframe
        output_columns = [col for col in output_columns if col in df_copy.columns]
        df_output = df_copy[output_columns]

        df_output.to_csv(file_path, index=False)
        print(f"Save complete. Final predictions shape: {df_output.shape}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        # import traceback; traceback.print_exc()


def run_prediction():
    """Orchestrates loading model, future data, and running prediction."""
    model = load_model()
    df_future = load_future_data()
    df_predictions = iterative_prediction(df_future, model)
    save_predictions(df_predictions, config.FINAL_PREDICTIONS)
    print("Prediction pipeline finished.")

if __name__ == "__main__":
    run_prediction()
