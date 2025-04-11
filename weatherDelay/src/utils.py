# FILE: src/utils.py
# --------------------------------------------------------------------------------
import joblib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd # Added pandas for json serializer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom JSON Serializer ---
def _default_serializer(obj):
    """ For saving evaluation results containing numpy/pandas types """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        # Handle NaN specifically for JSON compatibility
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj): # Handle pandas NaN/NaT
        return None
    elif isinstance(obj, Path): # Handle Path objects
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable for JSON")

# --- Model Handling ---
def save_model(model, file_path: Path):
    """Saves a trained model/pipeline using joblib."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, file_path)
        logging.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving model to {file_path}: {e}")

def load_model(file_path: Path):
    """Loads a trained model/pipeline using joblib."""
    if not file_path.exists():
        logging.error(f"Model file not found: {file_path}")
        return None
    try:
        model = joblib.load(file_path)
        logging.info(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {file_path}: {e}")
        return None

# --- Evaluation Results Handling ---
def save_evaluation_summary(results_list: list, file_path: Path):
    """Saves a list of evaluation result dictionaries to a CSV file."""
    if not results_list:
        logging.warning("No evaluation results to save.")
        return
    try:
        df = pd.DataFrame(results_list)
        # Optional: Sort by the configured best metric
        if config.BEST_MODEL_METRIC in df.columns:
            ascending = True # Assume lower is better for RMSE, MAE etc.
            df = df.sort_values(by=config.BEST_MODEL_METRIC, ascending=ascending)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False, float_format='%.4f')
        logging.info(f"Saved evaluation summary (sorted by {config.BEST_MODEL_METRIC}) to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save evaluation summary: {e}")

def save_dict_to_json(data: dict, file_path: Path):
    """Saves a dictionary to a JSON file using custom serializer."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4, default=_default_serializer)
        logging.info(f"Dictionary saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving dictionary to JSON {file_path}: {e}")

# --- Data Handling ---
def reduce_mem_usage(df, verbose=True):
    """Iterate through columns and downcast numeric types to save memory."""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Use float32 instead of float16 for wider compatibility / less precision loss
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logging.info(f'Mem. usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

# --- Import necessary configurations ---
try:
    from src import config
except ImportError:
    logging.error("Could not import config module in utils. Ensure it exists and is importable.")
    # Define a placeholder config object or raise an error
    class PlaceholderConfig:
        BEST_MODEL_METRIC = 'rmse' # Default fallback
    config = PlaceholderConfig()
