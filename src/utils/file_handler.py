# src/utils/file_handler.py
import joblib
import json
import logging
from pathlib import Path
import numpy as np # Needed for serializer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_model(model, file_path: Path):
    """Saves a trained model/pipeline using joblib or model's method."""
    try:
        # Add specific saving methods if needed (e.g., model.save_model for XGB/LGBM)
        # Defaulting to joblib for wider compatibility
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

def save_evaluation(results: dict, file_path: Path):
    """Saves evaluation results dictionary to a JSON file."""
    try:
        # Handle numpy types for JSON serialization
        def default_serializer(obj):
            if isinstance(obj, (np.integer, np.int64)): # Handle numpy integers
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)): # Handle numpy floats
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj): # Handle pandas NaN/NaT
                 return None
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4, default=default_serializer)
        logging.info(f"Evaluation results saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving evaluation results to {file_path}: {e}")
