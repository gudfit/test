"""
weather_predictor.py

This module loads a pre-trained weather model and predicts weather metrics (e.g., temperature, precipitation).
It demonstrates data preprocessing, model loading, and prediction.
"""

import os
import sys
import json
import argparse
import pandas as pd
import joblib
import traceback

# Import configuration from weather_config.py
try:
    from . import weather_config as config
except ImportError:
    import weather_config as config

class WeatherPredictor:
    """
    WeatherPredictor loads a weather model and uses it to predict weather conditions.
    """
    def __init__(self):
        print("Initializing WeatherPredictor...")
        self.config = config
        self.device = config.DEVICE  # if using torch-based models
        self.expected_features = config.EXPECTED_FEATURES
        
        # Determine paths before loading
        self.model_path = self._get_model_path(
            config.MODELS_DIR,
            config.WEATHER_MODEL_FILENAME,
            config.ORIGINAL_WEATHER_MODEL_PATH
        )
        
        # Load the weather model (here we use joblib; adjust if using different framework)
        self.model = self._load_weather_model()
        print("WeatherPredictor Initialization Complete.")

    def _get_model_path(self, local_dir, filename, fallback_path):
        """
        Check if the model exists in the local model directory; if not, use the fallback.
        """
        local_path = local_dir / filename
        if local_path.exists():
            print(f"Found weather model locally: {local_path}")
            return local_path
        elif fallback_path.exists():
            print(f"Model not found locally. Using fallback path: {fallback_path}")
            return fallback_path
        else:
            raise FileNotFoundError(
                f"Weather model '{filename}' not found in local dir ({local_path}) or fallback ({fallback_path})."
            )

    def _load_weather_model(self):
        """
        Load the weather prediction model from disk.
        """
        print(f"Loading weather model from: {self.model_path}")
        try:
            model = joblib.load(self.model_path)
            if not hasattr(model, 'predict'):
                raise TypeError("Loaded object is not a valid predictive model.")
            print("Weather model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading weather model: {e}")
            traceback.print_exc()
            raise

    def predict(self, input_data):
        """
        Preprocess the input data and perform weather prediction.
        :param input_data: A list or DataFrame of weather observations.
        :return: Model predictions.
        """
        # Ensure input_data is a pandas DataFrame
        if isinstance(input_data, list):
            try:
                df = pd.DataFrame(input_data)
            except Exception as e:
                print(f"Error converting input list to DataFrame: {e}")
                return None
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            print("Error: Input data must be a list of dictionaries or a pandas DataFrame.")
            return None

        # Check if expected features are present
        missing = [feature for feature in self.expected_features if feature not in df.columns]
        if missing:
            print(f"Error: Missing expected features: {missing}")
            return None

        # Optionally, you might apply preprocessing (scaling, encoding) here
        # For this example, we assume the model can work directly with the DataFrame
        try:
            predictions = self.model.predict(df[self.expected_features])
            return predictions
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            return None


# --- Main Execution Block (CLI Handling) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict weather conditions using the pre-trained weather model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input-file", type=str, required=True,
        help="Path to a JSON file containing weather observations (list of dicts)."
    )
    args = parser.parse_args()
    print("\n--- Running WeatherPredictor ---")
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    try:
        with open(args.input_file, 'r') as f:
            weather_data = json.load(f)
        if not isinstance(weather_data, list) or not weather_data:
            print("Error: JSON file must contain a non-empty list of observations.")
            sys.exit(1)
        print(f"Read {len(weather_data)} weather records.")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    try:
        predictor = WeatherPredictor()
        preds = predictor.predict(weather_data)
        if preds is not None:
            print("\n--- Prediction Result ---")
            print(preds)
        else:
            print("Prediction failed.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

