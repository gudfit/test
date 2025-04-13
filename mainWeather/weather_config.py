"""
weather_config.py

This file contains configuration settings for the weather prediction project.
It sets up important directory paths, filenames for models, and other hyperparameters.
"""

import pathlib
import os
import json
import torch  # if using a torch-based model; otherwise, remove or adjust accordingly

# --- Project Structure ---
# Resolve the directory containing this config file
WEATHER_PROJECT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = WEATHER_PROJECT_DIR / 'data'
MODELS_DIR = WEATHER_PROJECT_DIR / 'models'
RESULTS_DIR = WEATHER_PROJECT_DIR / 'results'

# --- Model Filenames ---
# Name of the weather prediction model file (e.g., a joblib or pickle file)
WEATHER_MODEL_FILENAME = 'best_weather_model.joblib'

# --- Fallback paths (if needed) ---
# If the model is not found locally, you can define a fallback location.
ORIGINAL_WEATHER_MODEL_PATH = WEATHER_PROJECT_DIR.parent / 'weatherLegacy' / 'models' / WEATHER_MODEL_FILENAME

# --- Prediction Configuration ---
# Optionally define device settings if using torch (or similar)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# List of feature names expected by the model.
EXPECTED_FEATURES = [
    'temp_c', 'humidity_pct', 'pressure_mb', 'wind_speed_kmph',
    'precip_mm', 'cloud_cover_pct', 'visibility_m', 'dew_point_c'
]

# You might also store paths to statistics (scalers, encoders) if needed.
WEATHER_STATS_FILE = DATA_DIR / 'weather_stats.json'

# --- Log Configuration ---
print("--- Weather Predictor Configuration ---")
print(f"Device: {DEVICE}")
print(f"Project Root: {WEATHER_PROJECT_DIR}")
print(f"Data Directory: {DATA_DIR}")
print(f"Models Directory: {MODELS_DIR}")
print(f"Weather Model Filename: {WEATHER_MODEL_FILENAME}")
print(f"Fallback Model Path: {ORIGINAL_WEATHER_MODEL_PATH}")
print("-------------------------------------")

# --- Basic Checks ---
if not DATA_DIR.exists():
    print(f"WARNING: Data directory not found at {DATA_DIR}")
if not WEATHER_STATS_FILE.exists():
    print(f"WARNING: Weather statistics file not found at {WEATHER_STATS_FILE}")

