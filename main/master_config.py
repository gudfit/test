# master_predictor/master_config.py
import pathlib
import torch
import pandas as pd
from datetime import datetime

# --- General Project Structure ---
# Assuming this config is inside 'master_predictor' directory
MASTER_PREDICTOR_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = MASTER_PREDICTOR_DIR.parent # Assumes master_predictor is at the root alongside flightChainClassifier and flightDelay

# --- Flight Chain Classifier Artifacts ---
CHAIN_CLASSIFIER_DIR = PROJECT_ROOT / 'flightChainClassifier'
CHAIN_CLASSIFIER_RESULTS_DIR = CHAIN_CLASSIFIER_DIR / 'results'
CHAIN_CLASSIFIER_PROCESSED_DIR = CHAIN_CLASSIFIER_DIR / 'mlData' / 'processed_chains'

# Path to the trained PyTorch model (.pt file)
# *** USER: Make sure this points to the correct saved model ***
# Example: Assumes the best QTSimAM model was saved
CHAIN_MODEL_PATH = CHAIN_CLASSIFIER_RESULTS_DIR / 'models' / 'flight_chain_model_best.pt'
# Path to the data statistics (for preprocessing)
CHAIN_DATA_STATS_FILE = CHAIN_CLASSIFIER_PROCESSED_DIR / 'data_stats.json'
# Chain length used during training (MUST match the trained model)
CHAIN_LENGTH = 3 # Get this from the classifier's original config if needed
CHAIN_TARGET_CLASSES = 5 # Number of output classes
CHAIN_MAX_GROUND_TIME = pd.Timedelta(hours=12) # From classifier's config
CHAIN_MIN_TURNAROUND_MINS = 15 # Minimum ground time in minutes for valid chain

# --- Flight Delay Regressor Artifacts ---
FLIGHT_DELAY_DIR = PROJECT_ROOT / 'flightDelay'
FLIGHT_DELAY_RESULTS_DIR = FLIGHT_DELAY_DIR / 'results'
FLIGHT_DELAY_DATA_DIR = FLIGHT_DELAY_DIR / 'data'
FLIGHT_DELAY_ML_DATA_DIR = FLIGHT_DELAY_DIR / 'mlData'

# Path to the trained scikit-learn pipeline (.joblib file)
# *** USER: Make sure this points to the correct saved model ***
REGRESSOR_MODEL_PATH = FLIGHT_DELAY_RESULTS_DIR / 'models' / 'fdpp_ml_voting_model.joblib'
# Optional: Path to airport coordinates if regressor uses orientation features
AIRPORT_COORDS_FILE = FLIGHT_DELAY_DATA_DIR / 'iata-icao.csv'

# --- Prediction Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Columns expected in the input dictionaries for prediction
# Should contain fields needed by *both* preprocessing pipelines
# Adjust based on the actual features used in both original projects
EXPECTED_INPUT_COLS = [
    'FlightDate', # YYYY-MM-DD or parsable date format
    'Tail_Number',
    'Reporting_Airline',
    'Flight_Number_Reporting_Airline',
    'Origin',
    'Dest',
    'CRSDepTime', # HHMM format (int or string)
    'DepTime',    # HHMM format (int or string) - Needed for actual ground time if used by classifier
    'DepDelayMinutes', # Or DepDelay - needed if used as feature or for PFD if target is arrival
    'CRSArrTime', # HHMM format (int or string)
    'ArrTime',    # HHMM format (int or string) - Needed for actual ground time if used by classifier
    'ArrDelayMinutes', # Or ArrDelay - needed if used as feature or for PFD if target is departure
    'CRSElapsedTime', # Needed for scheduled duration calc
    'ActualElapsedTime', # Potentially needed by classifier features
    'AirTime', # Potentially needed by classifier features
    'Distance', # Potentially needed by classifier features
    'WeatherDelay', # Potentially needed by classifier features
    'Cancelled', # 0.0 or 1.0
    'Diverted', # 0.0 or 1.0
    # Add any other columns used as features by EITHER model
    # e.g., 'TaxiOut', 'TaxiIn' if used by classifier
]
CHAIN_DELAY_THRESHOLDS = [-float('inf'), 15, 60, 120, 240, float('inf')]
# Features the Regressor model pipeline expects (raw, before pipeline's internal preprocessing)
# FTD and PFD are calculated dynamically during prediction
REGRESSOR_RAW_FEATURES = ['Origin', 'Dest', 'Reporting_Airline', # Categorical
                          'Month', 'DayOfWeek', 'Hour',          # Temporal (derived)
                          'FTD', 'PFD']                           # FDPP Features (calculated)
REGRESSOR_TARGET_COL = 'Flight_Delay' # Name used internally by regressor logic

# --- Ensure Artifacts Exist (Basic Checks) ---
print("--- Master Predictor Configuration ---")
print(f"Device: {DEVICE}")
print(f"Chain Length: {CHAIN_LENGTH}")
print(f"Chain Model Path: {CHAIN_MODEL_PATH}")
print(f"Chain Stats Path: {CHAIN_DATA_STATS_FILE}")
print(f"Regressor Model Path: {REGRESSOR_MODEL_PATH}")

if not CHAIN_MODEL_PATH.exists():
    print(f"ERROR: Chain Classifier model not found at {CHAIN_MODEL_PATH}")
if not CHAIN_DATA_STATS_FILE.exists():
    print(f"ERROR: Chain Classifier data stats not found at {CHAIN_DATA_STATS_FILE}")
if not REGRESSOR_MODEL_PATH.exists():
    print(f"ERROR: Regressor model not found at {REGRESSOR_MODEL_PATH}")
print("-------------------------------------")
