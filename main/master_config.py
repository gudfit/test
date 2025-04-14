# master_predictor/master_config.py
import pathlib
import torch
import pandas as pd
from datetime import datetime

MASTER_PREDICTOR_DIR           = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT                   = MASTER_PREDICTOR_DIR.parent
LOCAL_MODELS_DIR               = MASTER_PREDICTOR_DIR / 'models'
CHAIN_MODEL_FILENAME           = 'flight_chain_model_best.pt'
REGRESSOR_MODEL_FILENAME       = 'fdpp_ml_voting_model.joblib'

CHAIN_CLASSIFIER_DIR           = PROJECT_ROOT / 'flightChainClassifier'
CHAIN_CLASSIFIER_RESULTS_DIR   = CHAIN_CLASSIFIER_DIR / 'results'
CHAIN_CLASSIFIER_PROCESSED_DIR = CHAIN_CLASSIFIER_DIR / 'mlData' / 'processed_chains'

FLIGHT_DELAY_DIR               = PROJECT_ROOT / 'flightDelay'
FLIGHT_DELAY_RESULTS_DIR       = FLIGHT_DELAY_DIR / 'results'

ORIGINAL_CHAIN_MODEL_PATH      = CHAIN_CLASSIFIER_RESULTS_DIR / 'models' / CHAIN_MODEL_FILENAME
ORIGINAL_REGRESSOR_MODEL_PATH  = FLIGHT_DELAY_RESULTS_DIR / 'models'     / REGRESSOR_MODEL_FILENAME

CHAIN_DATA_STATS_FILE          = CHAIN_CLASSIFIER_PROCESSED_DIR          / 'data_stats.json'
ORIGINAL_HYPERPARAMS_PATH      = CHAIN_CLASSIFIER_RESULTS_DIR            / 'best_hyperparameters.json'

DEVICE                         = "cuda" if torch.cuda.is_available() else "cpu"
CHAIN_LENGTH                   = 3
CHAIN_TARGET_CLASSES           = 5
CHAIN_MAX_GROUND_TIME          = pd.Timedelta(hours=12)
CHAIN_MIN_TURNAROUND_MINS      = 15

CHAIN_DELAY_THRESHOLDS         = [-float('inf'), 15, 60, 120, 240, float('inf')]

EXPECTED_INPUT_COLS            = [
    'FlightDate',
    'Tail_Number',
    'Reporting_Airline',
    'Flight_Number_Reporting_Airline',
    'Origin',
    'Dest',
    'CRSDepTime',
    'DepTime',
    'DepDelayMinutes',
    'CRSArrTime',
    'ArrTime',
    'ArrDelayMinutes',
    'CRSElapsedTime',
    'ActualElapsedTime',
    'AirTime',
    'Distance',
    'WeatherDelay',
    'Cancelled',
    'Diverted',
]

REGRESSOR_RAW_FEATURES         = [
    'Origin',
    'Dest',
    'Reporting_Airline',
    'Month',
    'DayOfWeek',
    'Hour',
    'FTD',
    'PFD',
]

if not CHAIN_DATA_STATS_FILE.exists():
    print(f"WARNING: Chain Classifier data stats not found at {CHAIN_DATA_STATS_FILE}")

local_chain_path               = LOCAL_MODELS_DIR / CHAIN_MODEL_FILENAME
local_regressor_path           = LOCAL_MODELS_DIR / REGRESSOR_MODEL_FILENAME

if not local_chain_path.exists() and not ORIGINAL_CHAIN_MODEL_PATH.exists():
    print(f"ERROR: Chain Model cannot be found locally ({local_chain_path}) or in original location ({ORIGINAL_CHAIN_MODEL_PATH})")

if not local_regressor_path.exists() and not ORIGINAL_REGRESSOR_MODEL_PATH.exists():
    print(f"ERROR: Regressor Model cannot be found locally ({local_regressor_path}) or in original location ({ORIGINAL_REGRESSOR_MODEL_PATH})")
