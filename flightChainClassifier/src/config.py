# flightChainClassifier/src/config.py
import pathlib
import torch
import pandas as pd
import json

# --- Project Structure ---
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
ML_DATA_DIR = PROJECT_ROOT / 'mlData'
PROCESSED_DATA_DIR = ML_DATA_DIR / 'processed_chains'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
EVAL_DIR = RESULTS_DIR / 'evaluation'
PLOTS_DIR = RESULTS_DIR / 'plots'

# --- Input Data ---
RAW_DATA_FILE = DATA_DIR / '0_merged_raw_flights.csv' # Reusing merged file

# --- Data Processing Parameters ---
# *** YOU MUST REVIEW AND ADJUST THIS LIST BASED ON YOUR AVAILABLE COLUMNS ***
# Aim for features covering time, location, airline, aircraft, delays, flight stats
REQUIRED_RAW_COLS = [
    'FlightDate', 'Tail_Number', 'Reporting_Airline', 'Flight_Number_Reporting_Airline',
    'Origin', 'Dest',
    'CRSDepTime', 'DepTime', # Need Actual DepTime for validation potentially
    'DepDelay', 'DepDelayMinutes', #'DepDel15',
    'CRSArrTime', 'ArrTime', # Need Actual ArrTime for validation potentially
    'ArrDelay', 'ArrDelayMinutes', #'ArrDel15', # Use ArrDelayMinutes as target
    'Cancelled', 'Diverted',
    'CRSElapsedTime', 'ActualElapsedTime', 'AirTime', 'Distance',
    'WeatherDelay',
    # Add more relevant columns available in your merged file
]
# The actual number of features AFTER processing (encoding, scaling, deriving)
# Let's set this dynamically after feature engineering in chain_constructor.py
# NUM_FEATURES = 38 # Target number - This will be saved in data_stats.json instead

CHAIN_LENGTH = 3 # Number of flights per chain
TARGET_COL_RAW = 'ArrDelayMinutes' # Raw delay column used for labeling

# Max time diff allowed between arrival of flight N and departure of flight N+1
# Set to None to disable this check, or a value like pd.Timedelta(hours=12)
MAX_GROUND_TIME = pd.Timedelta(hours=12) # Example: 12 hours max allowed ground time

# Delay Classification Thresholds (in minutes, based on Arrival Delay)
# Bins: (-inf, 15], (15, 60], (60, 120], (120, 240], (240, inf)
# Labels:   0,       1,        2,         3,          4
DELAY_THRESHOLDS = [-float('inf'), 15, 60, 120, 240, float('inf')]
NUM_CLASSES = 5

# Train/Val/Test Split Ratios
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_STATE = 42

# --- Model Hyperparameters ---
# General
SUBSAMPLE_DATA = 0.2 # Fraction for initial testing (1.0 for full data) << SET TO 1.0 FOR FULL RUN
BATCH_SIZE = 32 # Reduce if GPU memory is low
EPOCHS = 10 # Start with few epochs for testing
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5 # Added weight decay for regularization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NUM_WORKERS = 4 # For DataLoader, adjust based on CPU cores

# Model Specific (Examples, adjust as needed)
# CNN Blocks
DEFAULT_CNN_CHANNELS = [64, 128, 256]
DEFAULT_KERNEL_SIZE = 3
DEFAULT_LSTM_HIDDEN_SIZE = 256
DEFAULT_LSTM_NUM_LAYERS = 2
DEFAULT_LSTM_BIDIRECTIONAL = False
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5

# LSTM/GRU (Using standard LSTM for now)
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 1
LSTM_BIDIRECTIONAL = False # Standard LSTM

# --- Tuned Hyperparameters File ---
BEST_PARAMS_FILE = RESULTS_DIR / "best_hyperparameters.json"

# --- Output Files ---
# Processed data
TRAIN_CHAINS_FILE = PROCESSED_DATA_DIR / 'train_chains.npy'
TRAIN_LABELS_FILE = PROCESSED_DATA_DIR / 'train_labels.npy'
VAL_CHAINS_FILE = PROCESSED_DATA_DIR / 'val_chains.npy'
VAL_LABELS_FILE = PROCESSED_DATA_DIR / 'val_labels.npy'
TEST_CHAINS_FILE = PROCESSED_DATA_DIR / 'test_chains.npy'
TEST_LABELS_FILE = PROCESSED_DATA_DIR / 'test_labels.npy'
DATA_STATS_FILE = PROCESSED_DATA_DIR / 'data_stats.json' # For scaler params, num_features, etc.
MODEL_SAVE_PATH = MODELS_DIR / 'flight_chain_model_best.pt'

# Models (Using generic name, specific model type tracked elsewhere)
MODEL_SAVE_PATH = MODELS_DIR / 'flight_chain_model_best.pt'

# --- Ensure Directories Exist ---
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"--- Configuration ---")
print(f"Device: {DEVICE}")
print(f"Raw Data File: {RAW_DATA_FILE}")
print(f"Subsample Fraction: {SUBSAMPLE_DATA if SUBSAMPLE_DATA < 1.0 else 'None (Full Data)'}")
print(f"Chain Length: {CHAIN_LENGTH}")
print(f"Number of Classes: {NUM_CLASSES}")
print(f"Target Raw Column: {TARGET_COL_RAW}")
print(f"Max Allowed Ground Time: {MAX_GROUND_TIME}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Output Model Path: {MODEL_SAVE_PATH}")
print(f"---------------------")

# --- Helper function (optional) ---
def load_data_stats():
    """Loads statistics saved during data processing."""
    if DATA_STATS_FILE.exists():
        try:
            with open(DATA_STATS_FILE, 'r') as f:
                stats = json.load(f)
            return stats
        except Exception as e:
            print(f"Warning: Could not load data stats from {DATA_STATS_FILE}: {e}")
            return None
    else:
        print(f"Warning: Data stats file not found at {DATA_STATS_FILE}")
        return None

def load_best_hyperparameters():
    """Loads the best hyperparameters found during tuning."""
    if BEST_PARAMS_FILE.exists():
        print(f"Loading best hyperparameters from: {BEST_PARAMS_FILE}")
        try:
            with open(BEST_PARAMS_FILE, 'r') as f:
                params = json.load(f)
            print("Best hyperparameters loaded successfully.")
            return params
        except Exception as e:
            print(f"Warning: Could not load best hyperparameters from {BEST_PARAMS_FILE}: {e}")
            return None
    else:
        print(f"Warning: Best hyperparameters file not found at {BEST_PARAMS_FILE}. Using defaults.")
        return None
