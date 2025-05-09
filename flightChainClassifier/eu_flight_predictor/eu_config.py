# eu_flight_predictor/eu_config.py
import pathlib
import torch
import json
import os
import sys  # For sys.exit

# --- Core Project Path Configuration ---

# Get the directory where this eu_config.py file is located
EU_FLIGHT_PREDICTOR_ROOT = pathlib.Path(__file__).resolve().parent

# MAIN_PROJECT_ROOT should be the parent directory of EU_FLIGHT_PREDICTOR_ROOT
# if eu_flight_predictor is directly inside flightChainClassifier
MAIN_PROJECT_ROOT = EU_FLIGHT_PREDICTOR_ROOT.parent

# --- Sanity check for MAIN_PROJECT_ROOT ---
expected_src_dir = MAIN_PROJECT_ROOT / "src"
if (
    not MAIN_PROJECT_ROOT.exists()
    or not MAIN_PROJECT_ROOT.is_dir()
    or MAIN_PROJECT_ROOT.name != "flightChainClassifier"
):
    print(f"CRITICAL ERROR: MAIN_PROJECT_ROOT is currently '{MAIN_PROJECT_ROOT}'")
    print(
        f"       It does not seem to be the correct 'flightChainClassifier' directory."
    )
    print(f"       Detected EU_FLIGHT_PREDICTOR_ROOT: {EU_FLIGHT_PREDICTOR_ROOT}")
    print(
        f"       Please ensure 'eu_flight_predictor' directory is directly inside your 'flightChainClassifier' project directory."
    )
    print(
        f"       If your structure is different, you'll need to adjust MAIN_PROJECT_ROOT manually in eu_config.py."
    )
    sys.exit(1)  # Exit if this fundamental path is wrong
elif not expected_src_dir.exists() or not expected_src_dir.is_dir():
    print(
        f"CRITICAL ERROR: MAIN_PROJECT_ROOT is '{MAIN_PROJECT_ROOT}', which seems to be 'flightChainClassifier',"
    )
    print(
        f"       but the 'src' subdirectory is missing within it ({expected_src_dir})."
    )
    print(f"       The 'src' directory is essential for importing models.")
    sys.exit(1)  # Exit if src is missing


# --- EU-Specific Data and Directories ---
# Directory containing the individual cleaned CSV files to be merged
EU_INDIVIDUAL_CLEANED_CSVS_DIR = (
    EU_FLIGHT_PREDICTOR_ROOT / "data" / "EU_Flights_Cleaned"
)

# Directory where the merged EU data file will be saved and read from
EU_MERGED_DATA_OUTPUT_DIR = EU_FLIGHT_PREDICTOR_ROOT / "merged_eu_data"
MERGED_EU_DATA_FILE = EU_MERGED_DATA_OUTPUT_DIR / "merged_eu_flights.csv"

# Output directory for processed EU chains, labels, and predictions
PROCESSED_EU_CHAINS_DIR = EU_FLIGHT_PREDICTOR_ROOT / "processed_eu_data"
EU_TEST_CHAINS_FILE = PROCESSED_EU_CHAINS_DIR / "eu_test_chains.npy"
EU_TEST_LABELS_FILE = PROCESSED_EU_CHAINS_DIR / "eu_test_labels.npy"
EU_DATA_STATS_FILE = PROCESSED_EU_CHAINS_DIR / "eu_data_processing_stats.json"

# --- EU Data Column Mapping and Processing ---
EU_COLUMN_MAPPING_RAW = {
    # Internal Name       : CSV Column Name in EU Data (based on your sample)
    "FlightDate": "flight_date",
    "Tail_Number": "tail_number",
    "Reporting_Airline": "airline",
    "Origin": "depart_from_iata",
    "Dest": "arrive_at_iata",
    "CRSDepTime": None,  # Will be derived from *_utc columns if available
    "DepTime": None,  # Will be derived from *_utc columns if available
    "DepDelay": "departure_delay",  # Direct from CSV
    "DepDelayMinutes": "departure_delay",  # Direct from CSV (often same as DepDelay)
    "CRSArrTime": None,  # Will be derived from *_utc columns if available
    "ArrTime": None,  # Will be derived from *_utc columns if available
    "ArrDelay": "arrival_delay",  # Direct from CSV
    "ArrDelayMinutes": "arrival_delay",  # TARGET_COL_INTERNAL_NAME refers to this, direct from CSV
    "Cancelled": None,  # To be derived from 'status' column
    "Diverted": None,  # To be derived from 'status' column
    "CRSElapsedTime": "scheduled_duration",  # Direct from CSV (might need conversion if not in minutes)
    "ActualElapsedTime": "actual_duration",  # Direct from CSV (might need conversion if not in minutes)
    "AirTime": None,  # Not in your sample, map if available or will be filled with 0/default
    "Distance": "distance",  # Direct from CSV
    "TaxiOut": None,  # Not in your sample, map if available or will be filled with 0/default
    "TaxiIn": None,  # Not in your sample, map if available or will be filled with 0/default
}

# Configuration for columns in EU data that store full datetime strings
# from which HHMM needs to be extracted.
# Map: New Internal HHMM Column Name : Source CSV Column Name with Full Datetime
EU_DATETIME_COLS_NEED_HHMM_PROCESSING = {
    "CRSDepTime_hhmm": "scheduled_departure_utc",  # e.g., CSV has '2022-03-01 01:35:00'
    "CRSArrTime_hhmm": "scheduled_arrival_utc",
    "DepTime_hhmm": "actual_departure_utc",
    "ArrTime_hhmm": "actual_arrival_utc",
}

# Name of the column in the EU CSV that indicates flight status (e.g., "Cancelled", "Diverted", "On-Time")
EU_STATUS_COL_CSV_NAME = "status"  # Based on your sample

# Internal name for the target variable (must match what `eu_chain_constructor` expects for labeling)
TARGET_COL_INTERNAL_NAME = "ArrDelayMinutes"

# --- Chain Construction Parameters (should align with original model training) ---
CHAIN_LENGTH = 3
MAX_GROUND_TIME_HOURS = 12
MIN_TURNAROUND_MINUTES = 15
DELAY_THRESHOLDS_CLASSIFICATION = [-float("inf"), 15, 60, 120, 240, float("inf")]
NUM_CLASSES_CLASSIFICATION = len(DELAY_THRESHOLDS_CLASSIFICATION) - 1  # Should be 5
RANDOM_STATE = 42

# --- Delay Status Mapping for EU Predictions ---
DELAY_STATUS_MAP_EU = {
    0: "On Time / Slight Delay (<= 15 min)",
    1: "Delayed (15-60 min)",
    2: "Significantly Delayed (60-120 min)",
    3: "Severely Delayed (120-240 min)",
    4: "Extremely Delayed (> 240 min)",
}
UNKNOWN_STATUS_EU = "Unknown Delay Status"

# --- Model Prediction Configuration ---
# Define which model type TO INSTANTIATE for EU predictions.
# Ensure this matches the architecture saved in MODEL_FILENAME_FOR_EU_PREDICTION
EU_PREDICTION_MODEL_TYPE = "qtsimam"  # Options: "cbam", "simam", "qtsimam", "qtsimam_mp" # Set to simam to match the file weights

# --- ADDED FLAG to Force Bidirectional LSTM/QMogrifier ---
# Set to True if the loaded model (.pt file) was trained with bidirectional=True
# Set to False if the loaded model was trained with bidirectional=False
# Set to None to rely on hyperparameters loaded from .meta.json or best_params.json (or default to False if none found)
FORCE_LSTM_BIDIR_FOR_EU = (
    True  # Set to True because your saved model expects bidirectional keys
)
# ---------------------------------

# Path to the pre-trained model (.pt file) from the flightChainClassifier project
MODEL_FILENAME_FOR_EU_PREDICTION = (
    "flight_chain_model_best.pt"  # Using generic name as per your tree
)
MODEL_PATH_FOR_EU_PREDICTION = (
    MAIN_PROJECT_ROOT / "results" / "models" / MODEL_FILENAME_FOR_EU_PREDICTION
)

# Path to the data_stats.json file generated during the original flightChainClassifier training.
ORIGINAL_DATA_STATS_PATH = (
    MAIN_PROJECT_ROOT / "mlData" / "processed_chains" / "data_stats.json"
)

# Path template for the hyperparameter file (e.g., best_hyperparameters_qtsimam.json)
ORIGINAL_MODEL_HYPERPARAMS_PATH_TEMPLATE = (
    MAIN_PROJECT_ROOT / "results" / "best_hyperparameters_{model_type}.json"
)
# Path to the .meta.json file which might contain hyperparameters for the generic model
MODEL_META_JSON_PATH = (
    MAIN_PROJECT_ROOT / "results" / "models" / "flight_chain_model_best.meta.json"
)


# --- General Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Helper Functions to Load Artifacts from flightChainClassifier Training ---
def load_original_data_stats_for_scaling():
    """Loads the data_stats.json from the original model training."""
    if not ORIGINAL_DATA_STATS_PATH.exists():
        print(
            f"ERROR: Original data_stats.json not found at {ORIGINAL_DATA_STATS_PATH}"
        )
        return None
    try:
        with open(ORIGINAL_DATA_STATS_PATH, "r") as f:
            stats = json.load(f)
        print(
            f"Successfully loaded original data stats from: {ORIGINAL_DATA_STATS_PATH}"
        )
        return stats
    except Exception as e:
        print(
            f"Error loading original data_stats.json from {ORIGINAL_DATA_STATS_PATH}: {e}"
        )
        return None


def load_model_hyperparameters():
    """
    Loads model hyperparameters.
    Priority:
    1. Specific best_hyperparameters_{model_type}.json (if you create these per model type for EU)
    2. Model's .meta.json file (flight_chain_model_best.meta.json), as it's tied to the generic model file
    3. Generic best_hyperparameters.json (if you create one in results/)
    """
    hyperparams_loaded_from = None
    hyperparams = None

    # 1. Try specific best_hyperparameters_{model_type}.json
    specific_hyperparams_file = pathlib.Path(
        str(ORIGINAL_MODEL_HYPERPARAMS_PATH_TEMPLATE).format(
            model_type=EU_PREDICTION_MODEL_TYPE
        )
    )
    if specific_hyperparams_file.exists():
        try:
            with open(specific_hyperparams_file, "r") as f:
                hyperparams = json.load(f)
            hyperparams_loaded_from = specific_hyperparams_file
        except Exception as e:
            print(
                f"Warning: Error loading specific hyperparameter file {specific_hyperparams_file}: {e}"
            )

    # 2. If not found or error, try model's .meta.json (flight_chain_model_best.meta.json)
    if hyperparams is None and MODEL_META_JSON_PATH.exists():
        try:
            with open(MODEL_META_JSON_PATH, "r") as f:
                meta_data = json.load(f)
                # .meta.json might store hyperparams directly or nested. Adjust if necessary.
                # Assuming they are top-level keys in the .meta.json
                hyperparams = meta_data
            hyperparams_loaded_from = MODEL_META_JSON_PATH
        except Exception as e:
            print(
                f"Warning: Error loading or parsing .meta.json file {MODEL_META_JSON_PATH}: {e}"
            )

    # 3. (Optional) Fallback to a very generic best_hyperparameters.json
    if hyperparams is None:
        generic_hyperparams_file = (
            MAIN_PROJECT_ROOT / "results" / "best_hyperparameters.json"
        )
        if (
            generic_hyperparams_file.exists()
        ):  # Your tree doesn't show this, but as a fallback
            try:
                with open(generic_hyperparams_file, "r") as f:
                    hyperparams = json.load(f)
                hyperparams_loaded_from = generic_hyperparams_file
            except Exception as e:
                print(
                    f"Warning: Error loading generic hyperparameter file {generic_hyperparams_file}: {e}"
                )

    if hyperparams:
        print(
            f"Successfully loaded model hyperparameters for '{EU_PREDICTION_MODEL_TYPE}' from: {hyperparams_loaded_from}"
        )
    else:
        print(
            f"Warning: Could not load hyperparameters for model type '{EU_PREDICTION_MODEL_TYPE}'. Fallback defaults will be used by the predictor."
        )

    return hyperparams


# --- Sanity Checks and Info ---
print("--- EU Predictor Configuration ---")
print(f"EU Flight Predictor Root: {EU_FLIGHT_PREDICTOR_ROOT}")
print(f"Main Project (flightChainClassifier) Root: {MAIN_PROJECT_ROOT}")
print(f"Directory for individual cleaned EU CSVs: {EU_INDIVIDUAL_CLEANED_CSVS_DIR}")
print(
    f"EU Merged Data File (Output of merge, Input for constructor): {MERGED_EU_DATA_FILE}"
)
print(f"Processed EU Chains Directory: {PROCESSED_EU_CHAINS_DIR}")
print(f"Prediction Model Type: {EU_PREDICTION_MODEL_TYPE.upper()}")
print(
    f"Force LSTM Bidirectional for EU: {FORCE_LSTM_BIDIR_FOR_EU}"
)  # Print the flag value
print(f"Path to Pre-trained Model for EU: {MODEL_PATH_FOR_EU_PREDICTION}")
print(f"Path to Original Data Stats (for scaling): {ORIGINAL_DATA_STATS_PATH}")
print(
    f"Path for Model Hyperparameters (best attempt): {ORIGINAL_MODEL_HYPERPARAMS_PATH_TEMPLATE.parent / f'best_hyperparameters_{EU_PREDICTION_MODEL_TYPE}.json'} or {MODEL_META_JSON_PATH}"
)
print(f"Device: {DEVICE}")
print(f"Chain Length: {CHAIN_LENGTH}, Num Classes: {NUM_CLASSES_CLASSIFICATION}")
print(f"Delay Status Map for EU: {DELAY_STATUS_MAP_EU}")
print("--- End EU Predictor Configuration ---")

# Ensure output directories exist
EU_INDIVIDUAL_CLEANED_CSVS_DIR.mkdir(parents=True, exist_ok=True)
EU_MERGED_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_EU_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
