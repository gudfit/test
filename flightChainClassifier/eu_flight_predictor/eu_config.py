# eu_flight_predictor/eu_config.py
import pathlib
import torch
import pandas as pd
import json  # For printing the mapping nicely

# --- Project Structure ---
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()  # This is /.../eu_flight_predictor
PROJECT_ROOT_EU_PREDICTOR = (
    SCRIPT_DIR  # Root of the current eu_flight_predictor sub-project
)

# MAIN_PROJECT_ROOT should point to the main 'flightChainClassifier' directory
# If eu_flight_predictor is a direct subfolder of flightChainClassifier:
# e.g., flightChainClassifier/eu_flight_predictor/eu_config.py
# then SCRIPT_DIR.parent is flightChainClassifier
MAIN_PROJECT_ROOT = SCRIPT_DIR.parent

# --- EU Data Specific ---
EU_RAW_DATA_DIR = PROJECT_ROOT_EU_PREDICTOR / "data" / "EU_Flights_Cleaned"
MERGED_EU_DATA_DIR = PROJECT_ROOT_EU_PREDICTOR / "merged_eu_data"
MERGED_EU_DATA_FILE = MERGED_EU_DATA_DIR / "merged_eu_flights.csv"
PROCESSED_EU_CHAINS_DIR = PROJECT_ROOT_EU_PREDICTOR / "processed_eu_chains"

# --- Column Mapping: Internal Name (Original Model's Expected Name) -> EU CSV Actual Column Name ---
EU_COLUMN_MAPPING_RAW = {
    "FlightDate": "flight_date",
    "Tail_Number": "tail_number",
    "Reporting_Airline": "airline",
    "Origin": "depart_from_iata",
    "Dest": "arrive_at_iata",
    "CRSDepTime": "scheduled_departure_utc",
    "CRSArrTime": "scheduled_arrival_utc",
    "DepDelayMinutes": "departure_delay",
    "ArrDelayMinutes": "arrival_delay",
    "DepTime": "actual_departure_utc",
    "ArrTime": "actual_arrival_utc",
    "CRSElapsedTime": "scheduled_duration",
    "ActualElapsedTime": "actual_duration",
    "Distance": "distance",
    "AirTime": None,
    "TaxiOut": None,
    "TaxiIn": None,
    "WeatherDelay": None,
}

EU_DATETIME_COLS_NEED_HHMM_PROCESSING = {
    "CRSDepTime_hhmm": EU_COLUMN_MAPPING_RAW.get(
        "CRSDepTime", "scheduled_departure_utc"
    ),
    "CRSArrTime_hhmm": EU_COLUMN_MAPPING_RAW.get("CRSArrTime", "scheduled_arrival_utc"),
    "DepTime_hhmm": EU_COLUMN_MAPPING_RAW.get("DepTime", "actual_departure_utc"),
    "ArrTime_hhmm": EU_COLUMN_MAPPING_RAW.get("ArrTime", "actual_arrival_utc"),
}
EU_STATUS_COL_CSV_NAME = "status"
TARGET_COL_INTERNAL_NAME = "ArrDelayMinutes"

# --- Chain Construction Parameters ---
CHAIN_LENGTH = 3
MAX_GROUND_TIME_HOURS = 12  # Consider increasing this if too many chains are skipped
MIN_TURNAROUND_MINUTES = 15
DELAY_THRESHOLDS_CLASSIFICATION = [-float("inf"), 15, 60, 120, 240, float("inf")]
NUM_CLASSES_CLASSIFICATION = 5
RANDOM_STATE = 42

# --- Model & Prediction ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths relative to MAIN_PROJECT_ROOT (flightChainClassifier directory)
PRETRAINED_MODEL_DIR_FROM_MAIN = MAIN_PROJECT_ROOT / "results" / "models"
PRETRAINED_MODEL_STATS_DIR_FROM_MAIN = MAIN_PROJECT_ROOT / "mlData" / "processed_chains"

CHAIN_MODEL_FILENAME = "flight_chain_model_best.pt"
MODEL_PATH_FOR_EU_PREDICTION = PRETRAINED_MODEL_DIR_FROM_MAIN / CHAIN_MODEL_FILENAME
DATA_STATS_PATH_FOR_EU_PREDICTION = (
    PRETRAINED_MODEL_STATS_DIR_FROM_MAIN / "data_stats.json"
)

# Path to the primary hyperparameter file (e.g., from Optuna for SimAM model)
GENERAL_HYPERPARAMS_FILE_PATH = (
    PRETRAINED_MODEL_DIR_FROM_MAIN.parent / "best_hyperparameters.json"
)  # results/best_hyperparameters.json
# Path to the .meta.json file, expected to be alongside the .pt model file
MODEL_META_JSON_PATH = MODEL_PATH_FOR_EU_PREDICTION.with_suffix(".meta.json")


# --- Output files for EU chain construction (within eu_flight_predictor) ---
EU_TEST_CHAINS_FILE = PROCESSED_EU_CHAINS_DIR / "eu_test_chains.npy"
EU_TEST_LABELS_FILE = PROCESSED_EU_CHAINS_DIR / "eu_test_labels.npy"
EU_DATA_STATS_FILE = PROCESSED_EU_CHAINS_DIR / "eu_data_stats_for_prediction.json"

# --- Ensure Directories Exist ---
MERGED_EU_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_EU_CHAINS_DIR.mkdir(parents=True, exist_ok=True)


def load_original_data_stats_for_scaling():
    if not DATA_STATS_PATH_FOR_EU_PREDICTION.exists():
        print(
            f"ERROR: Original data_stats.json for model scaling not found at {DATA_STATS_PATH_FOR_EU_PREDICTION}"
        )
        return None
    try:
        with open(DATA_STATS_PATH_FOR_EU_PREDICTION, "r") as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        print(
            f"Warning: Could not load original data stats from {DATA_STATS_PATH_FOR_EU_PREDICTION}: {e}"
        )
        return None


def load_model_hyperparameters():
    params_to_use = None
    loaded_from = ""

    if MODEL_META_JSON_PATH.exists():
        print(
            f"Attempting to load model architecture params from specific meta file: {MODEL_META_JSON_PATH}"
        )
        try:
            with open(MODEL_META_JSON_PATH, "r") as f:
                meta_params = json.load(f)
            params_to_use = meta_params
            loaded_from = str(MODEL_META_JSON_PATH.name)
            print(f"Successfully loaded params from {loaded_from}: {params_to_use}")
        except Exception as e:
            print(f"Warning: Could not load or parse {MODEL_META_JSON_PATH}: {e}")
            params_to_use = None

    if params_to_use is None and GENERAL_HYPERPARAMS_FILE_PATH.exists():
        print(
            f"Falling back to general hyperparameters file: {GENERAL_HYPERPARAMS_FILE_PATH}"
        )
        try:
            with open(GENERAL_HYPERPARAMS_FILE_PATH, "r") as f:
                general_params = json.load(f)
            params_to_use = general_params
            loaded_from = str(GENERAL_HYPERPARAMS_FILE_PATH.name)
            print(f"Successfully loaded params from {loaded_from}: {params_to_use}")
        except Exception as e:
            print(
                f"Warning: Could not load general hyperparameters from {GENERAL_HYPERPARAMS_FILE_PATH}: {e}"
            )
            params_to_use = None

    if params_to_use is None:
        print(
            f"Info: No model metadata or hyperparameter file found. Model will use fallback defaults for architecture."
        )
        return None

    # Standardize keys if necessary
    if "dropout" in params_to_use and "dropout_rate" not in params_to_use:
        params_to_use["dropout_rate"] = params_to_use["dropout"]
    if (
        "lstm_bidir" in params_to_use and "lstm_bidirectional" not in params_to_use
    ):  # common in meta.json
        params_to_use["lstm_bidirectional"] = params_to_use["lstm_bidir"]

    return params_to_use


print(f"--- EU Predictor Configuration Initialized ---")
print(f"Device: {DEVICE}")
print(
    f"MAIN_PROJECT_ROOT (for flightChainClassifier): {MAIN_PROJECT_ROOT}"
)  # For debugging import paths
print(f"EU Raw Data Dir: {EU_RAW_DATA_DIR}")
print(f"Pre-trained Model Path: {MODEL_PATH_FOR_EU_PREDICTION}")
print(f"Model Meta JSON Path (priority for arch params): {MODEL_META_JSON_PATH}")
print(
    f"General Hyperparams Path (fallback for arch params): {GENERAL_HYPERPARAMS_FILE_PATH}"
)
print(
    f"Original Data Stats (for scaling & features): {DATA_STATS_PATH_FOR_EU_PREDICTION}"
)
print(
    f"EU Column Mapping (Internal -> CSV Actual): {json.dumps(EU_COLUMN_MAPPING_RAW, indent=2)}"
)
print(f"Target Column (Internal Name for Processing): {TARGET_COL_INTERNAL_NAME}")
csv_target_name = EU_COLUMN_MAPPING_RAW.get(TARGET_COL_INTERNAL_NAME)
print(
    f"Target Column (Actual CSV Name based on mapping): {csv_target_name if csv_target_name else 'NOT MAPPED!'}"
)
print(f"Status Column in CSV (for Cancelled/Diverted): {EU_STATUS_COL_CSV_NAME}")
print(f"-------------------------------------------")
