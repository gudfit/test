# FILE: src/data_processing/config.py

import pathlib
from datetime import datetime

# --- Project Structure ---
SCRIPT_DIR        = pathlib.Path(__file__).parent.resolve()
SRC_DIR           = SCRIPT_DIR.parent
FLIGHT_DELAY_DIR  = SRC_DIR.parent

DATA_DIR          = FLIGHT_DELAY_DIR / 'data'
ML_DATA_DIR       = FLIGHT_DELAY_DIR / 'mlData'
PROCESSED_DATA_DIR= ML_DATA_DIR / 'processed'
RESULTS_DIR       = FLIGHT_DELAY_DIR / 'results'
PREDICTIONS_DIR   = RESULTS_DIR / 'predictions'
MODELS_DIR        = RESULTS_DIR / 'models'

# --- Input Data Files ---
RAW_FLIGHT_FILES_PATTERN   = 'On_Time_Reporting_Carrier_On_Time_Performance*.csv'
AIRPORT_COORDS_FILE        = DATA_DIR / 'iata-icao.csv'

# --- Processed Data Files ---
MERGED_RAW_FLIGHTS         = PROCESSED_DATA_DIR / '0_merged_raw_flights.csv'
CLEANED_FLIGHTS            = PROCESSED_DATA_DIR / '1_cleaned_flights.csv'
REFRAMED_POINTS            = PROCESSED_DATA_DIR / '2_reframed_points.csv'
HISTORICAL_POINTS          = PROCESSED_DATA_DIR / '4_historical_points_with_pfd.csv'
INITIAL_FUTURE_FILE        = PROCESSED_DATA_DIR / '3a_future_points_initial.csv'
FUTURE_PREPARED            = PROCESSED_DATA_DIR / '5_future_points_prepared.csv'
FINAL_PREDICTIONS          = PREDICTIONS_DIR   / 'final_predictions.csv'
TRAINED_MODEL_FILE         = MODELS_DIR        / 'fdpp_ml_voting_model.joblib'

# --- Column Names ---
FLIGHT_DATE_COL            = 'FlightDate'
ORIGIN_COL                 = 'Origin'
DEST_COL                   = 'Dest'
CARRIER_CODE_COL           = 'Reporting_Airline'
TAIL_NUM_COL               = 'Tail_Number'
FLIGHT_NUM_COL             = 'Flight_Number_Reporting_Airline'
SCHED_DEP_TIME_COL         = 'CRSDepTime'
SCHED_ARR_TIME_COL         = 'CRSArrTime'
DEP_DELAY_COL              = 'DepDelay'
ARR_DELAY_COL              = 'ArrDelay'
CANCELLED_COL              = 'Cancelled'
DIVERTED_COL               = 'Diverted'

ORIENTATION_COL            = 'Orientation'
SCHED_DATETIME_COL         = 'Schedule_DateTime'
FLIGHT_DELAY_COL           = 'Flight_Delay'
FTD_COL                    = 'FTD'
PFD_COL                    = 'PFD'
PREDICTED_DELAY_COL        = 'Predicted_Delay'

# --- New: Delay‐to‐class mapping (for classification) ---
DELAY_THRESHOLDS           = [-float("inf"), 15, 60, 120, 240, float("inf")]
NUM_CLASSES                = 5  # labels 0 … 4

# --- FDPP‐ML Algorithm Parameters ---
PARTITION_DATETIME         = datetime(2022, 11, 30, 23, 59, 59)

# --- Model Features ---
FDPP_FEATURES              = [FTD_COL, PFD_COL]
CATEGORICAL_MODEL_FEATURES = [ORIGIN_COL, DEST_COL, CARRIER_CODE_COL]
TEMPORAL_MODEL_FEATURES    = ['Month', 'DayOfWeek', 'Hour']

# Ensure output dirs exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Config loaded: Output directories ensured.")
print(f"Using partition time: {PARTITION_DATETIME}")
