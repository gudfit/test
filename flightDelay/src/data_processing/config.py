# flightDelay/src/data_processing/config.py
import pathlib
from datetime import datetime

# --- Project Structure ---
# Get the directory of the current file (config.py)
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Get the parent directory (src/)
SRC_DIR = SCRIPT_DIR.parent
# Get the parent directory of src/ (flightDelay/)
FLIGHT_DELAY_DIR = SRC_DIR.parent
# Optional: Get the parent of flightDelay/ if needed (PROJECT_ROOT)
# PROJECT_ROOT = FLIGHT_DELAY_DIR.parent

# Define main directories relative to flightDelay/
DATA_DIR = FLIGHT_DELAY_DIR / 'data'
ML_DATA_DIR = FLIGHT_DELAY_DIR / 'mlData'
PROCESSED_DATA_DIR = ML_DATA_DIR / 'processed'
RESULTS_DIR = FLIGHT_DELAY_DIR / 'results'
PREDICTIONS_DIR = RESULTS_DIR / 'predictions'
MODELS_DIR = RESULTS_DIR / 'models'

# --- Input Data Files ---
RAW_FLIGHT_FILES_PATTERN = 'On_Time_Reporting_Carrier_On_Time_Performance*.csv'
# Ensure the pattern matches your actual file names precisely
# Example if files are like: On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_3.csv
# RAW_FLIGHT_FILES_PATTERN = 'On_Time_Reporting_Carrier_On_Time_Performance_(*)_*.csv' # Adjust as needed

AIRPORT_COORDS_FILE = DATA_DIR / 'iata-icao.csv' # Optional: Only used if orientation calculation is added later

# --- Processed Data Files ---
MERGED_RAW_FLIGHTS = PROCESSED_DATA_DIR / '0_merged_raw_flights.csv'
CLEANED_FLIGHTS = PROCESSED_DATA_DIR / '1_cleaned_flights.csv' # Optional intermediate save
REFRAMED_POINTS = PROCESSED_DATA_DIR / '2_reframed_points.csv'
# POINTS_WITH_FTD = PROCESSED_DATA_DIR / '3_points_with_ftd.csv' # Optional intermediate save
INITIAL_FUTURE_FILE = PROCESSED_DATA_DIR / '3a_future_points_initial.csv'
HISTORICAL_POINTS = PROCESSED_DATA_DIR / '4_historical_points_with_pfd.csv' # Training Data
FUTURE_PREPARED = PROCESSED_DATA_DIR / '5_future_points_prepared.csv' # Prediction Input Data
FINAL_PREDICTIONS = PREDICTIONS_DIR / 'final_predictions.csv'
TRAINED_MODEL_FILE = MODELS_DIR / 'fdpp_ml_voting_model.joblib'

# --- Column Names (Aligned with Table 1 and common BTS data headers) ---
# Input Columns from Raw BTS Data needed for FDPP-ML processing:
FLIGHT_DATE_COL = 'FlightDate'          # e.g., '2022-03-01' (Format may vary, parsed later)
ORIGIN_COL = 'Origin'                   # e.g., 'SEA' (Airport Code)
DEST_COL = 'Dest'                       # e.g., 'JFK' (Airport Code)
CARRIER_CODE_COL = 'Reporting_Airline'  # e.g., 'AA', 'DL' (Used for dtype spec & potential feature) - THIS WAS THE MISSING ATTRIBUTE FIX
TAIL_NUM_COL = 'Tail_Number'            # e.g., 'N12345' (Aircraft Identifier)
FLIGHT_NUM_COL = 'Flight_Number_Reporting_Airline' # e.g., '1234' (Flight Identifier)
SCHED_DEP_TIME_COL = 'CRSDepTime'       # e.g., '0830', '1400' (HHMM format, local time)
SCHED_ARR_TIME_COL = 'CRSArrTime'       # e.g., '1045', '2215' (HHMM format, local time)
DEP_DELAY_COL = 'DepDelay'              # e.g., -5.0, 15.0 (Departure delay in minutes) - Used for Dep Point Target
ARR_DELAY_COL = 'ArrDelay'              # e.g., -2.0, 25.0 (Arrival delay in minutes) - Used for Arr Point Target
CANCELLED_COL = 'Cancelled'             # e.g., 0.0 or 1.0 (Flight Cancellation Indicator)
DIVERTED_COL = 'Diverted'               # e.g., 0.0 or 1.0 (Flight Diversion Indicator)

# Intermediate & Output Columns generated during FDPP-ML processing:
ORIENTATION_COL = 'Orientation'         # 'Departure' or 'Arrival' marker for reframed points
SCHED_DATETIME_COL = 'Schedule_DateTime' # Combined Pandas Timestamp object for sorting/FTD calc
FLIGHT_DELAY_COL = 'Flight_Delay'       # Unified delay (DepDelay or ArrDelay) for the point - THIS IS THE MODEL TARGET
FTD_COL = 'FTD'                         # Feature: Flight Time Duration (minutes between points on path)
PFD_COL = 'PFD'                         # Feature: Previous Flight Delay (minutes, actual or predicted)
PREDICTED_DELAY_COL = 'Predicted_Delay' # Final output of the prediction process

# --- FDPP-ML Algorithm Parameters ---
# Define the 'current time' for partitioning historical vs future data
# Example: Use data up to end of Nov 2022 as historical, predict Dec 2022
# *** ADJUST THIS DATE BASED ON YOUR DATA RANGE AND PREDICTION GOAL ***
PARTITION_DATETIME = datetime(2022, 11, 30, 23, 59, 59) # End of November 2022

# --- Model Features ---
# Define the sets of raw features the model pipeline will expect as input
# (Preprocessing like encoding/scaling happens inside the pipeline)

# Core FDPP-ML numerical features:
FDPP_FEATURES = [FTD_COL, PFD_COL]

# Categorical features (potentially used by model after encoding):
# Note: Tail_Number and Flight_Number are usually identifiers, not direct features,
# unless treated specifically (e.g., embedding lookup). Keep them out for standard models.
CATEGORICAL_MODEL_FEATURES = [ORIGIN_COL, DEST_COL, CARRIER_CODE_COL]

# Temporal features derived from SCHED_DATETIME_COL during training/prediction:
TEMPORAL_MODEL_FEATURES = ['Month', 'DayOfWeek', 'Hour'] # Example derivations

# --- Ensure Output Directories Exist ---
# Create directories if they don't exist when the config is loaded
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Config loaded: Output directories ensured.")
print(f"Using partition time: {PARTITION_DATETIME}")
