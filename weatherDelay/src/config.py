# FILE: src/config.py
# --------------------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Project Root ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data Paths ---
DATA_DIR = BASE_DIR / "data"

# File pairs for processing (Assuming these are the files to process)
# "top10_us_airport_weather_20220901_20220930.csv",
# "top10_us_airport_weather_20221201_20221231.csv",
WEATHER_FILES = [
    "top10_us_airport_weather_20220301_20220331.csv",
    "top10_us_airport_weather_20220601_20220630.csv",
]

# "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_9.csv",
# "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_12.csv",
FLIGHT_FILES = [
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_3.csv",
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_6.csv",
]

# --- Results Paths ---
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Run Configuration ---
PIPELINE_SUFFIX = "weather_pred_v1" # Suffix for output files/plots for this run
MODEL_TYPES_TO_RUN = ['LGBM'] # Models to train ('LGBM', 'XGB', 'CatBoost')
RUN_UNBALANCED = True
RUN_BALANCED = True
USE_BALANCING_WEIGHTS = True # Whether the "balanced" run actually uses weights
TUNE_LGBM = True # Whether to run hyperparameter tuning for LGBM

# --- Core Variables ---
TARGET_VARIABLE = "WeatherDelay"
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2

# --- Data Loading/Merging Configuration ---
FLIGHT_COLS_TO_LOAD = [
    'FlightDate', 'CRSDepTime', 'CRSArrTime', TARGET_VARIABLE,
    'Month', 'DayofMonth', 'DayOfWeek', 'Distance',
    'Reporting_Airline', 'Origin', 'Dest'
]
WEATHER_FEATURES_BASE = [
    'temp_c', 'feels_like_c', 'dew_point_c', 'humidity_pct', 'pressure_mb',
    'cloud_cover_pct', 'visibility_m', 'wind_speed_kmph', 'wind_gust_kmph',
    'wind_dir_deg', 'precip_mm', 'total_daily_snow_cm', 'weather_desc'
]
LAG_HOURS = [1, 3] # Weather lag features to generate
WEATHER_FEATURES_TO_LAG = [f for f in WEATHER_FEATURES_BASE if f != 'weather_desc']

# --- Feature Engineering Configuration ---
CATEGORICAL_FEATURES_BASE = ['Reporting_Airline', 'Origin', 'Dest']
CYCLICAL_FEATURES_BASE = ['CRSDepHour', 'Month', 'DayOfWeek'] # Wind direction handled separately
WEATHER_DESC_KEYWORDS = ['thunder', 'snow', 'ice', 'fog', 'heavy rain', 'freezing', 'mist', 'drizzle', 'hail', 'sleet']
LOW_VIS_THRESHOLD_M = 1600
HIGH_WIND_GUST_THRESHOLD_KMPH = 55 # Approx 30 knots
FREEZING_TEMP_LOW = -2
FREEZING_TEMP_HIGH = 2
TREND_FEATURES_BASE = ['pressure_mb', 'temp_c', 'visibility_m']
INTERACTION_FEATURE_PAIRS = [
    ('wind_speed_kmph', 'precip_mm', 'multiply'),
    ('visibility_m', 'wind_gust_kmph', 'multiply'),
    ('is_high_gust', 'is_freezing_precip', 'flag_and'), # Assumes flags are created first
    ('is_low_vis', 'is_high_gust', 'flag_and')         # Assumes flags are created first
]
# Placeholder for dynamically generated feature names (populated in data_preprocessing.py)
DYNAMIC_FEATURE_LISTS = {
    'WEATHER_DESC': [], 'CYCLICAL': [], 'THRESHOLD': [], 'TREND': [], 'INTERACTION': []
}
FINAL_MODEL_FEATURES = [] # Populated after preprocessing

# --- Preprocessing Configuration ---
DROP_HIGH_MISSING_COLUMNS = False # If True, drop cols with > 15% NaNs, else impute
MAX_CATEGORIES_OHE = 50 # Max categories per feature for OHE (Origin, Dest)
MAX_CATEGORIES_AIRLINE_OHE = 20 # Max categories for Reporting_Airline
CHUNK_SIZE_PREPROCESSING = 400000 # Process in chunks if > this many rows

# --- Modeling Parameters ---
LGBM_PARAMS = {'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1} # Base params, tuning overrides
XGB_PARAMS = {'random_state': RANDOM_STATE, 'n_jobs': -1, 'objective': 'reg:squarederror'}
CATBOOST_PARAMS = {'random_state': RANDOM_STATE, 'verbose': 0}

# Balancing Strategy
BALANCING_METHOD = 'quantile'  # Options: 'binary', 'balanced', 'quantile'
NON_ZERO_WEIGHT_MULTIPLIER = 10 # Used only if BALANCING_METHOD == 'binary'

# Hyperparameter Tuning (LGBM)
TUNING_N_ITER = 15
TUNING_CV = 3
TUNING_SCORING = 'neg_mean_absolute_error'

# --- Evaluation Configuration ---
BEST_MODEL_METRIC = 'rmse' # Metric to select best model (lower is better)
CLASSIFICATION_THRESHOLDS = [0, 5, 10, 15, 30] # Thresholds for classification proxy metrics
PLOT_SAMPLE_SIZE = 5000 # Sample size for scatter plots if dataset is large
PLOT_DPI = 150 
PLOT_CMAP = 'viridis'

# --- Logging ---
def setup_logging():
    log_file = RESULTS_DIR / f"pipeline_{PIPELINE_SUFFIX}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging setup complete. Log file: {log_file}")

setup_logging() # Setup logging when config is imported
