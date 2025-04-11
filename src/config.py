# src/config.py
import os
from pathlib import Path
import numpy as np

# --- Project Root ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data Paths ---
DATA_DIR = BASE_DIR / "data"

# File pairs for processing
WEATHER_FILES = [
    "top10_us_airport_weather_20220301_20220331.csv",
    "top10_us_airport_weather_20220601_20220630.csv",
    "top10_us_airport_weather_20220901_20220930.csv",
    "top10_us_airport_weather_20221201_20221231.csv",
]
FLIGHT_FILES = [
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_3.csv",
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_6.csv",
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_9.csv",
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_12.csv",
]

# --- Results Paths ---
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Saving ---
MODEL_SUFFIX = "tuned_engfeat_lags_bal" # Suffix for logging
# Actual save names determined in main script now
BEST_UNBALANCED_MODEL_SAVE_NAME_TEMPLATE = "best_unbalanced_{model_name}.joblib"
BEST_BALANCED_MODEL_SAVE_NAME_TEMPLATE = "best_balanced_{model_name}.joblib"

# --- Core Variables ---
TARGET_VARIABLE = "WeatherDelay"
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2

# --- Missing Value Handling ---
# If True, columns with >15% missing values will be dropped
# If False, they will be imputed with median/mode
DROP_HIGH_MISSING_COLUMNS = False

# --- Memory Management Settings ---
# Maximum number of categories to keep for each categorical feature
MAX_CATEGORIES_DEST = 50       # For destination airports
MAX_CATEGORIES_ORIGIN = 50     # For origin airports  
MAX_CATEGORIES_AIRLINE = 20    # For airlines

# --- Base Feature Definitions ---
# Base schedule features from notebook logic
SCHEDULE_FEATURES = [
    'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepHour', 'Distance',
    'Reporting_Airline', 'Origin', 'Dest' # Base categoricals
]
# Base weather features from notebook logic
WEATHER_FEATURES_BASE = [
    'temp_c', 'feels_like_c', 'dew_point_c', 'humidity_pct', 'pressure_mb',
    'cloud_cover_pct', 'visibility_m', 'wind_speed_kmph', 'wind_gust_kmph',
    'wind_dir_deg', 'precip_mm', 'total_daily_snow_cm'
]
# Base features needed for loading weather files initially (for merge function)
NOTEBOOK_FEATURES_TO_LOAD_WEATHER = WEATHER_FEATURES_BASE + ['weather_desc']
# Base features needed for loading flight files initially
FLIGHT_COLS_TO_LOAD = list(set(['FlightDate', 'CRSDepTime', 'CRSArrTime', TARGET_VARIABLE] + SCHEDULE_FEATURES))

# --- Feature Engineering Parameters ---
# Lag Features
WEATHER_FEATURES_TO_LAG = WEATHER_FEATURES_BASE
LAG_HOURS = [1, 3]

# Weather Description Keywords
WEATHER_DESC_KEYWORDS = ['thunder', 'snow', 'ice', 'fog', 'heavy rain', 'freezing', 'mist', 'drizzle', 'hail', 'sleet']

# Cyclical Feature Bases
CYCLICAL_FEATURES_BASE = ['CRSDepHour', 'Month', 'DayOfWeek', 'wind_dir_deg']

# Threshold Feature Bases & Values
THRESHOLD_FEATURES_BASE = ['visibility_m', 'wind_gust_kmph', 'temp_c', 'precip_mm']
LOW_VIS_THRESHOLD_M = 1600
HIGH_WIND_GUST_THRESHOLD_KMPH = 55 # Approx 30 knots
FREEZING_TEMP_LOW = -2
FREEZING_TEMP_HIGH = 2

# Trend Feature Bases
TREND_FEATURES_BASE = ['pressure_mb', 'temp_c', 'visibility_m']

# Interaction Feature Pairs (Example - function in feature_engineering needs update to use this)
INTERACTION_FEATURE_PAIRS = [
    ('wind_speed_kmph', 'precip_mm', 'multiply'),
    ('visibility_m', 'wind_gust_kmph', 'multiply'),
    ('is_high_gust', 'is_freezing_precip', 'flag_and'),
    ('is_low_vis', 'is_high_gust', 'flag_and')
]

# Features to Drop *After* Engineering
FEATURES_TO_DROP_POST_ENG = list(set(
    # Drop original cyclical base features if transformed versions were created
    [f for f in CYCLICAL_FEATURES_BASE if f != 'wind_dir_deg'] + # Don't drop wind_dir_deg base name yet
    [f'{feat}_origin' for feat in CYCLICAL_FEATURES_BASE if feat != 'wind_dir_deg'] +
    [f'{feat}_dest' for feat in CYCLICAL_FEATURES_BASE if feat != 'wind_dir_deg'] +
    # Drop original wind_dir_deg everywhere (base + lags)
    [f'wind_dir_deg_{loc}{lag_suffix}' for loc in ['origin', 'dest'] for lag_suffix in [''] + [f'_lag{h}h' for h in LAG_HOURS]] +
    # Drop original weather_desc everywhere
    [f'weather_desc_{loc}{lag_suffix}' for loc in ['origin', 'dest'] for lag_suffix in [''] + [f'_lag{h}h' for h in LAG_HOURS]]
))

# --- Placeholders for dynamic feature lists (populated by feature_engineering.py) ---
# These lists are generated and used *within* the feature_engineering module primarily
# to construct the final MODEL_FEATURES list.
WEATHER_DESC_FEATURES_ALL = []
CYCLICAL_FEATURES_ALL = []
THRESHOLD_FEATURES_ALL = []
TREND_FEATURES_ALL = []
INTERACTION_FEATURES_ALL = []
MODEL_FEATURES = [] # THIS LIST IS THE IMPORTANT ONE - updated by engineer_features

# Base Categoricals (for OHE step in preprocessing)
CATEGORICAL_FEATURES_BASE = ['Reporting_Airline', 'Origin', 'Dest']

# --- Modeling Parameters ---
# Base model params (used if tuning is disabled or for non-LGBM models)
LGBM_PARAMS = {'n_estimators': 300, 'learning_rate': 0.05, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1}
XGB_PARAMS = {'n_estimators': 300, 'learning_rate': 0.05, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'objective': 'reg:squarederror'}
CATBOOST_PARAMS = {'iterations': 300, 'learning_rate': 0.05, 'random_state': RANDOM_STATE, 'verbose': 0}

# Balancing Strategy - Enhanced
USE_BALANCING = True # Control whether the "balanced" model run uses weights
NON_ZERO_WEIGHT_MULTIPLIER = 10
# New option: choose balancing method
BALANCING_METHOD = 'quantile'  # Options: 'binary', 'balanced', 'quantile'

# Hyperparameter Tuning Parameters
TUNING_ENABLED = True
TUNING_N_ITER = 15  # Number of random configurations to try
TUNING_CV = 3       # Cross-validation folds
TUNING_SCORING = 'neg_mean_absolute_error' # Optimize for MAE

# Evaluation Metric for Comparison (used in main.py to pick best)
BEST_MODEL_METRIC = 'rmse' # Lower is better
CLASSIFICATION_THRESHOLD = 0  # Above this is considered delay

# --- Plot Configuration ---
PLOT_DPI = 300  # Resolution for saved plots
PLOT_CMAP = 'viridis'  # Default colormap
SAVE_ADDITIONAL_PLOTS = True  # Whether to save additional analysis plots
