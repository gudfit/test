# src/data_preprocessing/feature_engineering.py
import pandas as pd
import numpy as np
import logging
import re # For keyword matching

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- format_time_str function (no changes) ---
def format_time_str(time_val):
    if pd.isna(time_val): return None
    try:
        time_str = str(int(time_val)).zfill(4)
        if len(time_str) > 4:
             if time_str.startswith('24'): return '00' + time_str[2:]
             else: return None
        return time_str
    except ValueError: return None

# --- create_datetime_features function (no changes) ---
def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (Keep previous implementation - crucial for merge_data) ...
    logging.info("Creating datetime features...")
    df_copy = df.copy()
    df_copy['FlightDate'] = pd.to_datetime(df_copy['FlightDate'], errors='coerce')
    df_copy['CRSDepTime_str'] = df_copy['CRSDepTime'].apply(format_time_str)
    df_copy['CRSArrTime_str'] = df_copy['CRSArrTime'].apply(format_time_str) # Assuming merge needs this
    df_copy = df_copy.dropna(subset=['CRSDepTime_str']) # Need CRSDepTime for CRSDepHour
    dep_dt_str = df_copy['FlightDate'].dt.strftime('%Y-%m-%d') + ' ' + df_copy['CRSDepTime_str']
    # arr_dt_str logic might be needed if CRSArrTime is used for merge keys
    df_copy['DepartureDT'] = pd.to_datetime(dep_dt_str, format='%Y-%m-%d %H%M', errors='coerce')
    # ... (Handle ArrivalDT and midnight crossing if needed by merge) ...
    df_copy.dropna(subset=['DepartureDT'], inplace=True) # Need DepartureDT

    df_copy['Month'] = df_copy['DepartureDT'].dt.month
    df_copy['DayofMonth'] = df_copy['DepartureDT'].dt.day
    df_copy['DayOfWeek'] = df_copy['DepartureDT'].dt.dayofweek
    df_copy['CRSDepHour'] = df_copy['DepartureDT'].dt.hour # Keep this for cyclical engineering
    df_copy['DepartureDT_hour'] = df_copy['DepartureDT'].dt.round('h')
    # Handle ArrivalDT_hour rounding if used in merge_data
    if 'CRSArrTime' in df.columns:
        arr_dt_str = df_copy['FlightDate'].dt.strftime('%Y-%m-%d') + ' ' + df_copy['CRSArrTime_str'].fillna('0000')
        df_copy['ArrivalDT'] = pd.to_datetime(arr_dt_str, format='%Y-%m-%d %H%M', errors='coerce')
        # Apply midnight crossing if needed
        # ...
        df_copy['ArrivalDT_hour'] = df_copy['ArrivalDT'].dt.round('h')

    # Drop columns only needed for intermediate steps IF they aren't needed later
    df_copy.drop(columns=['CRSDepTime_str', 'CRSArrTime_str', 'DepartureDT', 'ArrivalDT'], inplace=True, errors='ignore')
    logging.info("Finished creating base datetime features.")
    return df_copy

# --- NEW Function ---

def create_weather_desc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary flag features based on keywords in weather_desc columns."""
    logging.info("Creating weather description keyword features...")
    df_copy = df.copy()
    keywords = config.WEATHER_DESC_KEYWORDS
    new_cols = []
    # Define all possible suffixes including base, origin/dest, and lags
    suffixes = ['_origin', '_dest'] + \
               [f'_origin_lag{l}h' for l in config.LAG_HOURS] + \
               [f'_dest_lag{l}h' for l in config.LAG_HOURS]

    for suffix in suffixes:
        desc_col = f'weather_desc{suffix}'
        if desc_col not in df_copy.columns:
            # logging.debug(f"Weather description column '{desc_col}' not found. Skipping.")
            continue # Silently skip if column doesn't exist after merge

        logging.debug(f"Processing weather desc col: {desc_col}")
        desc_series = df_copy[desc_col].fillna('').astype(str).str.lower()
        # Create a clean prefix for the new feature names
        location_prefix = suffix.replace("_lag", "lag") # e.g., _origin, _dest, lag1h_origin

        for keyword in keywords:
            feature_name = f'wd{location_prefix}_{keyword.replace(" ", "_")}'
            df_copy[feature_name] = desc_series.str.contains(keyword, case=False, na=False).astype(int)
            new_cols.append(feature_name)
            # logging.debug(f"Created feature '{feature_name}'")

    config.WEATHER_DESC_FEATURES_ALL = list(set(new_cols)) # Update global list
    logging.info(f"Finished creating {len(config.WEATHER_DESC_FEATURES_ALL)} weather description features.")
    return df_copy


# --- Cyclical Features ---
def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates sine/cosine features for cyclical data."""
    logging.info("Creating cyclical features...")
    df_copy = df.copy()
    new_cols = []
    # Add base time features first
    time_features = {'CRSDepHour': 24.0, 'Month': 12.0, 'DayOfWeek': 7.0}
    for feat, period in time_features.items():
        if feat in df_copy.columns:
            rad = 2 * np.pi * df_copy[feat]/period
            df_copy[f'{feat}_sin'] = np.sin(rad)
            df_copy[f'{feat}_cos'] = np.cos(rad)
            new_cols.extend([f'{feat}_sin', f'{feat}_cos'])
            logging.debug(f"Created cyclical features for {feat}.")

    # Wind Direction (Origin, Dest, Lags)
    suffixes = ['_origin', '_dest'] + \
               [f'_origin_lag{l}h' for l in config.LAG_HOURS] + \
               [f'_dest_lag{l}h' for l in config.LAG_HOURS]
    for suffix in suffixes:
        col = f'wind_dir_deg{suffix}'
        if col in df_copy.columns:
            rad_col = np.radians(df_copy[col].fillna(0)) # Fill NaNs with 0 degrees
            sin_col_name = f'wind_dir{suffix}_sin'
            cos_col_name = f'wind_dir{suffix}_cos'
            df_copy[sin_col_name] = np.sin(rad_col)
            df_copy[cos_col_name] = np.cos(rad_col)
            new_cols.extend([sin_col_name, cos_col_name])
            logging.debug(f"Created cyclical features for {col}.")

    config.CYCLICAL_FEATURES_ALL = list(set(new_cols))
    logging.info(f"Finished creating {len(config.CYCLICAL_FEATURES_ALL)} cyclical features.")
    return df_copy

# --- Threshold Features ---
def create_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary flags based on weather thresholds."""
    logging.info("Creating threshold-based weather features...")
    df_copy = df.copy()
    new_cols = []
    suffixes = ['_origin', '_dest'] + \
               [f'_origin_lag{l}h' for l in config.LAG_HOURS] + \
               [f'_dest_lag{l}h' for l in config.LAG_HOURS]

    for suffix in suffixes:
        # Low Visibility
        vis_col = f'visibility_m{suffix}'
        if vis_col in df_copy.columns:
            flag_name = f'is_low_vis{suffix}'
            df_copy[flag_name] = (df_copy[vis_col].fillna(99999) < config.LOW_VIS_THRESHOLD_M).astype(int)
            new_cols.append(flag_name)

        # High Wind Gust
        gust_col = f'wind_gust_kmph{suffix}'
        if gust_col in df_copy.columns:
            flag_name = f'is_high_gust{suffix}'
            df_copy[flag_name] = (df_copy[gust_col].fillna(0) > config.HIGH_WIND_GUST_THRESHOLD_KMPH).astype(int)
            new_cols.append(flag_name)

        # Freezing Precip (potential)
        temp_col = f'temp_c{suffix}'
        precip_col = f'precip_mm{suffix}'
        if temp_col in df_copy.columns and precip_col in df_copy.columns:
            flag_name = f'is_freezing_precip{suffix}'
            df_copy[flag_name] = (
                (df_copy[precip_col].fillna(0) > 0) &
                (df_copy[temp_col].fillna(99) >= config.FREEZING_TEMP_LOW) &
                (df_copy[temp_col].fillna(99) <= config.FREEZING_TEMP_HIGH)
            ).astype(int)
            new_cols.append(flag_name)

    config.THRESHOLD_FEATURES_ALL = list(set(new_cols))
    logging.info(f"Finished creating {len(config.THRESHOLD_FEATURES_ALL)} threshold features.")
    return df_copy

# --- Trend Features ---
def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates change in weather features over lag intervals."""
    logging.info("Creating weather trend features...")
    df_copy = df.copy()
    new_cols = []
    suffixes = ['_origin', '_dest']

    for feat_base in config.TREND_FEATURES_BASE:
        for suffix in suffixes:
            current_col = f'{feat_base}{suffix}'
            if current_col not in df_copy.columns: continue

            for lag in config.LAG_HOURS:
                lag_col = f'{feat_base}{suffix}_lag{lag}h'
                if lag_col in df_copy.columns:
                    trend_col_name = f'{feat_base}{suffix}_trend{lag}h'
                    df_copy[trend_col_name] = (df_copy[current_col] - df_copy[lag_col]).fillna(0)
                    new_cols.append(trend_col_name)
                    logging.debug(f"Created trend feature '{trend_col_name}'.")

    config.TREND_FEATURES_ALL = list(set(new_cols))
    logging.info(f"Finished creating {len(config.TREND_FEATURES_ALL)} trend features.")
    return df_copy

# --- Interaction Features ---
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates interaction terms based on config."""
    logging.info("Creating interaction features...")
    df_copy = df.copy()
    new_cols = []
    suffixes = ['_origin', '_dest'] # Apply interactions for T=0 weather

    for col1_base, col2_base, type in config.INTERACTION_FEATURE_PAIRS:
        for suffix in suffixes:
            col1 = f"{col1_base}{suffix}"
            col2 = f"{col2_base}{suffix}"
            interact_col = f'{col1_base}_x_{col2_base}{suffix}'

            if col1 in df_copy.columns and col2 in df_copy.columns:
                if type == 'multiply':
                    df_copy[interact_col] = (df_copy[col1].fillna(0) * df_copy[col2].fillna(0)).fillna(0)
                    new_cols.append(interact_col)
                    logging.debug(f"Created interaction feature '{interact_col}'.")
                # Add 'flag_and' logic if needed, ensuring flag columns exist first
                elif type == 'flag_and':
                     # Example assumes flag columns were already created (e.g., is_high_gust_origin)
                     if col1 in config.THRESHOLD_FEATURES_ALL and col2 in config.THRESHOLD_FEATURES_ALL:
                         df_copy[interact_col] = (df_copy[col1] * df_copy[col2]).astype(int)
                         new_cols.append(interact_col)
                         logging.debug(f"Created interaction feature '{interact_col}'.")
                     else:
                         logging.warning(f"Cannot create flag interaction '{interact_col}', base flags not found.")

            else:
                logging.warning(f"Cannot create interaction '{interact_col}', base columns missing.")


    config.INTERACTION_FEATURES_ALL = list(set(new_cols))
    logging.info(f"Finished creating {len(config.INTERACTION_FEATURES_ALL)} interaction features.")
    return df_copy

# --- Main Engineering Function ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all feature engineering steps and updates final feature list in config."""
    logging.info("--- Starting Feature Engineering ---")
    df_copy = df.copy() # Work on a copy

    # Apply individual engineering steps
    df_copy = create_weather_desc_features(df_copy)
    df_copy = create_cyclical_features(df_copy)
    df_copy = create_threshold_features(df_copy)
    df_copy = create_trend_features(df_copy)
    df_copy = create_interaction_features(df_copy) # Requires update to use config list

    # --- Dynamically Determine Final Features ---
    logging.info("Determining final feature set...")
    all_current_columns = list(df_copy.columns)

    # Define features to drop (check if they exist before trying to drop)
    features_to_drop = [f for f in config.FEATURES_TO_DROP_POST_ENG if f in all_current_columns]
    logging.debug(f"Features marked for dropping: {features_to_drop}")

    # Final model features are all current columns EXCEPT target and those marked for dropping
    final_model_features = [
        f for f in all_current_columns
        if f != config.TARGET_VARIABLE and f not in features_to_drop
    ]

    # Ensure base categoricals required for OHE later are still present
    missing_base_cats = [f for f in config.CATEGORICAL_FEATURES_BASE if f not in final_model_features and f in df.columns]
    if missing_base_cats:
        logging.warning(f"Base categorical features were dropped unexpectedly during engineering: {missing_base_cats}. Re-adding them.")
        final_model_features.extend(missing_base_cats)
        final_model_features = list(set(final_model_features)) # Ensure uniqueness

    # Update config list (this will be used by preprocessing)
    config.MODEL_FEATURES = final_model_features
    logging.info(f"Total features determined after engineering: {len(config.MODEL_FEATURES)}")
    logging.debug(f"First few features determined: {config.MODEL_FEATURES[:20]}") # Log more features

    logging.info("--- Finished Feature Engineering ---")
    return df_copy
