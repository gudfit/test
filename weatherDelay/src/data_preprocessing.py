# FILE: src/data_preprocessing.py
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import logging
import re
import gc
from pathlib import Path
from datetime import timedelta

from src import config # Import configurations
from src import utils # Import utilities like reduce_mem_usage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# === Data Loading and Merging ===

def _format_time_str(time_val):
    """Helper to format time strings (HHMM)."""
    if pd.isna(time_val): return None
    try:
        time_str = str(int(time_val)).zfill(4)
        if len(time_str) > 4:
             return '00' + time_str[2:] if time_str.startswith('24') else None
        return time_str
    except ValueError: return None

def load_and_merge_single_pair(weather_filepath: Path, flight_filepath: Path) -> pd.DataFrame:
    """Loads one pair of weather/flight files, merges, adds lags."""
    logging.debug(f"Processing pair: {weather_filepath.name} & {flight_filepath.name}")
    try:
        # --- Load Weather ---
        if not weather_filepath.exists():
            logging.error(f"Weather file not found: {weather_filepath}")
            return pd.DataFrame()
        weather_cols_needed = list(set(config.WEATHER_FEATURES_BASE + ['timestamp', 'airport_code_iata']))
        weather = pd.read_csv(weather_filepath, usecols=lambda c: c in weather_cols_needed)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather['timestamp_hour'] = weather['timestamp'].dt.floor('h')
        weather['airport_code_iata'] = weather['airport_code_iata'].str.strip().str.upper()

        # Filter to necessary cols and remove original timestamp
        weather = weather[[c for c in weather.columns if c != 'timestamp']].copy()

        # --- Create Lagged Weather ---
        weather_for_merge = weather.copy()
        for lag_hours in config.LAG_HOURS:
            lag_weather = weather.copy()
            # Adjust timestamp FOR MERGING KEY
            lag_weather['timestamp_hour'] = lag_weather['timestamp_hour'] + pd.Timedelta(hours=lag_hours)
            # Rename columns to indicate lag
            lag_cols_rename = {col: f"{col}_lag{lag_hours}h"
                               for col in config.WEATHER_FEATURES_TO_LAG if col in lag_weather.columns}
            lag_weather.rename(columns=lag_cols_rename, inplace=True)
            # Select only needed columns for the merge
            cols_to_merge = ['airport_code_iata', 'timestamp_hour'] + list(lag_cols_rename.values())
            # Merge lagged features onto the main weather df
            weather_for_merge = pd.merge(
                weather_for_merge,
                lag_weather[cols_to_merge],
                on=['airport_code_iata', 'timestamp_hour'],
                how='left'
            )
            del lag_weather # Free memory
            gc.collect()

        # --- Load Flight ---
        if not flight_filepath.exists():
            logging.error(f"Flight file not found: {flight_filepath}")
            return pd.DataFrame()
        # Load only potentially relevant flight columns specified in config
        flight = pd.read_csv(flight_filepath, low_memory=False,
                             usecols=lambda c: c in config.FLIGHT_COLS_TO_LOAD or 'Unnamed' in c)
        flight = flight.loc[:, ~flight.columns.str.contains('^Unnamed')]
        flight = flight.dropna(subset=['CRSDepTime', 'Origin', 'Dest']).copy() # Drop essential NaNs early
        flight['FlightDate'] = pd.to_datetime(flight['FlightDate'])
        flight['CRSDepTime_str'] = flight['CRSDepTime'].apply(_format_time_str)
        flight = flight.dropna(subset=['CRSDepTime_str'])

        # Create precise departure timestamp for merging
        dep_dt_str = flight['FlightDate'].dt.strftime('%Y-%m-%d') + ' ' + flight['CRSDepTime_str']
        flight['DepartureDT'] = pd.to_datetime(dep_dt_str, format='%Y-%m-%d %H%M', errors='coerce')
        flight = flight.dropna(subset=['DepartureDT'])
        flight['timestamp_hour'] = flight['DepartureDT'].dt.floor('h') # Key for weather merge

        # Clean origin/dest for merge
        flight['Origin'] = flight['Origin'].str.strip().str.upper()
        flight['Dest'] = flight['Dest'].str.strip().str.upper()

        # --- Merge Flight with Weather (Origin) ---
        merged_df = pd.merge(
            flight,
            weather_for_merge,
            left_on=['Origin', 'timestamp_hour'],
            right_on=['airport_code_iata', 'timestamp_hour'],
            how='left',
            suffixes=('', '_duplicate') # Avoid duplicating timestamp_hour, airport_code
        )
        merged_df.drop(columns=[c for c in merged_df.columns if '_duplicate' in c or c == 'airport_code_iata'], inplace=True)
        # Rename origin weather columns
        origin_rename_map = {col: f"{col}_origin" for col in config.WEATHER_FEATURES_BASE if col in merged_df.columns}
        for lag in config.LAG_HOURS:
             origin_rename_map.update({f"{col}_lag{lag}h": f"{col}_origin_lag{lag}h"
                                       for col in config.WEATHER_FEATURES_TO_LAG if f"{col}_lag{lag}h" in merged_df.columns})
        merged_df.rename(columns=origin_rename_map, inplace=True)

        # --- Merge Flight with Weather (Destination) ---
        merged_df = pd.merge(
            merged_df,
            weather_for_merge,
            left_on=['Dest', 'timestamp_hour'],
            right_on=['airport_code_iata', 'timestamp_hour'],
            how='left',
            suffixes=('', '_duplicate') # Avoid duplicating timestamp_hour, airport_code
        )
        merged_df.drop(columns=[c for c in merged_df.columns if '_duplicate' in c or c == 'airport_code_iata'], inplace=True)
        # Rename dest weather columns (ensure not overwriting origin cols)
        dest_rename_map = {col: f"{col}_dest" for col in config.WEATHER_FEATURES_BASE
                          if col in merged_df.columns and not col.endswith('_origin')}
        for lag in config.LAG_HOURS:
             dest_rename_map.update({f"{col}_lag{lag}h": f"{col}_dest_lag{lag}h"
                                     for col in config.WEATHER_FEATURES_TO_LAG
                                     if f"{col}_lag{lag}h" in merged_df.columns and not f"{col}_lag{lag}h".endswith('_origin')})
        merged_df.rename(columns=dest_rename_map, inplace=True)

        # --- Final Cleanup ---
        # Drop intermediate cols
        merged_df.drop(columns=['CRSDepTime_str', 'DepartureDT', 'timestamp_hour'], inplace=True, errors='ignore')
        # Convert categoricals early
        for col in config.CATEGORICAL_FEATURES_BASE:
             if col in merged_df.columns:
                 merged_df[col] = merged_df[col].astype('category')

        logging.debug(f"Merge complete for pair. Shape: {merged_df.shape}")
        return merged_df

    except Exception as e:
        logging.error(f"Error merging pair {flight_filepath.name}/{weather_filepath.name}: {e}", exc_info=True)
        return pd.DataFrame()

def load_all_data() -> pd.DataFrame:
    """Loads and merges all specified flight and weather file pairs."""
    logging.info("Loading and merging all data file pairs...")
    all_merged_list = []
    file_pairs = zip(config.WEATHER_FILES, config.FLIGHT_FILES)

    for weather_file, flight_file in file_pairs:
        weather_fp = config.DATA_DIR / weather_file
        flight_fp = config.DATA_DIR / flight_file
        merged_df = load_and_merge_single_pair(weather_fp, flight_fp)
        if not merged_df.empty:
            all_merged_list.append(merged_df)
        gc.collect() # Collect garbage after each pair

    if not all_merged_list:
        logging.error("No data loaded from any file pairs.")
        return pd.DataFrame()

    combined_df = pd.concat(all_merged_list, ignore_index=True)
    logging.info(f"Combined DataFrame shape after merging all pairs: {combined_df.shape}")
    del all_merged_list
    gc.collect()
    return combined_df


# === Feature Engineering ===

def _create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates basic time features needed for other steps."""
    if 'FlightDate' not in df.columns or 'CRSDepTime' not in df.columns:
        logging.error("Missing FlightDate or CRSDepTime for datetime feature creation.")
        return df

    df_copy = df.copy()
    try:
        df_copy['FlightDate'] = pd.to_datetime(df_copy['FlightDate'])
        df_copy['CRSDepTime_str'] = df_copy['CRSDepTime'].apply(_format_time_str)
        df_copy = df_copy.dropna(subset=['CRSDepTime_str'])

        dep_dt_str = df_copy['FlightDate'].dt.strftime('%Y-%m-%d') + ' ' + df_copy['CRSDepTime_str']
        df_copy['DepartureDT'] = pd.to_datetime(dep_dt_str, format='%Y-%m-%d %H%M', errors='coerce')
        df_copy = df_copy.dropna(subset=['DepartureDT'])

        # Extract components needed later
        df_copy['Month'] = df_copy['DepartureDT'].dt.month
        df_copy['DayofMonth'] = df_copy['DepartureDT'].dt.day # Keep if needed
        df_copy['DayOfWeek'] = df_copy['DepartureDT'].dt.dayofweek
        df_copy['CRSDepHour'] = df_copy['DepartureDT'].dt.hour

        # Drop intermediate columns
        df_copy.drop(columns=['CRSDepTime_str', 'DepartureDT'], inplace=True, errors='ignore')
        logging.debug("Created base datetime features (Month, DayOfWeek, CRSDepHour).")

    except Exception as e:
        logging.error(f"Error creating datetime features: {e}")
        return df # Return original df on error
    return df_copy

def _create_weather_desc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary flags from weather_desc keywords."""
    logging.debug("Creating weather description keyword features...")
    df_copy = df.copy()
    keywords = config.WEATHER_DESC_KEYWORDS
    new_cols = []
    suffixes = ['_origin', '_dest'] + \
               [f'_origin_lag{l}h' for l in config.LAG_HOURS] + \
               [f'_dest_lag{l}h' for l in config.LAG_HOURS]

    for suffix in suffixes:
        desc_col = f'weather_desc{suffix}'
        if desc_col in df_copy.columns:
            logging.debug(f"Processing weather desc col: {desc_col}")
            desc_series = df_copy[desc_col].fillna('').astype(str).str.lower()
            loc_prefix = suffix.replace("_lag", "lag") # e.g., _origin, lag1h_origin
            for keyword in keywords:
                feature_name = f'wd{loc_prefix}_{keyword.replace(" ", "_")}'
                df_copy[feature_name] = desc_series.str.contains(keyword, case=False, na=False).astype(np.int8)
                new_cols.append(feature_name)
            # Drop the original weather_desc column after processing
            df_copy.drop(columns=[desc_col], inplace=True)

    config.DYNAMIC_FEATURE_LISTS['WEATHER_DESC'] = list(set(new_cols))
    logging.debug(f"Finished creating {len(config.DYNAMIC_FEATURE_LISTS['WEATHER_DESC'])} weather desc features.")
    return df_copy

def _create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates sine/cosine features for cyclical data."""
    logging.debug("Creating cyclical features...")
    df_copy = df.copy()
    new_cols = []

    # Time features
    time_features_map = {'CRSDepHour': 24.0, 'Month': 12.0, 'DayOfWeek': 7.0}
    for feat, period in time_features_map.items():
        if feat in df_copy.columns:
            rad = (2 * np.pi * df_copy[feat]) / period
            df_copy[f'{feat}_sin'] = np.sin(rad).astype(np.float32)
            df_copy[f'{feat}_cos'] = np.cos(rad).astype(np.float32)
            new_cols.extend([f'{feat}_sin', f'{feat}_cos'])
            df_copy.drop(columns=[feat], inplace=True) # Drop original

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
            df_copy[sin_col_name] = np.sin(rad_col).astype(np.float32)
            df_copy[cos_col_name] = np.cos(rad_col).astype(np.float32)
            new_cols.extend([sin_col_name, cos_col_name])
            df_copy.drop(columns=[col], inplace=True) # Drop original

    config.DYNAMIC_FEATURE_LISTS['CYCLICAL'] = list(set(new_cols))
    logging.debug(f"Finished creating {len(config.DYNAMIC_FEATURE_LISTS['CYCLICAL'])} cyclical features.")
    return df_copy

def _create_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary flags based on weather thresholds."""
    logging.debug("Creating threshold-based weather features...")
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
            df_copy[flag_name] = (df_copy[vis_col].fillna(99999) < config.LOW_VIS_THRESHOLD_M).astype(np.int8)
            new_cols.append(flag_name)

        # High Wind Gust
        gust_col = f'wind_gust_kmph{suffix}'
        if gust_col in df_copy.columns:
            flag_name = f'is_high_gust{suffix}'
            df_copy[flag_name] = (df_copy[gust_col].fillna(0) > config.HIGH_WIND_GUST_THRESHOLD_KMPH).astype(np.int8)
            new_cols.append(flag_name)

        # Freezing Precip (potential)
        temp_col = f'temp_c{suffix}'
        precip_col = f'precip_mm{suffix}'
        if temp_col in df_copy.columns and precip_col in df_copy.columns:
            flag_name = f'is_freezing_precip{suffix}'
            df_copy[flag_name] = ((df_copy[precip_col].fillna(0) > 0) &
                                 (df_copy[temp_col].fillna(99) >= config.FREEZING_TEMP_LOW) &
                                 (df_copy[temp_col].fillna(99) <= config.FREEZING_TEMP_HIGH)
                                 ).astype(np.int8)
            new_cols.append(flag_name)

    config.DYNAMIC_FEATURE_LISTS['THRESHOLD'] = list(set(new_cols))
    logging.debug(f"Finished creating {len(config.DYNAMIC_FEATURE_LISTS['THRESHOLD'])} threshold features.")
    return df_copy

def _create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates change in weather features over lag intervals."""
    logging.debug("Creating weather trend features...")
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
                    # Ensure both columns are numeric before subtracting
                    df_copy[current_col] = pd.to_numeric(df_copy[current_col], errors='coerce')
                    df_copy[lag_col] = pd.to_numeric(df_copy[lag_col], errors='coerce')
                    df_copy[trend_col_name] = (df_copy[current_col] - df_copy[lag_col]).fillna(0).astype(np.float32)
                    new_cols.append(trend_col_name)

    config.DYNAMIC_FEATURE_LISTS['TREND'] = list(set(new_cols))
    logging.debug(f"Finished creating {len(config.DYNAMIC_FEATURE_LISTS['TREND'])} trend features.")
    return df_copy

def _create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates interaction terms based on config."""
    logging.debug("Creating interaction features...")
    df_copy = df.copy()
    new_cols = []
    suffixes = ['_origin', '_dest']

    for col1_base, col2_base, type in config.INTERACTION_FEATURE_PAIRS:
        for suffix in suffixes:
            col1 = f"{col1_base}{suffix}"
            col2 = f"{col2_base}{suffix}"
            interact_col = f'{col1_base}_x_{col2_base}{suffix}'

            if col1 in df_copy.columns and col2 in df_copy.columns:
                try:
                    # Ensure numeric types for multiplication
                    col1_num = pd.to_numeric(df_copy[col1], errors='coerce').fillna(0)
                    col2_num = pd.to_numeric(df_copy[col2], errors='coerce').fillna(0)

                    if type == 'multiply':
                        df_copy[interact_col] = (col1_num * col2_num).astype(np.float32)
                        new_cols.append(interact_col)
                    elif type == 'flag_and':
                        # Assumes col1 and col2 are already 0/1 flags
                        df_copy[interact_col] = ((col1_num > 0) & (col2_num > 0)).astype(np.int8)
                        new_cols.append(interact_col)
                except Exception as e:
                     logging.warning(f"Could not create interaction '{interact_col}' for suffix '{suffix}': {e}")
            # else:
            #      logging.debug(f"Skipping interaction '{interact_col}', missing base columns {col1} or {col2}")

    config.DYNAMIC_FEATURE_LISTS['INTERACTION'] = list(set(new_cols))
    logging.debug(f"Finished creating {len(config.DYNAMIC_FEATURE_LISTS['INTERACTION'])} interaction features.")
    return df_copy

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all feature engineering steps sequentially."""
    logging.info("Starting feature engineering...")
    if df.empty:
        logging.warning("Input DataFrame is empty, skipping feature engineering.")
        return df

    start_shape = df.shape
    df = _create_datetime_features(df) # Needed first for cyclical
    df = _create_weather_desc_features(df)
    df = _create_cyclical_features(df)
    df = _create_threshold_features(df)
    df = _create_trend_features(df)
    df = _create_interaction_features(df)
    # df = utils.reduce_mem_usage(df) # Reduce memory after engineering
    gc.collect()
    logging.info(f"Feature engineering complete. Shape change: {start_shape} -> {df.shape}")
    return df


# === Preprocessing (Cleaning, Imputation, Encoding) ===

def _handle_missing_values(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Imputes or drops columns based on missing value percentage."""
    logging.debug("Handling missing values...")
    df_copy = df.copy()
    na_counts = df_copy[features].isna().sum()
    na_percentage = (na_counts / len(df_copy)) * 100
    cols_to_drop = []

    for col in features:
        if na_percentage[col] > 0:
            if na_percentage[col] > 15 and config.DROP_HIGH_MISSING_COLUMNS:
                cols_to_drop.append(col)
                logging.warning(f"Dropping column '{col}' due to high missing percentage: {na_percentage[col]:.2f}%")
            else:
                # Impute: Median for numeric, Mode for categorical/object
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    fill_value = df_copy[col].median()
                    # Check if median is NaN (can happen if all values are NaN)
                    if pd.isna(fill_value):
                         fill_value = 0 # Fallback to 0 if median is NaN
                         logging.warning(f"Median for column '{col}' is NaN, imputing with 0.")
                else:
                    # Calculate mode, handle empty or multi-modal cases
                    mode_val = df_copy[col].mode()
                    fill_value = mode_val[0] if not mode_val.empty else df_copy[col].dtype.type() # Use dtype default if no mode
                    if len(mode_val) > 1:
                        logging.debug(f"Column '{col}' has multiple modes, using first: {fill_value}")
                
                df_copy[col] = df_copy[col].fillna(fill_value)
                logging.debug(f"Imputed missing values in '{col}' with '{fill_value}' ({na_percentage[col]:.2f}% missing).")

    if cols_to_drop:
        df_copy.drop(columns=cols_to_drop, inplace=True)
    return df_copy

def _one_hot_encode(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Performs one-hot encoding with category limits."""
    logging.debug(f"Performing one-hot encoding on: {categorical_cols}")
    df_copy = df.copy()
    encoded_cols_all = []

    for col in categorical_cols:
        if col not in df_copy.columns:
            logging.warning(f"Categorical column '{col}' not found for OHE.")
            continue

        # Determine max categories based on column name
        if col == 'Reporting_Airline':
            max_cats = config.MAX_CATEGORIES_AIRLINE_OHE
        else: # Assume Origin/Dest
            max_cats = config.MAX_CATEGORIES_OHE

        # Get top categories
        top_categories = df_copy[col].value_counts().nlargest(max_cats).index.tolist()
        logging.debug(f"Encoding top {len(top_categories)} categories for '{col}'.")

        # Create 'Other' category for values not in top N
        df_copy[col] = df_copy[col].apply(lambda x: x if x in top_categories else 'Other')

        # Perform OHE
        dummies = pd.get_dummies(df_copy[col], prefix=col, prefix_sep='_', dummy_na=False, dtype=np.int8)
        encoded_cols_all.extend(dummies.columns.tolist())

        # Drop original and join dummies
        df_copy = df_copy.drop(columns=[col])
        df_copy = pd.concat([df_copy, dummies], axis=1)
        gc.collect()

    logging.debug(f"Created {len(encoded_cols_all)} OHE features.")
    return df_copy

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """Applies cleaning, feature engineering, and preprocessing steps."""
    logging.info("Starting full preprocessing pipeline...")
    if df.empty:
        logging.error("Input DataFrame is empty. Cannot preprocess.")
        return pd.DataFrame(), pd.Series(), []

    start_shape = df.shape

    # 1. Feature Engineering
    df = engineer_features(df)
    if df.empty: return pd.DataFrame(), pd.Series(), [] # Check if empty after FE

    # 2. Define initial feature list (all columns except target)
    if config.TARGET_VARIABLE not in df.columns:
         logging.error(f"Target variable '{config.TARGET_VARIABLE}' not found after feature engineering.")
         return pd.DataFrame(), pd.Series(), []

    current_features = [col for col in df.columns if col != config.TARGET_VARIABLE]

    # 3. Handle Target Variable (coerce to numeric, fillna, clip)
    df[config.TARGET_VARIABLE] = pd.to_numeric(df[config.TARGET_VARIABLE], errors='coerce').fillna(0)
    df[config.TARGET_VARIABLE] = df[config.TARGET_VARIABLE].clip(lower=0)
    logging.debug("Target variable cleaned.")

    # 4. Handle Missing Values (Impute or Drop based on config)
    df = _handle_missing_values(df, current_features)
    # Update feature list if columns were dropped
    current_features = [col for col in current_features if col in df.columns]
    if not current_features:
        logging.error("No features remaining after handling missing values.")
        return pd.DataFrame(), pd.Series(), []

    # 5. One-Hot Encode Categorical Features
    ohe_cols = [col for col in config.CATEGORICAL_FEATURES_BASE if col in df.columns]
    if ohe_cols:
        df = _one_hot_encode(df, ohe_cols)
    else:
        logging.warning("No base categorical features found for OHE.")

    # 6. Final Feature Selection (Select only numeric columns)
    y = df[config.TARGET_VARIABLE].astype(np.float32) # Ensure target is float
    X = df.drop(columns=[config.TARGET_VARIABLE])

    # Ensure all remaining columns are numeric
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    if non_numeric_cols:
        logging.warning(f"Dropping non-numeric columns remaining before final selection: {non_numeric_cols}")
        X = X[numeric_cols]

    final_feature_names = numeric_cols
    config.FINAL_MODEL_FEATURES = final_feature_names # Update config

    if X.empty or not final_feature_names:
        logging.error("Preprocessing resulted in empty features (X).")
        return pd.DataFrame(), pd.Series(), []

    # 7. Final Memory Reduction
    X = utils.reduce_mem_usage(X)
    gc.collect()

    logging.info(f"Preprocessing complete. Initial shape: {start_shape}, Final X shape: {X.shape}, y shape: {y.shape}")
    logging.info(f"Total features used for model: {len(final_feature_names)}")
    return X, y, final_feature_names


# === Main Processing Function with Chunking ===

def process_in_chunks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """Processes the dataframe in chunks if it's too large."""
    max_rows = config.CHUNK_SIZE_PREPROCESSING
    if len(df) <= max_rows:
        logging.info("Dataset size within limits, processing as a single chunk.")
        return preprocess_data(df)
    else:
        logging.info(f"Dataset too large ({len(df)} rows), processing in chunks of {max_rows}...")
        n_chunks = int(np.ceil(len(df) / max_rows))
        all_X = []
        all_y = []
        final_feature_names_list = []

        for i in range(n_chunks):
            start_idx = i * max_rows
            end_idx = min((i + 1) * max_rows, len(df))
            logging.info(f"--- Processing chunk {i+1}/{n_chunks} (rows {start_idx}-{end_idx}) ---")
            chunk_df = df.iloc[start_idx:end_idx].copy()
            X_chunk, y_chunk, features_chunk = preprocess_data(chunk_df)

            if not X_chunk.empty:
                all_X.append(X_chunk)
                all_y.append(y_chunk)
                if not final_feature_names_list: # Store feature names from the first valid chunk
                    final_feature_names_list = features_chunk
            del chunk_df, X_chunk, y_chunk # Free memory
            gc.collect()

        if not all_X:
            logging.error("No valid data after processing all chunks.")
            return pd.DataFrame(), pd.Series(), []

        # Combine results - ensure consistent columns
        logging.info("Combining results from all chunks...")
        final_X = pd.concat(all_X, ignore_index=True)
        final_y = pd.concat(all_y, ignore_index=True)
        del all_X, all_y # Free memory
        gc.collect()

        # Realign columns in case some chunks missed certain OHE features
        if final_feature_names_list:
            final_X = final_X.reindex(columns=final_feature_names_list, fill_value=0)
            logging.info(f"Realigned columns based on first chunk's features: {len(final_feature_names_list)}")
        else:
            logging.error("Could not determine final feature names from chunks.")
            return pd.DataFrame(), pd.Series(), []

        logging.info(f"Combined chunk processing complete. Final X shape: {final_X.shape}, y shape: {final_y.shape}")
        return final_X, final_y, final_feature_names_list
