# eu_flight_predictor/eu_chain_constructor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import sys
import os
import json
from tqdm import tqdm
import warnings
from datetime import datetime # Not strictly used directly, but good to have if needed

# Assuming eu_config is in the same directory
from eu_config import (
    MERGED_EU_DATA_FILE,
    EU_COLUMN_MAPPING_RAW,
    EU_DATETIME_COLS_NEED_HHMM_PROCESSING,
    EU_STATUS_COL_CSV_NAME,
    TARGET_COL_INTERNAL_NAME,
    CHAIN_LENGTH,
    MAX_GROUND_TIME_HOURS,
    MIN_TURNAROUND_MINUTES,
    DELAY_THRESHOLDS_CLASSIFICATION,
    NUM_CLASSES_CLASSIFICATION,
    RANDOM_STATE,
    PROCESSED_EU_CHAINS_DIR,
    EU_TEST_CHAINS_FILE,
    EU_TEST_LABELS_FILE,
    EU_DATA_STATS_FILE,
    load_original_data_stats_for_scaling,
)
# Ensure utils.datetime_helpers is accessible
# If utils is a direct subdir of eu_flight_predictor:
from utils.datetime_helpers import parse_hhmm_to_time, combine_date_time_objects
# If you get an import error for utils, you might need to adjust sys.path
# or ensure eu_flight_predictor/utils/__init__.py exists.

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def extract_hhmm_from_datetime_str(dt_str):
    """Extracts HHMM string from a full datetime string like 'YYYY-MM-DD HH:MM' or ISO."""
    if pd.isna(dt_str):
        return None
    try:
        # Pandas to_datetime is quite robust for various ISO and common formats
        dt_obj = pd.to_datetime(dt_str)
        return dt_obj.strftime("%H%M")
    except (ValueError, TypeError):
        # Fallback for potentially non-standard strings if pd.to_datetime fails
        str_val = str(dt_str)
        # Basic attempt to find a time-like part (e.g., "10:30", "1030", "T10:30:00Z")
        import re
        match = re.search(r'(\d{1,2})[: ]?(\d{2})(?:[: ]?(\d{2}))?', str_val)
        if match:
            h, m = match.group(1), match.group(2)
            return f"{int(h):02d}{int(m):02d}"

        # If still not found, try to extract all digits and see if it forms HHMM
        digits = "".join(filter(str.isdigit, str_val))
        # Look for sequences of 4 digits that could be time, prioritizing later occurrences
        for i in range(len(digits) - 3, -1, -1):
            potential_hhmm = digits[i:i+4]
            try:
                # Validate if it's a plausible HHMM
                h_int = int(potential_hhmm[:2])
                m_int = int(potential_hhmm[2:])
                if 0 <= h_int <= 23 and 0 <= m_int <= 59:
                    return potential_hhmm
            except ValueError:
                continue
        return None


def load_and_preprocess_eu_data(file_path):
    print(f"Loading merged EU data from {file_path}...")
    if not file_path.exists():
        sys.exit(f"Error: Merged EU data file not found at {file_path}")

    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df)} rows. Initial columns: {df.columns.tolist()}")

    # Rename columns based on EU_COLUMN_MAPPING_RAW
    # Internal names (model's expectation) are keys, CSV names are values
    rename_map = {
        csv_name: internal_name
        for internal_name, csv_name in EU_COLUMN_MAPPING_RAW.items()
        if csv_name is not None
        and csv_name in df.columns # Only map if CSV name exists in df and is not None in mapping
    }
    df.rename(columns=rename_map, inplace=True)
    print(f"Renamed columns. Current columns after rename: {df.columns.tolist()}")

    # HHMM Extraction
    # EU_DATETIME_COLS_NEED_HHMM_PROCESSING maps:
    # new_internal_hhmm_col_name : source_CSV_col_name_containing_full_datetime
    for (
        new_internal_hhmm_col,      # e.g., "CRSDepTime_hhmm"
        source_csv_col_original_name, # e.g., "scheduled_departure_utc"
    ) in EU_DATETIME_COLS_NEED_HHMM_PROCESSING.items():

        column_to_extract_from = None
        # Check if the original CSV column name exists directly in the DataFrame
        # This is the most likely case if it wasn't a target for renaming in EU_COLUMN_MAPPING_RAW
        if source_csv_col_original_name in df.columns:
            column_to_extract_from = source_csv_col_original_name
        else:
            # If not found directly, check if it was renamed
            # This means source_csv_col_original_name was a VALUE in EU_COLUMN_MAPPING_RAW
            # and got mapped to some internal_key.
            for internal_key, csv_val_in_map in EU_COLUMN_MAPPING_RAW.items():
                if csv_val_in_map == source_csv_col_original_name:
                    if internal_key in df.columns: # The new name (internal_key) should be in df
                        column_to_extract_from = internal_key
                        break
        
        if column_to_extract_from and column_to_extract_from in df.columns:
            print(
                f"Extracting HHMM from column '{column_to_extract_from}' (original CSV name: '{source_csv_col_original_name}') into new column '{new_internal_hhmm_col}'..."
            )
            df[new_internal_hhmm_col] = df[column_to_extract_from].apply(
                extract_hhmm_from_datetime_str
            )
            if df[new_internal_hhmm_col].isnull().all():
                print(f"  WARNING: New column '{new_internal_hhmm_col}' was created from '{column_to_extract_from}' but is ALL NaN. ")
                print(f"           Please check the format of data in '{column_to_extract_from}' and the 'extract_hhmm_from_datetime_str' function.")
                print(f"           Sample values from '{column_to_extract_from}': {df[column_to_extract_from].dropna().head().tolist()}")

            elif df[new_internal_hhmm_col].notnull().sum() < 0.1 * len(df) and len(df) > 0: # If less than 10% are valid
                 print(f"  WARNING: New column '{new_internal_hhmm_col}' from '{column_to_extract_from}' has very few non-NaN values ({df[new_internal_hhmm_col].notnull().sum()} out of {len(df)}).")
                 print(f"           Consider reviewing data quality or extraction logic.")

        else:
            print(
                f"WARNING: Source column for HHMM extraction (to create '{new_internal_hhmm_col}') "
                f"derived from CSV column '{source_csv_col_original_name}' could not be found in the DataFrame. Skipping this HHMM column."
            )
            print(f"         Attempted to find: '{source_csv_col_original_name}' or its potential renamed version.")
            print(f"         Available DataFrame columns: {df.columns.tolist()}")


    # Cancelled/Diverted Flags
    # EU_STATUS_COL_CSV_NAME is the name of the column in the CSV file (e.g., "status")
    # We need to check if this column exists in df *after* potential renames.
    # So, find what EU_STATUS_COL_CSV_NAME might have been renamed to, or use it directly.
    status_col_in_df = None
    if EU_STATUS_COL_CSV_NAME in df.columns:
        status_col_in_df = EU_STATUS_COL_CSV_NAME
    else:
        for internal_name, csv_name_in_map in EU_COLUMN_MAPPING_RAW.items():
            if csv_name_in_map == EU_STATUS_COL_CSV_NAME:
                if internal_name in df.columns:
                    status_col_in_df = internal_name
                    break
    
    if status_col_in_df:
        print(
            f"Processing status column '{status_col_in_df}' (from CSV '{EU_STATUS_COL_CSV_NAME}') to create 'Cancelled' and 'Diverted' features..."
        )
        df["Cancelled"] = df[status_col_in_df].apply(
            lambda x: 1.0 if isinstance(x, str) and x.lower() == "cancelled" else 0.0
        )
        df["Diverted"] = df[status_col_in_df].apply(
            lambda x: 1.0 if isinstance(x, str) and x.lower() == "diverted" else 0.0
        )
    else:
        print(
            f"Warning: Status CSV column '{EU_STATUS_COL_CSV_NAME}' (or its renamed version) not found in DataFrame. 'Cancelled' & 'Diverted' features set to 0."
        )
        print(f"Available columns: {df.columns.tolist()}")
        df["Cancelled"] = 0.0
        df["Diverted"] = 0.0

    # NaN drop and filter
    critical_cols_for_nan_drop = [
        "FlightDate",
        "Tail_Number",
        "CRSDepTime_hhmm", # These are the NEWLY CREATED HHMM columns
        "CRSArrTime_hhmm",
        TARGET_COL_INTERNAL_NAME,
        "Origin",
        "Dest",
        "DepTime_hhmm",    # These are the NEWLY CREATED HHMM columns
        "ArrTime_hhmm",
        "CRSElapsedTime",
        "Cancelled",      # These are the NEWLY CREATED flag columns
        "Diverted",
    ]

    actual_critical_for_drop = [col for col in critical_cols_for_nan_drop if col in df.columns]
    missing_for_drop = set(critical_cols_for_nan_drop) - set(actual_critical_for_drop)

    if missing_for_drop:
        print(f"Warning: Columns missing for NaN drop: {missing_for_drop}.")
        print(f"         Will attempt to drop NaNs from available critical columns: {actual_critical_for_drop}")
    
    if not all(hhmm_col in df.columns for hhmm_col in ["CRSDepTime_hhmm", "CRSArrTime_hhmm", "DepTime_hhmm", "ArrTime_hhmm"]):
        print("CRITICAL WARNING: Not all required HHMM columns (CRSDepTime_hhmm, CRSArrTime_hhmm, DepTime_hhmm, ArrTime_hhmm) were created.")
        print("                  This will likely lead to errors in datetime engineering. Please check HHMM extraction logic and source data.")

    if actual_critical_for_drop:
        print(f"Shape before NaN drop on critical columns: {df.shape}")
        df.dropna(subset=actual_critical_for_drop, inplace=True)
        print(f"Shape after NaN drop: {df.shape}")
    else:
        print("Warning: No critical columns (including HHMM) found to perform NaN drop. Data might be unsuitable.")


    if "Cancelled" in df.columns and "Diverted" in df.columns:
        df = df[(df["Cancelled"] != 1.0) & (df["Diverted"] != 1.0)]
    else:
        print("Error: 'Cancelled' or 'Diverted' columns not present for filtering, though they should have been created.")

    print(f"Data shape after cleaning (NaN drop and Cancelled/Diverted filter): {df.shape}")
    if df.empty:
        sys.exit("Error: Data empty after cleaning step. Check NaN drop logic and HHMM column creation.")
    return df.reset_index(drop=True)


def engineer_features_for_eu(df, original_model_stats):
    print(f"Engineering features for EU data...")
    features_df = df.copy()

    internal_flight_date_col = "FlightDate"
    if internal_flight_date_col not in features_df.columns:
        sys.exit(
            f"ERROR: Crucial column '{internal_flight_date_col}' not in DataFrame for feature engineering. Columns: {features_df.columns.tolist()}"
        )
    features_df["FlightDate_dt_obj"] = pd.to_datetime(
        features_df[internal_flight_date_col], errors="coerce"
    ).dt.date # Keep as date object for combine_date_time_objects

    time_cols_map = {
        "CRSDepTime_t_obj": "CRSDepTime_hhmm",
        "CRSArrTime_t_obj": "CRSArrTime_hhmm",
        "DepTime_t_obj": "DepTime_hhmm",
        "ArrTime_t_obj": "ArrTime_hhmm",
    }
    for new_obj_col, src_hhmm_col in time_cols_map.items():
        if src_hhmm_col in features_df.columns:
            # Ensure the source column is not all NaN before applying parse_hhmm_to_time
            if features_df[src_hhmm_col].isnull().all():
                 print(f"Warning: Source HHMM column '{src_hhmm_col}' for creating '{new_obj_col}' is all NaN. '{new_obj_col}' will be all NaT.")
                 features_df[new_obj_col] = pd.NaT
            else:
                features_df[new_obj_col] = features_df[src_hhmm_col].apply(parse_hhmm_to_time)
        else:
            print(
                f"Warning: Source HHMM column '{src_hhmm_col}' for creating '{new_obj_col}' not found in features_df. Setting to NaT."
            )
            features_df[new_obj_col] = pd.NaT

    datetime_map = {
        "SchedDepDateTime": ("FlightDate_dt_obj", "CRSDepTime_t_obj"),
        "SchedArrDateTime": ("FlightDate_dt_obj", "CRSArrTime_t_obj"),
        "ActualDepDateTime": ("FlightDate_dt_obj", "DepTime_t_obj"),
        "ActualArrDateTime": ("FlightDate_dt_obj", "ArrTime_t_obj"),
    }
    for dt_col, (date_src, time_src) in datetime_map.items():
        if date_src in features_df.columns and time_src in features_df.columns:
            # Check if source columns for combination are all NaT/NaN
            if features_df[date_src].isnull().all() or features_df[time_src].isnull().all():
                print(f"Warning: One or both source columns ('{date_src}', '{time_src}') for '{dt_col}' are all NaT/NaN. '{dt_col}' will be all NaT.")
                features_df[dt_col] = pd.NaT
            else:
                features_df[dt_col] = features_df.apply(
                    lambda r: combine_date_time_objects(r[date_src], r[time_src]), axis=1
                )
        else:
            print(
                f"Warning: Cannot create '{dt_col}' due to missing source '{date_src}' or '{time_src}'. Setting to NaT."
            )
            features_df[dt_col] = pd.NaT

    datetime_cols_to_check_after_combine = [
        "SchedDepDateTime",
        "SchedArrDateTime",
        "ActualDepDateTime",
        "ActualArrDateTime",
    ]
    print(f"Shape before NaN drop on combined datetime objects: {features_df.shape}")
    features_df.dropna(subset=datetime_cols_to_check_after_combine, inplace=True)
    print(f"Shape after NaN drop on combined datetime objects: {features_df.shape}")

    if features_df.empty:
        sys.exit("Error: No valid rows after datetime object combination and NaN drop. Check parsing of HHMM columns and date/time combination.")

    for sched_col, base_col in [
        ("SchedArrDateTime", "SchedDepDateTime"),
        ("ActualArrDateTime", "ActualDepDateTime"),
    ]:
        if sched_col in features_df.columns and base_col in features_df.columns:
            mask = features_df[sched_col] < features_df[base_col]
            features_df.loc[mask, sched_col] += pd.Timedelta(days=1)

    if "SchedDepDateTime" in features_df.columns:
        features_df["DateTime_ToSortBy"] = features_df["SchedDepDateTime"]
    else:
        sys.exit("Error: SchedDepDateTime missing, cannot create DateTime_ToSortBy.")

    if not original_model_stats or "feature_names" not in original_model_stats:
        sys.exit("ERROR: Original model's feature_names not found in stats. Cannot proceed.")

    original_target_feature_names = original_model_stats["feature_names"]
    print(f"Model expects {len(original_target_feature_names)} features in this order: {original_target_feature_names}")

    if "Month" in original_target_feature_names and "SchedDepDateTime" in features_df:
        features_df["Month"] = features_df["SchedDepDateTime"].dt.month
    if "DayOfMonth" in original_target_feature_names and "SchedDepDateTime" in features_df:
        features_df["DayOfMonth"] = features_df["SchedDepDateTime"].dt.day
    if "DayOfWeek" in original_target_feature_names and "SchedDepDateTime" in features_df:
        features_df["DayOfWeek"] = features_df["SchedDepDateTime"].dt.dayofweek
    if "Hour" in original_target_feature_names and "SchedDepDateTime" in features_df:
        features_df["Hour"] = features_df["SchedDepDateTime"].dt.hour

    final_features_for_model_df = pd.DataFrame(index=features_df.index)
    original_encoder_cats = original_model_stats.get("encoder_categories", {})
    original_categorical_feature_names_from_stats = list(original_encoder_cats.keys())

    if original_categorical_feature_names_from_stats:
        print(f"Processing original categorical features from stats: {original_categorical_feature_names_from_stats}")
        df_for_encoding = pd.DataFrame(index=features_df.index)
        for orig_cat_feat in original_categorical_feature_names_from_stats:
            if orig_cat_feat in features_df.columns:
                df_for_encoding[orig_cat_feat] = features_df[orig_cat_feat].fillna("__MISSING__").astype(str)
            else:
                print(f"Warning: Original categorical feature '{orig_cat_feat}' not found in EU data. Filling with '__MISSING__' for encoder.")
                df_for_encoding[orig_cat_feat] = "__MISSING__"
        try:
            encoder_categories_list = [np.array(original_encoder_cats[feat], dtype=object) for feat in original_categorical_feature_names_from_stats]
            encoder = OrdinalEncoder(categories=encoder_categories_list, handle_unknown="use_encoded_value", unknown_value=-1)
            dummy_fit_data = {feat: original_encoder_cats[feat] for feat in original_categorical_feature_names_from_stats}
            encoder.fit(pd.DataFrame(dummy_fit_data))
            encoded_data_np = encoder.transform(df_for_encoding[original_categorical_feature_names_from_stats])
            for i, col_name in enumerate(original_categorical_feature_names_from_stats):
                final_features_for_model_df[col_name] = encoded_data_np[:, i]
        except Exception as e:
            print(f"Error during Ordinal Encoding for features {original_categorical_feature_names_from_stats}: {e}. Filling with -1.")
            for col_name in original_categorical_feature_names_from_stats:
                final_features_for_model_df[col_name] = -1.0

    for feat_name in original_target_feature_names:
        if feat_name not in final_features_for_model_df.columns:
            if feat_name in features_df.columns:
                final_features_for_model_df[feat_name] = features_df[feat_name]
            else:
                print(f"Warning: Feature '{feat_name}' expected by model not found in EU data or created. Filling with 0 before scaling.")
                final_features_for_model_df[feat_name] = 0.0

    original_col_stats_numeric = original_model_stats.get("numeric_stats", {})
    for col in original_target_feature_names:
        if col in final_features_for_model_df.columns:
            final_features_for_model_df[col] = pd.to_numeric(final_features_for_model_df[col], errors="coerce")
            default_fill_value = 0.0
            if col in original_categorical_feature_names_from_stats:
                default_fill_value = -1.0
            fill_val = original_col_stats_numeric.get(col, {}).get("median", original_col_stats_numeric.get(col, {}).get("mean", default_fill_value))
            final_features_for_model_df[col].fillna(fill_val, inplace=True)
        else:
            print(f"Critical Warning: Column '{col}' is in original_target_feature_names but not populated in final_features_for_model_df before scaling. Filling with 0.")
            final_features_for_model_df[col] = 0.0

    original_scaler_params = original_model_stats.get("scaler_params", {})
    features_to_scale_as_per_original_scaler = original_scaler_params.get("feature_names", [])

    if features_to_scale_as_per_original_scaler and "min" in original_scaler_params and "scale" in original_scaler_params:
        print(f"Scaling features using original model's scaler. Features for scaler: {features_to_scale_as_per_original_scaler}")
        scaler = MinMaxScaler()
        scaler.min_ = np.array(original_scaler_params["min"])
        scaler.scale_ = np.array(original_scaler_params["scale"])
        df_for_scaling_ordered = final_features_for_model_df.reindex(columns=features_to_scale_as_per_original_scaler, fill_value=0.0)
        try:
            scaled_values = scaler.transform(df_for_scaling_ordered)
            scaled_df = pd.DataFrame(scaled_values, columns=features_to_scale_as_per_original_scaler, index=final_features_for_model_df.index)
            for col_scaled in features_to_scale_as_per_original_scaler:
                if col_scaled in final_features_for_model_df.columns:
                    final_features_for_model_df[col_scaled] = scaled_df[col_scaled]
                else:
                    print(f"Error: Column {col_scaled} was scaled but not found in final_features_for_model_df to update.")
        except ValueError as ve:
            print(f"ValueError during MinMax scaling: {ve}. This often means a feature expected by the scaler is not found or has an incorrect type in df_for_scaling_ordered.")
            print(f"Scaler expected features: {features_to_scale_as_per_original_scaler}")
            print(f"Columns in df_for_scaling_ordered: {df_for_scaling_ordered.columns.tolist()}")
            print(f"dtypes of df_for_scaling_ordered: \n{df_for_scaling_ordered.dtypes}")
            print(f"NaN sum in df_for_scaling_ordered: \n{df_for_scaling_ordered.isnull().sum()}")
        except Exception as e:
            print(f"General error during MinMax scaling: {e}.")
    else:
        print("Warning: Original scaler params missing or no features listed for scaler. Proceeding with unscaled (but filled) data where applicable.")

    processed_features_df = final_features_for_model_df.reindex(columns=original_target_feature_names, fill_value=0.0)
    num_final_features = len(original_target_feature_names)
    print(f"Final features for model input ({num_final_features}), aligned with original model: {original_target_feature_names}")

    cols_for_chaining = [
        "Tail_Number", "DateTime_ToSortBy", TARGET_COL_INTERNAL_NAME,
        "SchedArrDateTime", "ActualDepDateTime", "ActualArrDateTime",
    ]
    # Ensure essential columns for chaining are present in the original features_df (before it was refined into final_features_for_model_df)
    # Note: features_df here is the one passed into this function, which is df from load_and_preprocess_eu_data
    # It *should* have these columns if they were correctly named/created.
    missing_chaining_cols_from_input_df = [col for col in cols_for_chaining if col not in df.columns] # Check 'df' not 'features_df' for this specific check
    if missing_chaining_cols_from_input_df:
        # DateTime_ToSortBy, SchedArrDateTime etc. are CREATED in this function, so they wouldn't be in input `df`.
        # Check `features_df` (the copy that gets modified) for these later.
        # For now, only check for "Tail_Number" and TARGET_COL_INTERNAL_NAME in the input `df`.
        check_in_input_df = ["Tail_Number", TARGET_COL_INTERNAL_NAME]
        missing_from_input_df_for_concat = [col for col in check_in_input_df if col not in df.columns]
        if missing_from_input_df_for_concat:
             sys.exit(
                f"ERROR: Critical columns for chain construction ('Tail_Number', '{TARGET_COL_INTERNAL_NAME}') missing from initial DataFrame "
                f"passed to engineer_features_for_eu: {missing_from_input_df_for_concat}. Available: {df.columns.tolist()}"
            )
        # For the datetime columns created within this function, we'll rely on their presence in `features_df` (the working copy)
        # and handle errors if they are not created properly.
        df_for_concat_base = df[check_in_input_df].copy()
    else: # All original columns are present, use them
        df_for_concat_base = df[cols_for_chaining].copy()


    # Now add the newly created datetime columns from features_df (the working copy)
    # to the df_for_concat_base
    datetime_cols_to_add_for_chaining = ["DateTime_ToSortBy", "SchedArrDateTime", "ActualDepDateTime", "ActualArrDateTime"]
    for dt_col in datetime_cols_to_add_for_chaining:
        if dt_col in features_df.columns:
            df_for_concat_base[dt_col] = features_df[dt_col]
        else:
            sys.exit(f"ERROR: Datetime column '{dt_col}' essential for chaining was not successfully created or is missing from features_df.")


    output_df = pd.concat([df_for_concat_base, processed_features_df], axis=1)
    eu_data_stats_summary = {
        "num_features": num_final_features,
        "feature_names_order": original_target_feature_names,
    }
    return output_df, original_target_feature_names, eu_data_stats_summary


def create_chains_for_eu(df, feature_cols, target_col_internal_name):
    print("Constructing flight chains for EU data...")
    if df.empty:
        print("Warning: DataFrame provided to create_chains_for_eu is empty. No chains will be constructed.")
        num_features_placeholder = len(feature_cols) if feature_cols else original_model_stats.get("num_features", 1) if 'original_model_stats' in locals() else 1
        return np.array([]).reshape(0, CHAIN_LENGTH, num_features_placeholder), np.array([])

    num_features = len(feature_cols)

    required_dt_cols = ["DateTime_ToSortBy", "SchedArrDateTime", "ActualDepDateTime", "ActualArrDateTime"]
    missing_dt_for_chain = [col for col in required_dt_cols if col not in df.columns]
    if missing_dt_for_chain:
        sys.exit(f"ERROR: Essential datetime columns for chain construction missing from input df: {missing_dt_for_chain}. DF cols: {df.columns.tolist()}")

    # Convert to datetime just in case, though they should be from engineering
    for col in required_dt_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df.dropna(subset=required_dt_cols, inplace=True)
    if df.empty:
        print("Warning: DataFrame empty after ensuring valid datetimes for chain construction.")
        return np.array([]).reshape(0, CHAIN_LENGTH, num_features), np.array([])

    if "Tail_Number" not in df.columns or target_col_internal_name not in df.columns:
        sys.exit(f"ERROR: 'Tail_Number' or target '{target_col_internal_name}' missing for chain construction. DF cols: {df.columns.tolist()}")

    df = df.sort_values(by=["Tail_Number", "DateTime_ToSortBy"]).reset_index(drop=True)

    missing_features_for_np = [fc for fc in feature_cols if fc not in df.columns]
    if missing_features_for_np:
        sys.exit(f"ERROR: Model input features missing from DataFrame before creating chains: {missing_features_for_np}. DF cols: {df.columns.tolist()}")

    tail_numbers_np = df["Tail_Number"].values
    actual_dep_times_sec = df["ActualDepDateTime"].values.astype(np.int64) // 10**9
    actual_arr_times_sec = df["ActualArrDateTime"].values.astype(np.int64) // 10**9
    features_np = df[feature_cols].values.astype(np.float32)
    target_np = df[target_col_internal_name].values.astype(np.float32) # Ensure target is float for pd.cut

    max_ground_seconds = MAX_GROUND_TIME_HOURS * 3600
    min_turnaround_sec = MIN_TURNAROUND_MINUTES * 60

    chains, labels, skipped_validation = [], [], 0
    unique_tails = df["Tail_Number"].unique()

    for tail_num in tqdm(unique_tails, desc="Processing Tail Numbers for Chains"):
        group_indices = df[df["Tail_Number"] == tail_num].index
        if len(group_indices) < CHAIN_LENGTH:
            continue

        current_dep_times = actual_dep_times_sec[group_indices]
        current_arr_times = actual_arr_times_sec[group_indices]
        current_features = features_np[group_indices]
        current_targets = target_np[group_indices]

        for j in range(len(group_indices) - CHAIN_LENGTH + 1):
            dep_f2_val = current_dep_times[j + 1]
            arr_f1_val = current_arr_times[j]
            dep_f3_val = current_dep_times[j + 2]
            arr_f2_val = current_arr_times[j + 1]

            valid = True
            if not (dep_f2_val > (arr_f1_val + min_turnaround_sec) and \
                    dep_f3_val > (arr_f2_val + min_turnaround_sec)):
                valid = False
            if valid:
                ground_time12 = dep_f2_val - arr_f1_val
                ground_time23 = dep_f3_val - arr_f2_val
                if not (ground_time12 <= max_ground_seconds and \
                        ground_time23 <= max_ground_seconds):
                    valid = False
            if not valid:
                skipped_validation += 1
                continue

            chain_data = current_features[j : j + CHAIN_LENGTH]
            label_raw = current_targets[j + CHAIN_LENGTH -1] # Target is from the 3rd flight (index 2)

            try:
                # Ensure label_raw is a scalar or 1-element array for pd.cut
                if not isinstance(label_raw, (int, float, np.number)):
                    if isinstance(label_raw, (list, np.ndarray)) and len(label_raw) == 1:
                        label_raw_scalar = label_raw[0]
                    else: # Should not happen if target_np is 1D
                        print(f"Unexpected label_raw type or shape: {label_raw}, type: {type(label_raw)}")
                        skipped_validation +=1
                        continue
                else:
                    label_raw_scalar = label_raw

                label_class_arr = pd.cut(
                    [label_raw_scalar], # Pass as a list/array-like
                    bins=DELAY_THRESHOLDS_CLASSIFICATION,
                    labels=False,
                    right=True,
                    include_lowest=True # Ensure the lowest bound is included
                )
                label_class = label_class_arr[0]

                if pd.isna(label_class): # Handle cases falling outside defined bins if include_lowest=False or other edge cases
                    print(f"Warning: label_raw {label_raw_scalar} resulted in NaN class. Bins: {DELAY_THRESHOLDS_CLASSIFICATION}")
                    # Fallback logic for NaN class (e.g. assign to most frequent or an 'unknown' class if you have one)
                    # For now, let's try to assign based on simple threshold as a fallback
                    if label_raw_scalar <= DELAY_THRESHOLDS_CLASSIFICATION[1]: # <= 15 min
                        label_class = 0
                    else: # Treat as extremely delayed if unclassified
                        label_class = NUM_CLASSES_CLASSIFICATION - 1
                
                chains.append(chain_data)
                labels.append(int(label_class))
            except Exception as e:
                print(f"Error during label classification for raw value {label_raw}: {e}")
                skipped_validation += 1
                continue
    
    print(f"Skipped {skipped_validation} chains (validation/labeling). Constructed {len(chains)} valid chains.")
    if not chains:
        print("Warning: No valid chains were constructed from EU data.")
        return np.array([]).reshape(0, CHAIN_LENGTH, num_features), np.array([])
    return np.array(chains, dtype=np.float32), np.array(labels, dtype=np.int64)


def run_eu_chain_construction():
    print("--- Starting EU Chain Construction for Prediction ---")
    original_model_stats = load_original_data_stats_for_scaling()
    if not original_model_stats:
        sys.exit("Original model stats not loaded. Cannot proceed.")

    df_processed_from_load = load_and_preprocess_eu_data(MERGED_EU_DATA_FILE)
    (
        df_final_featured,
        final_model_feature_names_in_order,
        eu_processing_stats_summary,
    ) = engineer_features_for_eu(df_processed_from_load, original_model_stats)

    if TARGET_COL_INTERNAL_NAME not in df_final_featured.columns:
        sys.exit(
            f"Target column '{TARGET_COL_INTERNAL_NAME}' missing after feature engineering. Columns available: {df_final_featured.columns.tolist()}"
        )

    chains_np, labels_np = create_chains_for_eu(
        df_final_featured, final_model_feature_names_in_order, TARGET_COL_INTERNAL_NAME
    )

    if len(chains_np) > 0:
        print("Saving EU chains and labels for prediction/evaluation...")
        PROCESSED_EU_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EU_TEST_CHAINS_FILE, chains_np)
        np.save(EU_TEST_LABELS_FILE, labels_np)
        try:
            with open(EU_DATA_STATS_FILE, "w") as f:
                json.dump(eu_processing_stats_summary, f, indent=4)
            print(f"Saved EU data processing summary stats to {EU_DATA_STATS_FILE}")
        except Exception as e:
            print(f"Error saving EU data processing summary stats: {e}")
        print(f"Processed data saved to directory: {PROCESSED_EU_CHAINS_DIR}")
    else:
        print("No EU chains were constructed. Nothing to save.")

    print("--- EU Chain Construction Finished ---")


if __name__ == "__main__":
    run_eu_chain_construction()
