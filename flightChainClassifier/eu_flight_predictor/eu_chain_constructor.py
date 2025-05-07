# eu_flight_predictor/eu_chain_constructor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import sys
import os
import json
from tqdm import tqdm
import warnings
from datetime import datetime

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
from utils.datetime_helpers import parse_hhmm_to_time, combine_date_time_objects

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def extract_hhmm_from_datetime_str(dt_str):
    """Extracts HHMM string from a full datetime string like 'YYYY-MM-DD HH:MM' or ISO."""
    if pd.isna(dt_str):
        return None
    try:
        dt_obj = pd.to_datetime(
            dt_str
        )  # Pandas is good at parsing various datetime formats
        return dt_obj.strftime("%H%M")
    except (ValueError, TypeError):  # Fallback for non-standard strings
        # Attempt to find a time-like part if direct parsing fails
        str_val = str(dt_str)
        parts = str_val.split()
        time_like_part = (
            parts[-1] if parts else str_val
        )  # Use last part or whole string
        hhmm = "".join(filter(str.isdigit, time_like_part))
        if len(hhmm) >= 4:
            return hhmm[:4]  # If '10:30:00' -> '1030'
        if len(hhmm) == 3:
            return "0" + hhmm  # If '930' -> '0930'
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
        and csv_name
        in df.columns  # Only map if CSV name exists in df and is not None in mapping
    }
    df.rename(columns=rename_map, inplace=True)
    print(f"Renamed columns. Current columns after rename: {df.columns.tolist()}")

    # HHMM Extraction
    # EU_DATETIME_COLS_NEED_HHMM_PROCESSING maps:
    # new_internal_hhmm_col_name : source_CSV_col_name_containing_full_datetime
    for (
        new_internal_hhmm_col,
        source_csv_col_original_name,
    ) in EU_DATETIME_COLS_NEED_HHMM_PROCESSING.items():
        # Find what source_csv_col_original_name was renamed to (its internal PascalCase name)
        source_internal_name_after_rename = None
        for internal_key_in_map, mapped_csv_val in EU_COLUMN_MAPPING_RAW.items():
            if mapped_csv_val == source_csv_col_original_name:
                source_internal_name_after_rename = internal_key_in_map
                break

        if (
            source_internal_name_after_rename
            and source_internal_name_after_rename in df.columns
        ):
            print(
                f"Extracting HHMM for '{source_internal_name_after_rename}' (from CSV '{source_csv_col_original_name}') into '{new_internal_hhmm_col}'..."
            )
            df[new_internal_hhmm_col] = df[source_internal_name_after_rename].apply(
                extract_hhmm_from_datetime_str
            )
        else:
            print(
                f"Warning: Source for HHMM extraction '{new_internal_hhmm_col}' (expected from CSV '{source_csv_col_original_name}', mapped to internal '{source_internal_name_after_rename}') not found in DataFrame. Skipping."
            )

    # Cancelled/Diverted Flags - original model expects features "Cancelled" and "Diverted" (PascalCase)
    # EU_STATUS_COL_CSV_NAME is the name of the column in the CSV file (e.g., "status")
    if (
        EU_STATUS_COL_CSV_NAME and EU_STATUS_COL_CSV_NAME in df.columns
    ):  # Check if original CSV 'status' col is still there
        print(
            f"Processing CSV column '{EU_STATUS_COL_CSV_NAME}' to create 'Cancelled' and 'Diverted' features..."
        )
        df["Cancelled"] = df[EU_STATUS_COL_CSV_NAME].apply(
            lambda x: 1.0 if isinstance(x, str) and x.lower() == "cancelled" else 0.0
        )
        df["Diverted"] = df[EU_STATUS_COL_CSV_NAME].apply(
            lambda x: 1.0 if isinstance(x, str) and x.lower() == "diverted" else 0.0
        )
    else:
        # This case could also be hit if "status" was renamed via EU_COLUMN_MAPPING_RAW to something else.
        # The current config does not do that, so we expect 'status' to be the column name here.
        print(
            f"Warning: Status CSV column '{EU_STATUS_COL_CSV_NAME}' not found in DataFrame (columns: {df.columns.tolist()}). 'Cancelled' & 'Diverted' features set to 0."
        )
        df["Cancelled"] = 0.0
        df["Diverted"] = 0.0

    # NaN drop and filter
    # These are INTERNAL names the rest of the script/model expects.
    critical_cols_for_nan_drop = [
        "FlightDate",
        "Tail_Number",
        "CRSDepTime_hhmm",
        "CRSArrTime_hhmm",
        TARGET_COL_INTERNAL_NAME,  # This is "ArrDelayMinutes"
        "Origin",
        "Dest",
        "DepTime_hhmm",
        "ArrTime_hhmm",
        "CRSElapsedTime",  # This is an internal name, mapped from "scheduled_duration"
        "Cancelled",
        "Diverted",  # These are newly created internal feature names
    ]

    actual_critical_for_drop = [
        col for col in critical_cols_for_nan_drop if col in df.columns
    ]
    missing_for_drop = set(critical_cols_for_nan_drop) - set(actual_critical_for_drop)

    if missing_for_drop:
        print(f"Warning: Columns missing for NaN drop: {missing_for_drop}.")
        print(
            f"Will attempt to drop NaNs from available critical columns: {actual_critical_for_drop}"
        )
        print(f"All df columns before drop: {df.columns.tolist()}")

    if actual_critical_for_drop:  # Only drop if there are columns to check based on
        df.dropna(subset=actual_critical_for_drop, inplace=True)
    else:
        print(
            "Warning: No critical columns found to perform NaN drop. Data might be unsuitable."
        )

    # Filter based on the "Cancelled" and "Diverted" features we just created
    if "Cancelled" in df.columns and "Diverted" in df.columns:
        df = df[(df["Cancelled"] != 1.0) & (df["Diverted"] != 1.0)]
    else:
        # This should not happen if the above status processing worked
        print("Error: 'Cancelled' or 'Diverted' columns not created for filtering.")

    print(
        f"Data shape after cleaning (NaN drop and Cancelled/Diverted filter): {df.shape}"
    )
    if df.empty:
        sys.exit("Error: Data empty after cleaning step.")
    return df.reset_index(drop=True)


def engineer_features_for_eu(df, original_model_stats):
    print(f"Engineering features for EU data...")
    features_df = (
        df.copy()
    )  # df here has renamed columns and new _hhmm, Cancelled, Diverted columns

    # --- Datetime object creation (using the _hhmm columns) ---
    internal_flight_date_col = "FlightDate"  # This is the internal name AFTER rename
    if internal_flight_date_col not in features_df.columns:
        sys.exit(
            f"ERROR: Crucial column '{internal_flight_date_col}' not in DataFrame for feature engineering. Columns: {features_df.columns.tolist()}"
        )
    features_df["FlightDate_dt_obj"] = pd.to_datetime(
        features_df[internal_flight_date_col], errors="coerce"
    ).dt.date

    time_cols_map = {
        "CRSDepTime_t_obj": "CRSDepTime_hhmm",
        "CRSArrTime_t_obj": "CRSArrTime_hhmm",
        "DepTime_t_obj": "DepTime_hhmm",
        "ArrTime_t_obj": "ArrTime_hhmm",
    }
    for new_obj_col, src_hhmm_col in time_cols_map.items():
        if src_hhmm_col in features_df.columns:
            features_df[new_obj_col] = features_df[src_hhmm_col].apply(
                parse_hhmm_to_time
            )
        else:
            print(
                f"Warning: Source HHMM column '{src_hhmm_col}' for creating '{new_obj_col}' not found in features_df. Setting to NaT."
            )
            features_df[new_obj_col] = (
                pd.NaT
            )  # Assign NaT so subsequent combination might fail gracefully

    datetime_map = {
        "SchedDepDateTime": ("FlightDate_dt_obj", "CRSDepTime_t_obj"),
        "SchedArrDateTime": ("FlightDate_dt_obj", "CRSArrTime_t_obj"),
        "ActualDepDateTime": ("FlightDate_dt_obj", "DepTime_t_obj"),
        "ActualArrDateTime": ("FlightDate_dt_obj", "ArrTime_t_obj"),
    }
    for dt_col, (date_src, time_src) in datetime_map.items():
        if date_src in features_df.columns and time_src in features_df.columns:
            features_df[dt_col] = features_df.apply(
                lambda r: combine_date_time_objects(r[date_src], r[time_src]), axis=1
            )
        else:
            print(
                f"Warning: Cannot create '{dt_col}' due to missing source '{date_src}' or '{time_src}'. Setting to NaT."
            )
            features_df[dt_col] = pd.NaT  # Assign NaT

    datetime_cols_to_check_after_combine = [
        "SchedDepDateTime",
        "SchedArrDateTime",
        "ActualDepDateTime",
        "ActualArrDateTime",
    ]
    features_df.dropna(subset=datetime_cols_to_check_after_combine, inplace=True)
    if features_df.empty:
        sys.exit("Error: No valid rows after datetime object combination and NaN drop.")

    # Adjust for overnight flights
    for sched_col, base_col in [
        ("SchedArrDateTime", "SchedDepDateTime"),
        ("ActualArrDateTime", "ActualDepDateTime"),
    ]:
        if (
            sched_col in features_df.columns and base_col in features_df.columns
        ):  # Check presence
            mask = features_df[sched_col] < features_df[base_col]
            features_df.loc[mask, sched_col] += pd.Timedelta(days=1)
    if "SchedDepDateTime" in features_df.columns:
        features_df["DateTime_ToSortBy"] = features_df["SchedDepDateTime"]
    else:  # Should not happen if previous dropna worked
        sys.exit("Error: SchedDepDateTime missing, cannot create DateTime_ToSortBy.")

    # Feature engineering based on original_model_stats
    if not original_model_stats or "feature_names" not in original_model_stats:
        sys.exit(
            "ERROR: Original model's feature_names not found in stats. Cannot proceed."
        )

    original_target_feature_names = original_model_stats[
        "feature_names"
    ]  # The final list of features model expects IN ORDER
    print(
        f"Model expects {len(original_target_feature_names)} features in this order: {original_target_feature_names}"
    )

    # Temporal features (Month, DayOfMonth, DayOfWeek, Hour from SchedDepDateTime)
    # These are common features and should be named as per original_target_feature_names if they are there
    if "Month" in original_target_feature_names:
        features_df["Month"] = features_df["SchedDepDateTime"].dt.month
    if "DayOfMonth" in original_target_feature_names:
        features_df["DayOfMonth"] = features_df["SchedDepDateTime"].dt.day
    if "DayOfWeek" in original_target_feature_names:
        features_df["DayOfWeek"] = features_df["SchedDepDateTime"].dt.dayofweek
    if "Hour" in original_target_feature_names:
        features_df["Hour"] = features_df["SchedDepDateTime"].dt.hour

    # Prepare a DataFrame that will hold the final features in the correct order
    final_features_for_model_df = pd.DataFrame(index=features_df.index)

    # Categorical features: Process based on original_encoder_cats
    original_encoder_cats = original_model_stats.get("encoder_categories", {})
    original_categorical_feature_names_from_stats = list(
        original_encoder_cats.keys()
    )  # e.g., ["Reporting_Airline", "Origin", "Dest"]

    if original_categorical_feature_names_from_stats:
        print(
            f"Processing original categorical features from stats: {original_categorical_feature_names_from_stats}"
        )

        df_for_encoding = pd.DataFrame(index=features_df.index)
        # We need to ensure df_for_encoding has columns matching original_categorical_feature_names_from_stats
        for orig_cat_feat in original_categorical_feature_names_from_stats:
            if (
                orig_cat_feat in features_df.columns
            ):  # e.g., "Reporting_Airline" is in features_df
                df_for_encoding[orig_cat_feat] = (
                    features_df[orig_cat_feat].fillna("__MISSING__").astype(str)
                )
            else:
                # This means an original categorical feature is not in the current EU data (after renames)
                print(
                    f"Warning: Original categorical feature '{orig_cat_feat}' not found in EU data. Filling with '__MISSING__' for encoder."
                )
                df_for_encoding[orig_cat_feat] = "__MISSING__"

        try:
            encoder_categories_list = [
                np.array(original_encoder_cats[feat], dtype=object)
                for feat in original_categorical_feature_names_from_stats
            ]
            encoder = OrdinalEncoder(
                categories=encoder_categories_list,
                handle_unknown="use_encoded_value",
                unknown_value=-1,  # This value will be scaled later
            )
            # Fit with dummy data using the original categories to set up the encoder correctly
            dummy_fit_data = {
                feat: original_encoder_cats[feat]
                for feat in original_categorical_feature_names_from_stats
            }
            encoder.fit(pd.DataFrame(dummy_fit_data))

            encoded_data_np = encoder.transform(
                df_for_encoding[original_categorical_feature_names_from_stats]
            )

            for i, col_name in enumerate(original_categorical_feature_names_from_stats):
                final_features_for_model_df[col_name] = encoded_data_np[:, i]
        except Exception as e:
            print(
                f"Error during Ordinal Encoding for features {original_categorical_feature_names_from_stats}: {e}. Filling with -1."
            )
            for col_name in original_categorical_feature_names_from_stats:
                final_features_for_model_df[col_name] = -1.0

    # Populate all features the original model expects into final_features_for_model_df
    # original_target_feature_names is the definitive list from data_stats.json["feature_names"]
    for feat_name in original_target_feature_names:
        if (
            feat_name not in final_features_for_model_df.columns
        ):  # If not already added (e.g., as an encoded categorical)
            if (
                feat_name in features_df.columns
            ):  # Check if it's a direct column (numerical, temporal, or created flags like Cancelled)
                final_features_for_model_df[feat_name] = features_df[feat_name]
            else:
                # This feature is expected by the model but is not in EU data directly,
                # nor was it an original categorical feature. (e.g. "AirTime", "TaxiOut" if mapping was None)
                print(
                    f"Warning: Feature '{feat_name}' expected by model not found in EU data or created. Filling with 0 before scaling."
                )
                final_features_for_model_df[feat_name] = 0.0

    # Ensure all columns are numeric and fill NaNs before scaling, using original stats if possible
    original_col_stats_numeric = original_model_stats.get("numeric_stats", {})
    for (
        col
    ) in (
        original_target_feature_names
    ):  # Iterate over all features the final model expects
        if col in final_features_for_model_df.columns:
            final_features_for_model_df[col] = pd.to_numeric(
                final_features_for_model_df[col], errors="coerce"
            )
            # Determine fill value: Median from original stats > Mean from original stats > 0.0
            # For encoded categoricals that became -1 and then NaN via to_numeric, they should be filled appropriately too.
            # If -1 was the unknown_value, it should remain -1 or be scaled.
            # If a feature was purely numeric, use its stats.
            default_fill_value = 0.0
            if (
                col in original_categorical_feature_names_from_stats
            ):  # If it was an encoded categorical
                default_fill_value = (
                    -1.0
                )  # Assuming -1 was unknown_value and should be preserved if it became NaN

            fill_val = original_col_stats_numeric.get(col, {}).get(
                "median",
                original_col_stats_numeric.get(col, {}).get("mean", default_fill_value),
            )

            final_features_for_model_df[col].fillna(fill_val, inplace=True)
        else:
            # This should ideally not happen if the previous loop correctly populated all original_target_feature_names
            print(
                f"Critical Warning: Column '{col}' is in original_target_feature_names but not populated in final_features_for_model_df before scaling. Filling with 0."
            )
            final_features_for_model_df[col] = 0.0

    # Scaling: Apply original scaler.
    # The scaler was fit on features listed in original_model_stats["scaler_params"]["feature_names"].
    # This list usually contains all features (numerical, temporal, and ORDINAL ENCODED categoricals).
    original_scaler_params = original_model_stats.get("scaler_params", {})
    features_to_scale_as_per_original_scaler = original_scaler_params.get(
        "feature_names", []
    )  # This is the crucial list for scaler

    if (
        features_to_scale_as_per_original_scaler
        and "min" in original_scaler_params
        and "scale" in original_scaler_params
    ):
        print(
            f"Scaling features using original model's scaler. Features for scaler: {features_to_scale_as_per_original_scaler}"
        )
        scaler = MinMaxScaler()
        scaler.min_ = np.array(original_scaler_params["min"])
        scaler.scale_ = np.array(original_scaler_params["scale"])

        # Ensure df for scaling has columns in correct order and all are present
        # It should use final_features_for_model_df which has undergone encoding and filling.
        # The columns in final_features_for_model_df should ideally already match original_target_feature_names.
        # And features_to_scale_as_per_original_scaler should be a subset of or equal to original_target_feature_names.

        # Reindex final_features_for_model_df to match the exact order and columns the scaler expects.
        df_for_scaling_ordered = final_features_for_model_df.reindex(
            columns=features_to_scale_as_per_original_scaler, fill_value=0.0
        )

        try:
            scaled_values = scaler.transform(df_for_scaling_ordered)
            scaled_df = pd.DataFrame(
                scaled_values,
                columns=features_to_scale_as_per_original_scaler,
                index=final_features_for_model_df.index,
            )
            # Update final_features_for_model_df with scaled values for the columns that were scaled
            for col_scaled in features_to_scale_as_per_original_scaler:
                if (
                    col_scaled in final_features_for_model_df.columns
                ):  # Should always be true
                    final_features_for_model_df[col_scaled] = scaled_df[col_scaled]
                else:  # Should not happen
                    print(
                        f"Error: Column {col_scaled} was scaled but not found in final_features_for_model_df to update."
                    )
        except ValueError as ve:
            print(
                f"ValueError during MinMax scaling: {ve}. This often means a feature expected by the scaler is not found or has an incorrect type in df_for_scaling_ordered."
            )
            print(
                f"Scaler expected features: {features_to_scale_as_per_original_scaler}"
            )
            print(
                f"Columns in df_for_scaling_ordered: {df_for_scaling_ordered.columns.tolist()}"
            )
            print(
                f"dtypes of df_for_scaling_ordered: \n{df_for_scaling_ordered.dtypes}"
            )
            print(
                f"NaN sum in df_for_scaling_ordered: \n{df_for_scaling_ordered.isnull().sum()}"
            )
            # Fallback: proceed with unscaled data for these columns if error.
        except Exception as e:
            print(f"General error during MinMax scaling: {e}.")
    else:
        print(
            "Warning: Original scaler params missing or no features listed for scaler. Proceeding with unscaled (but filled) data where applicable."
        )

    # Final DataFrame for the model input, ensuring columns are in the order of original_target_feature_names
    # final_features_for_model_df should now contain all necessary features, some scaled, some encoded.
    processed_features_df = final_features_for_model_df.reindex(
        columns=original_target_feature_names, fill_value=0.0
    )

    num_final_features = len(original_target_feature_names)
    print(
        f"Final features for model input ({num_final_features}), aligned with original model: {original_target_feature_names}"
    )

    # Combine with essential non-feature columns for chain construction
    cols_for_chaining = [
        "Tail_Number",  # This is already the internal name after rename
        "DateTime_ToSortBy",
        TARGET_COL_INTERNAL_NAME,  # Internal name for target
        "SchedArrDateTime",
        "ActualDepDateTime",
        "ActualArrDateTime",
    ]
    missing_chaining_cols = [
        col for col in cols_for_chaining if col not in features_df.columns
    ]
    if missing_chaining_cols:
        sys.exit(
            f"ERROR: Critical columns for chain construction missing from features_df: {missing_chaining_cols}. Available: {features_df.columns.tolist()}"
        )

    output_df = pd.concat(
        [features_df[cols_for_chaining], processed_features_df], axis=1
    )

    eu_data_stats_summary = {
        "num_features": num_final_features,
        "feature_names_order": original_target_feature_names,
    }
    return output_df, original_target_feature_names, eu_data_stats_summary


def create_chains_for_eu(df, feature_cols, target_col_internal_name):
    print("Constructing flight chains for EU data...")
    num_features = len(feature_cols)

    required_dt_cols = [
        "DateTime_ToSortBy",
        "SchedArrDateTime",
        "ActualDepDateTime",
        "ActualArrDateTime",
    ]
    missing_dt_for_chain = [col for col in required_dt_cols if col not in df.columns]
    if missing_dt_for_chain:
        sys.exit(
            f"ERROR: Essential datetime columns for chain construction missing from input df: {missing_dt_for_chain}. DF cols: {df.columns.tolist()}"
        )

    df.dropna(subset=required_dt_cols, inplace=True)
    if df.empty:
        print(
            "Warning: DataFrame empty after ensuring valid datetimes for chain construction."
        )
        return np.array([]).reshape(0, CHAIN_LENGTH, num_features), np.array([])

    if "Tail_Number" not in df.columns or target_col_internal_name not in df.columns:
        sys.exit(
            f"ERROR: 'Tail_Number' or target '{target_col_internal_name}' missing for chain construction. DF cols: {df.columns.tolist()}"
        )

    df = df.sort_values(by=["Tail_Number", "DateTime_ToSortBy"]).reset_index(drop=True)

    missing_features_for_np = [fc for fc in feature_cols if fc not in df.columns]
    if missing_features_for_np:
        sys.exit(
            f"ERROR: Model input features missing from DataFrame before creating chains: {missing_features_for_np}. DF cols: {df.columns.tolist()}"
        )

    tail_numbers_np = df["Tail_Number"].values
    actual_dep_times_sec = df["ActualDepDateTime"].values.astype(np.int64) // 10**9
    actual_arr_times_sec = df["ActualArrDateTime"].values.astype(np.int64) // 10**9
    features_np = df[feature_cols].values.astype(np.float32)
    target_np = df[target_col_internal_name].values

    max_ground_seconds = MAX_GROUND_TIME_HOURS * 3600
    min_turnaround_sec = MIN_TURNAROUND_MINUTES * 60

    chains, labels, skipped_validation = [], [], 0

    unique_tails = df["Tail_Number"].unique()
    for tail_num in tqdm(unique_tails, desc="Processing Tail Numbers for Chains"):
        group_indices = df[
            df["Tail_Number"] == tail_num
        ].index  # Get indices from the full sorted df
        if len(group_indices) < CHAIN_LENGTH:
            continue

        current_dep_times = actual_dep_times_sec[group_indices]
        current_arr_times = actual_arr_times_sec[group_indices]
        current_features = features_np[group_indices]
        current_targets = target_np[group_indices]

        for j in range(len(group_indices) - CHAIN_LENGTH + 1):
            # Sliced arrays are 0-indexed for the current group
            dep_f2_val = current_dep_times[j + 1]
            arr_f1_val = current_arr_times[j]
            dep_f3_val = current_dep_times[j + 2]
            arr_f2_val = current_arr_times[j + 1]

            valid = True
            if not (
                dep_f2_val > (arr_f1_val + min_turnaround_sec)
                and dep_f3_val > (arr_f2_val + min_turnaround_sec)
            ):
                valid = False

            if valid:  # Only check ground time if basic turnaround is met
                ground_time12 = dep_f2_val - arr_f1_val
                ground_time23 = dep_f3_val - arr_f2_val
                if not (
                    ground_time12 <= max_ground_seconds
                    and ground_time23 <= max_ground_seconds
                ):
                    valid = False

            if not valid:
                skipped_validation += 1
                continue

            chain_data = current_features[j : j + CHAIN_LENGTH]
            label_raw = current_targets[
                j + 2
            ]  # Target is from the 3rd flight in the chain

            try:
                label_class = pd.cut(
                    [label_raw],
                    bins=DELAY_THRESHOLDS_CLASSIFICATION,
                    labels=False,
                    right=True,
                )[0]
                if pd.isna(label_class):
                    label_class = (
                        0
                        if label_raw <= DELAY_THRESHOLDS_CLASSIFICATION[1]
                        else NUM_CLASSES_CLASSIFICATION - 1
                    )
                chains.append(chain_data)
                labels.append(int(label_class))
            except Exception:
                skipped_validation += 1
                continue

    print(
        f"Skipped {skipped_validation} chains (validation/labeling). Constructed {len(chains)} valid chains."
    )
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
        PROCESSED_EU_CHAINS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
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
