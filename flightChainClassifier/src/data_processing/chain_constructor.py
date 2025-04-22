# flightChainClassifier/src/data_processing/chain_constructor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import sys
import os
import json
from tqdm import tqdm
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # Ignore numpy/pandas future warnings if any

# --- Path Setup & Config Import ---
try:
    # Assumes this file is in src/data_processing
    from .. import config
except ImportError:
    # Handle case where script might be run directly or paths are different
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)  # Up one level to src/
    sys.path.insert(0, src_dir)  # Add src to path
    try:
        import config
    except ImportError:
        print(
            "CRITICAL: Cannot find 'config.py'. Ensure it's in the current working directory or src/ directory and PYTHONPATH is set correctly."
        )
        sys.exit(1)

# --- Helper Functions ---


def parse_hhmm(time_col_series):
    """Safely parses HHMM format (potentially float or string) to time objects."""
    # Ensure input is treated as nullable strings initially
    time_str = time_col_series.astype(str).replace(
        {r"\.0$": ""}, regex=True
    )  # Remove .0 from floats
    time_str = time_str.replace(
        {"nan": pd.NA, "None": pd.NA}
    )  # Handle string 'nan'/'None'

    # Pad with leading zeros ONLY if length < 4 and not NA
    time_str = time_str.apply(lambda x: x.zfill(4) if pd.notna(x) and len(x) < 4 else x)

    # Handle '2400' -> '0000'
    time_str = time_str.replace("2400", "0000")

    # Validate format HHMM (0000-2359)
    valid_time_mask = time_str.str.match(r"^([01]\d|2[0-3])([0-5]\d)$", na=False)

    # Convert valid times to time objects, invalid to NaT
    parsed_time = pd.to_datetime(
        time_str.where(valid_time_mask), format="%H%M", errors="coerce"
    ).dt.time
    return parsed_time


def combine_date_time(row, date_col, time_col):
    """Combines date object and time object into timestamp."""
    if pd.isna(row[date_col]) or pd.isna(row[time_col]):
        return pd.NaT
    try:
        return pd.Timestamp.combine(row[date_col], row[time_col])
    except (TypeError, ValueError):
        return pd.NaT


# --- Main Processing Functions ---


def load_raw_data(file_path):
    """Loads raw flight data, selecting required columns."""
    print(f"Loading raw data from {file_path}...")
    if not file_path.exists():
        print(f"Error: Raw data file not found at {file_path}")
        sys.exit(1)
    try:
        # Check available columns
        header = pd.read_csv(file_path, nrows=0).columns.tolist()
        available_cols = [col for col in config.REQUIRED_RAW_COLS if col in header]
        missing_req_cols = set(config.REQUIRED_RAW_COLS) - set(available_cols)
        if missing_req_cols:
            print(
                f"Warning: Required columns missing from raw file and will be skipped: {missing_req_cols}"
            )

        print(f"Loading available required columns: {available_cols}")
        # Specify dtype for potentially large string columns to avoid mixed type warnings
        dtype_spec = {
            col: str
            for col in ["Tail_Number", "Reporting_Airline", "Origin", "Dest"]
            if col in available_cols
        }
        df = pd.read_csv(
            file_path, usecols=available_cols, dtype=dtype_spec, low_memory=False
        )
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"Error loading raw data: {e}")
        sys.exit(1)


def clean_data(df):
    """Basic cleaning: Handle NaNs in key cols, filter cancelled/diverted."""
    print("Cleaning data...")
    initial_rows = len(df)

    # 1. Drop rows with NaNs in absolutely critical columns for chaining/labeling
    critical_cols_check = [
        "Tail_Number",
        "FlightDate",
        "CRSDepTime",
        "CRSArrTime",
        config.TARGET_COL_RAW,
        "Origin",
        "Dest",
        "DepTime",
        "ArrTime",
        "CRSElapsedTime",
    ]
    # Filter list based on columns actually loaded
    critical_cols_check = [col for col in critical_cols_check if col in df.columns]
    print(f"Checking NaNs in critical columns: {critical_cols_check}")
    df.dropna(subset=critical_cols_check, inplace=True)
    print(f"Dropped {initial_rows - len(df)} rows with NaNs in critical columns.")
    current_rows = len(df)

    # 2. Filter cancelled/diverted flights
    if "Cancelled" in df.columns and "Diverted" in df.columns:
        df["Cancelled"] = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0)
        df["Diverted"] = pd.to_numeric(df["Diverted"], errors="coerce").fillna(0)
        df = df[(df["Cancelled"] != 1.0) & (df["Diverted"] != 1.0)]
        print(f"Removed {current_rows - len(df)} cancelled/diverted flights.")
    else:
        print("Warning: Cancelled/Diverted columns not found, skipping filter.")

    if df.empty:
        print("Error: Data empty after cleaning.")
        sys.exit(1)
    return df.reset_index(drop=True)


def engineer_features(df):
    """Selects, creates, encodes, and scales features."""
    print(f"Engineering features...")
    features_df = df.copy()

    # --- 1. Create Reliable Datetime Columns for Sorting and Validation ---
    print("Parsing and creating datetime columns...")
    # ... (datetime parsing and overnight handling remains the same) ...
    try:
        features_df["FlightDate_dt"] = pd.to_datetime(
            features_df["FlightDate"], errors="coerce"
        ).dt.date
        features_df["CRSDepTime_t"] = parse_hhmm(features_df["CRSDepTime"])
        features_df["CRSArrTime_t"] = parse_hhmm(features_df["CRSArrTime"])
        features_df["DepTime_t"] = parse_hhmm(features_df["DepTime"])
        features_df["ArrTime_t"] = parse_hhmm(features_df["ArrTime"])
    except Exception as e:
        print(f"Error during basic time parsing: {e}")
        sys.exit(1)

    features_df["SchedDepDateTime"] = features_df.apply(
        combine_date_time, axis=1, date_col="FlightDate_dt", time_col="CRSDepTime_t"
    )
    features_df["SchedArrDateTime"] = features_df.apply(
        combine_date_time, axis=1, date_col="FlightDate_dt", time_col="CRSArrTime_t"
    )
    features_df["ActualDepDateTime"] = features_df.apply(
        combine_date_time, axis=1, date_col="FlightDate_dt", time_col="DepTime_t"
    )
    features_df["ActualArrDateTime"] = features_df.apply(
        combine_date_time, axis=1, date_col="FlightDate_dt", time_col="ArrTime_t"
    )

    datetime_cols_check = [
        "SchedDepDateTime",
        "SchedArrDateTime",
        "ActualDepDateTime",
        "ActualArrDateTime",
    ]
    initial_rows = len(features_df)
    features_df.dropna(subset=datetime_cols_check, inplace=True)
    print(
        f"Dropped {initial_rows - len(features_df)} rows with invalid datetime parsing/combination."
    )
    if features_df.empty:
        print("Error: No valid rows after datetime processing.")
        sys.exit(1)

    for sched_col, base_col in [
        ("SchedArrDateTime", "SchedDepDateTime"),
        ("ActualArrDateTime", "ActualDepDateTime"),
    ]:
        overnight_mask = features_df[sched_col] < features_df[base_col]
        features_df.loc[overnight_mask, sched_col] += pd.Timedelta(days=1)
        # print(f"Adjusted {overnight_mask.sum()} arrivals in '{sched_col}' for overnight flights.") # Less verbose logging

    features_df["DateTime_ToSortBy"] = features_df["SchedDepDateTime"]

    # --- 2. Feature Selection & Derivation ---
    numerical_features_base = [
        "DepDelayMinutes",
        "CRSElapsedTime",
        "ActualElapsedTime",
        "AirTime",
        "Distance",
        "TaxiOut",
        "TaxiIn",
    ]
    numerical_features = [
        col for col in numerical_features_base if col in features_df.columns
    ]

    features_df["Month"] = features_df["SchedDepDateTime"].dt.month
    features_df["DayOfMonth"] = features_df["SchedDepDateTime"].dt.day
    features_df["DayOfWeek"] = features_df["SchedDepDateTime"].dt.dayofweek
    features_df["Hour"] = features_df["SchedDepDateTime"].dt.hour
    temporal_features = ["Month", "DayOfMonth", "DayOfWeek", "Hour"]

    categorical_features = [
        "Reporting_Airline",
        "Origin",
        "Dest",
    ]
    categorical_features = [
        col for col in categorical_features if col in features_df.columns
    ]

    features_to_process = numerical_features + temporal_features + categorical_features
    print(f"Columns identified for processing into features: {features_to_process}")

    # *** CORRECTION HERE: Ensure ALL columns needed later are kept ***
    # Columns needed for processing AND columns needed for chaining/validation/labeling
    required_cols_later = [
        "Tail_Number",
        "DateTime_ToSortBy",
        config.TARGET_COL_RAW,
        "SchedArrDateTime",
        "ActualDepDateTime",
        "ActualArrDateTime",
    ]  # Ensure ActualArrDateTime is here
    cols_to_keep_initial = list(set(required_cols_later + features_to_process))

    # Check if all initially kept columns exist
    missing_keep_cols = [
        col for col in cols_to_keep_initial if col not in features_df.columns
    ]
    if missing_keep_cols:
        print(
            f"Error: Columns needed for processing/chaining are missing *before* feature engineering: {missing_keep_cols}"
        )
        sys.exit(1)

    # Select these columns initially
    features_df = features_df[cols_to_keep_initial].copy()

    # --- 3. Encoding Categorical Features ---
    print(f"Encoding categorical columns: {categorical_features}")
    for col in categorical_features:
        features_df[col] = features_df[col].fillna("__MISSING__").astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    features_df[categorical_features] = encoder.fit_transform(
        features_df[categorical_features]
    )
    encoder_categories = {
        col: cats.tolist()
        for col, cats in zip(categorical_features, encoder.categories_)
    }

    # --- 4. Scaling Numerical Features ---
    numerical_to_scale = numerical_features + temporal_features + categorical_features
    print(f"Scaling numerical/encoded columns: {numerical_to_scale}")
    for col in numerical_to_scale:
        if features_df[col].isnull().any():
            median_val = features_df[col].median()
            # print(f"Filling {features_df[col].isnull().sum()} NaNs in '{col}' with median {median_val}") # Less verbose
            features_df[col].fillna(median_val, inplace=True)
    scaler = MinMaxScaler()
    features_df[numerical_to_scale] = scaler.fit_transform(
        features_df[numerical_to_scale]
    )
    scaler_params = {
        "min": scaler.min_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": numerical_to_scale,
    }

    # --- 5. Define Final Feature Set ---
    # THESE are the columns that will form the input tensor for the model
    final_feature_cols = numerical_to_scale  # The final features are the scaled numerical/temporal/encoded categoricals
    num_final_features = len(final_feature_cols)
    print(
        f"Final features for model input ({num_final_features}): {final_feature_cols}"
    )

    # Store stats
    data_stats = {
        "num_features": num_final_features,
        "feature_names": final_feature_cols,
        "encoder_categories": encoder_categories,
        "scaler_params": scaler_params,
    }

    # Return the DataFrame (which still contains the necessary ID/Time columns for chaining)
    # along with the list of columns that constitute the *final features* for the model.
    return features_df, final_feature_cols, data_stats


def create_chains(df, feature_cols, target_col, stats):
    """Constructs sequences of 3 consecutive flights per aircraft with validation."""
    print("Constructing flight chains...")
    num_features = stats["num_features"]

    # Ensure datetimes are correct type
    required_dt_cols = [
        "DateTime_ToSortBy",
        "SchedArrDateTime",
        "ActualDepDateTime",
        "ActualArrDateTime",
    ]
    for col in required_dt_cols:
        if col not in df.columns:
            print(f"Error: Required datetime column '{col}' missing.")
            sys.exit(1)
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception as e:
            print(f"Error converting column '{col}' to datetime: {e}")
            sys.exit(1)
    df.dropna(subset=required_dt_cols, inplace=True)
    if df.empty:
        print("Error: DataFrame empty after ensuring valid datetimes.")
        sys.exit(1)

    # Sort by aircraft and time
    df = df.sort_values(by=["Tail_Number", "DateTime_ToSortBy"]).reset_index(drop=True)

    print("Extracting columns to NumPy arrays for faster processing...")
    tail_numbers = df["Tail_Number"].values
    actual_dep_times_sec = df["ActualDepDateTime"].values.astype(np.int64) // 10**9
    actual_arr_times_sec = df["ActualArrDateTime"].values.astype(np.int64) // 10**9
    features_np = df[feature_cols].values.astype(np.float32)
    target_np = df[target_col].values

    max_ground_seconds = (
        config.MAX_GROUND_TIME.total_seconds()
        if config.MAX_GROUND_TIME is not None
        else None
    )
    min_turnaround_sec = 15 * 60  # Example: 15 minutes minimum turnaround

    chains = []
    labels = []
    skipped_validation = 0
    processed_groups_count = 0
    total_processed_flights = 0

    start_idx_group = 0  # Start index of the current group being tracked

    print("Constructing chains with validation (Revised Loop)...")
    # Iterate up to len(df) to handle the last group after the loop
    for i in range(len(df) + 1):  # Iterate one index beyond the end
        # Determine if the current group should end
        # It ends if we are past the last element OR if the tail number changes
        should_process_group = False
        if i == len(df):  # Reached the end of the dataframe
            should_process_group = True
        elif i > 0 and tail_numbers[i] != tail_numbers[i - 1]:  # Tail number changed
            should_process_group = True

        if should_process_group:
            # Define the group that *just ended*
            # Previous group ran from start_idx_group up to (but not including) index i
            end_idx_group = i
            group_len = end_idx_group - start_idx_group

            if group_len >= config.CHAIN_LENGTH:
                processed_groups_count += 1
                total_processed_flights += group_len
                # Sliding window over the indices of this completed group
                for j in range(
                    start_idx_group, end_idx_group - config.CHAIN_LENGTH + 1
                ):
                    idx_f1, idx_f2, idx_f3 = j, j + 1, j + 2

                    # --- Validation Logic (using NumPy arrays) ---
                    valid_chain = True
                    # Check 1: Actual Dep f2 > Actual Arr f1 + Min Turnaround
                    if actual_dep_times_sec[idx_f2] <= (
                        actual_arr_times_sec[idx_f1] + min_turnaround_sec
                    ):
                        valid_chain = False
                    # Check 2: Actual Dep f3 > Actual Arr f2 + Min Turnaround
                    if valid_chain and actual_dep_times_sec[idx_f3] <= (
                        actual_arr_times_sec[idx_f2] + min_turnaround_sec
                    ):
                        valid_chain = False
                    # Check 3: Max Ground Time (Actual Arr to Actual Dep)
                    if valid_chain and max_ground_seconds is not None:
                        ground_time12_sec = (
                            actual_dep_times_sec[idx_f2] - actual_arr_times_sec[idx_f1]
                        )
                        ground_time23_sec = (
                            actual_dep_times_sec[idx_f3] - actual_arr_times_sec[idx_f2]
                        )
                        if (
                            ground_time12_sec > max_ground_seconds
                            or ground_time23_sec > max_ground_seconds
                        ):
                            valid_chain = False
                    # --- End Validation Logic ---

                    if not valid_chain:
                        skipped_validation += 1
                        continue

                    # If valid, create chain and label
                    chain = features_np[j : j + config.CHAIN_LENGTH]
                    label_raw = target_np[idx_f3]
                    try:
                        label_class = pd.cut(
                            [label_raw],
                            bins=config.DELAY_THRESHOLDS,
                            labels=False,
                            right=True,
                        )[0]
                        if pd.isna(label_class):
                            if label_raw <= config.DELAY_THRESHOLDS[1]:
                                label_class = 0
                            else:
                                label_class = config.NUM_CLASSES - 1
                    except Exception as e:
                        print(
                            f"\nError classifying delay {label_raw} for index {j}: {e}"
                        )
                        skipped_validation += 1
                        continue
                    chains.append(chain)
                    labels.append(int(label_class))
            # else: # Group too short, do nothing (count skipped later)
            #     pass

            # If we haven't reached the end of the dataframe, start the next group
            if i < len(df):
                start_idx_group = i  # The current index starts the new group

    # --- Reporting ---
    total_tails_in_data = df["Tail_Number"].nunique() if not df.empty else 0
    skipped_short = total_tails_in_data - processed_groups_count

    print(f"\nTotal flights processed (after NaN drop): {len(df)}")
    print(f"Total unique tail numbers found: {total_tails_in_data}")
    print(
        f"Tail number groups processed (>= {config.CHAIN_LENGTH} flights): {processed_groups_count}"
    )
    print(f"Skipped {skipped_short} tail numbers with < {config.CHAIN_LENGTH} flights.")
    print(
        f"Skipped {skipped_validation} potential chains due to time validation failures."
    )

    if not chains:
        print("Warning: No valid chains were constructed after validation.")
        return np.array([]).reshape(0, config.CHAIN_LENGTH, num_features), np.array([])

    print(f"Constructed {len(chains)} valid chains.")
    return np.array(chains, dtype=np.float32), np.array(labels, dtype=np.int64)


def save_processed_data(chains, labels, stats):
    if len(chains) == 0 or len(labels) == 0:
        print("No chains or labels to save.")
        return
    print(f"Splitting {len(chains)} chains into Train/Validation/Test sets...")
    indices = np.arange(len(chains))
    valid_stratify = True
    stratify_split = None
    try:
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_samples_per_class_needed = 2
        if len(unique_labels) < config.NUM_CLASSES:
            print(
                f"Warning: Only {len(unique_labels)} classes present. Cannot fully stratify."
            )
            valid_stratify = False
        for count in counts:
            if count < min_samples_per_class_needed * 2:
                print(
                    f"Warning: A class has only {count} samples. Stratification unreliable."
                )
                valid_stratify = False
                break
        stratify_split = labels if valid_stratify else None
        if not valid_stratify:
            print("Proceeding with non-stratified split.")
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=config.TEST_SPLIT_RATIO,
            random_state=config.RANDOM_STATE,
            stratify=stratify_split,
        )
        relative_val_size = config.VAL_SPLIT_RATIO / (1 - config.TEST_SPLIT_RATIO)
        stratify_tv_split = (
            labels[train_val_indices] if stratify_split is not None else None
        )
        if stratify_tv_split is not None:
            tv_unique_labels, tv_counts = np.unique(
                stratify_tv_split, return_counts=True
            )
            if len(tv_unique_labels) < len(unique_labels):
                print(
                    "Warn: Not all classes in train/val subset. Using non-stratified for second split."
                )
                stratify_tv_split = None
            else:
                for count in tv_counts:
                    if count < min_samples_per_class_needed:
                        print(
                            f"Warn: Class in train/val subset has only {count} samples. Using non-stratified for second split."
                        )
                        stratify_tv_split = None
                        break
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=relative_val_size,
            random_state=config.RANDOM_STATE,
            stratify=stratify_tv_split,
        )
    except ValueError as e:
        print(
            f"Error during stratified split: {e}. Falling back to non-stratified split."
        )
        train_val_indices, test_indices = train_test_split(
            indices, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_STATE
        )
        relative_val_size = config.VAL_SPLIT_RATIO / (1 - config.TEST_SPLIT_RATIO)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=relative_val_size,
            random_state=config.RANDOM_STATE,
        )

    train_chains, train_labels = chains[train_indices], labels[train_indices]
    val_chains, val_labels = chains[val_indices], labels[val_indices]
    test_chains, test_labels = chains[test_indices], labels[test_indices]
    print(f"Train set: {len(train_chains)} chains")
    print(f"Validation set: {len(val_chains)} chains")
    print(f"Test set: {len(test_chains)} chains")

    # NEW: Balance the training set if the flag is enabled.
    if hasattr(config, "BALANCED") and config.BALANCED:
        print("Performing oversampling to balance the training set...")
        from collections import Counter

        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        max_count = class_counts.max()
        oversampled_chains = []
        oversampled_labels = []
        for cls in unique_classes:
            cls_indices = np.where(train_labels == cls)[0]
            n_samples = len(cls_indices)
            n_to_add = max_count - n_samples
            oversampled_chains.append(train_chains[cls_indices])
            oversampled_labels.append(train_labels[cls_indices])
            if n_to_add > 0:
                add_indices = np.random.choice(cls_indices, size=n_to_add, replace=True)
                oversampled_chains.append(train_chains[add_indices])
                oversampled_labels.append(train_labels[add_indices])
        train_chains = np.concatenate(oversampled_chains, axis=0)
        train_labels = np.concatenate(oversampled_labels, axis=0)
        shuf_indices = np.arange(len(train_labels))
        np.random.shuffle(shuf_indices)
        train_chains = train_chains[shuf_indices]
        train_labels = train_labels[shuf_indices]
        print(f"Balanced training set size: {len(train_labels)}")

    print("Saving processed data...")
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.TRAIN_CHAINS_FILE, train_chains)
    np.save(config.TRAIN_LABELS_FILE, train_labels)
    np.save(config.VAL_CHAINS_FILE, val_chains)
    np.save(config.VAL_LABELS_FILE, val_labels)
    np.save(config.TEST_CHAINS_FILE, test_chains)
    np.save(config.TEST_LABELS_FILE, test_labels)
    print("Saving data statistics...")
    try:

        def convert_np_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_np_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_np_types(i) for i in obj]
            return obj

        serializable_stats = convert_np_types(stats)
        with open(config.DATA_STATS_FILE, "w") as f:
            json.dump(serializable_stats, f, indent=4)
    except Exception as e:
        print(f"Error saving data stats: {e}")
    print("Processed data saving complete.")


def run_chain_construction():
    print("--- Starting Chain Construction ---")
    df_raw = load_raw_data(config.RAW_DATA_FILE)
    df_clean = clean_data(df_raw)
    df_featured, final_cols, data_stats = engineer_features(df_clean)
    if config.TARGET_COL_RAW not in df_featured.columns:
        print(f"Error: Target column '{config.TARGET_COL_RAW}' not found.")
        sys.exit(1)
    if "DateTime_ToSortBy" not in df_featured.columns:
        print(f"Error: Sorting key 'DateTime_ToSortBy' not found.")
        sys.exit(1)
    if config.SUBSAMPLE_DATA < 1.0:
        print(
            f"!!! SUBSAMPLING featured data to {config.SUBSAMPLE_DATA*100:.1f}% BEFORE CHAIN CREATION !!!"
        )
        df_featured = df_featured.sample(
            frac=config.SUBSAMPLE_DATA, random_state=config.RANDOM_STATE
        ).copy()
        print(f"Using {len(df_featured)} rows for chain creation.")
        if df_featured.empty:
            print("Error: Data empty after subsampling.")
            sys.exit(1)
    chains_np, labels_np = create_chains(
        df_featured, final_cols, config.TARGET_COL_RAW, data_stats
    )
    save_processed_data(chains_np, labels_np, data_stats)
    print("--- Chain Construction Finished ---")


if __name__ == "__main__":
    run_chain_construction()
