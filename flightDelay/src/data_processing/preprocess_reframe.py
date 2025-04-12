# flightDelay/src/data_processing/preprocess_reframe.py
import pandas as pd
import numpy as np
import sys
import traceback
import os # Import os for path operations if needed

# --- Path Setup ---
# Try relative import first
try:
    from . import config
except ImportError:
    # Handle case where script might be run directly or paths are different
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir) # Up one level to src/
    sys.path.insert(0, src_dir) # Add src to path
    try:
        import config
    except ImportError:
        print("CRITICAL: Cannot find 'config.py'. Ensure it's in the 'data_processing' directory or adjust sys.path.")
        sys.exit(1)

def load_data(file_path):
    """Loads the merged CSV data."""
    print(f"Loading data from: {file_path}")
    if not file_path.exists():
        print(f"Error: Input file not found at {file_path}")
        sys.exit(1)
    try:
        # Specify dtypes for key columns to prevent issues
        # *** THIS IS THE CORRECTED SECTION ***
        dtype_spec = {
            config.TAIL_NUM_COL: str,
            config.CARRIER_CODE_COL: str, # Use the correct variable from config.py
            config.FLIGHT_NUM_COL: str,
            config.ORIGIN_COL: str,
            config.DEST_COL: str,
            # Make dtype specification robust against missing columns
            # Only specify dtypes for columns expected to exist after merging
        }
        # Handle potential missing Cancelled/Diverted columns more gracefully
        if config.CANCELLED_COL in pd.read_csv(file_path, nrows=1).columns:
             dtype_spec[config.CANCELLED_COL] = float
        if config.DIVERTED_COL in pd.read_csv(file_path, nrows=1).columns:
             dtype_spec[config.DIVERTED_COL] = float

        # Load the actual data
        df = pd.read_csv(file_path, low_memory=False, dtype=dtype_spec)
        print(f"Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

def get_top_10_airports(weather_data_path):
    """Loads weather data and extracts unique airport codes."""
    top_airports = set()
    weather_merged_file = config.DATA_DIR / 'top10_weather_merged.csv' # Assuming it exists from weatherDelay merge
    if not weather_merged_file.exists():
        print(f"Warning: Merged weather file not found at {weather_merged_file}. Cannot filter by top 10 airports.")
        # Decide how to proceed: return None, empty set, or raise error?
        # Returning None will skip filtering later.
        return None
    try:
        print(f"Loading {weather_merged_file.name} to identify top 10 airports...")
        # Only need the 'airport' column
        weather_df = pd.read_csv(weather_merged_file, usecols=['airport'], low_memory=False)
        # Ensure codes are uppercase and stripped, matching flight data format
        top_airports = set(weather_df['airport'].astype(str).str.strip().str.upper().unique())
        print(f"Identified top airports for filtering: {top_airports}")
        return top_airports
    except Exception as e:
        print(f"Error reading weather file to get airports: {e}. Skipping airport filter.")
        return None

def basic_clean(df):
    """Performs basic cleaning based on config columns and paper description."""
    print("Starting basic cleaning...")
    initial_rows = len(df)
    top_10_airport_list = get_top_10_airports(config.DATA_DIR) 
    if top_10_airport_list:
        print(f"Filtering flights to include only those involving top airports: {top_10_airport_list}")
        rows_before_filter = len(df)
        # Keep flights where EITHER Origin OR Destination is in the list
        df = df[
            df[config.ORIGIN_COL].isin(top_10_airport_list) |
            df[config.DEST_COL].isin(top_10_airport_list)
        ].copy() # Use .copy() after filtering
        print(f"Removed {rows_before_filter - len(df)} flights not involving top 10 airports.")
        if df.empty:
            print("Error: No flights remaining after filtering for top 10 airports.")
            sys.exit(1)
    else:
        print("Skipping top 10 airport filtering step.")
        
    # 1. Drop exact duplicates
    current_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows.")
    current_rows = len(df)

    # 2. Filter Cancelled and Diverted flights (if columns exist)
    cancelled_exists = config.CANCELLED_COL in df.columns
    diverted_exists = config.DIVERTED_COL in df.columns

    if cancelled_exists and diverted_exists:
        df[config.CANCELLED_COL] = pd.to_numeric(df[config.CANCELLED_COL], errors='coerce')
        df[config.DIVERTED_COL] = pd.to_numeric(df[config.DIVERTED_COL], errors='coerce')
        # Keep rows where Cancelled is NOT 1.0 AND Diverted is NOT 1.0
        # Handle potential NaNs introduced by coerce - treat NaN as not cancelled/diverted? Or drop?
        # Safest might be to keep if not explicitly 1.0
        df = df[(df[config.CANCELLED_COL] != 1.0) & (df[config.DIVERTED_COL] != 1.0)]
        print(f"Removed {current_rows - len(df)} cancelled (==1.0) or diverted (==1.0) flights.")
        current_rows = len(df)
    elif cancelled_exists:
        df[config.CANCELLED_COL] = pd.to_numeric(df[config.CANCELLED_COL], errors='coerce')
        df = df[df[config.CANCELLED_COL] != 1.0]
        print(f"Removed {current_rows - len(df)} cancelled flights. ('{config.DIVERTED_COL}' not found).")
        current_rows = len(df)
    elif diverted_exists:
        df[config.DIVERTED_COL] = pd.to_numeric(df[config.DIVERTED_COL], errors='coerce')
        df = df[df[config.DIVERTED_COL] != 1.0]
        print(f"Removed {current_rows - len(df)} diverted flights. ('{config.CANCELLED_COL}' not found).")
        current_rows = len(df)
    else:
        print(f"Warning: '{config.CANCELLED_COL}' or '{config.DIVERTED_COL}' column not found, skipping this filter.")


    # 3. Define essential columns based on Table 1 + FDPP-ML needs
    essential_cols_fdpp = [
        config.FLIGHT_DATE_COL, config.TAIL_NUM_COL,
        config.SCHED_DEP_TIME_COL, config.SCHED_ARR_TIME_COL,
        config.DEP_DELAY_COL, config.ARR_DELAY_COL,
        config.ORIGIN_COL, config.DEST_COL,
        config.CARRIER_CODE_COL, config.FLIGHT_NUM_COL
    ]
    # Keep only essential columns that actually exist in the input dataframe
    cols_to_keep = [col for col in essential_cols_fdpp if col in df.columns]

    # Check if any columns absolutely required for the core logic are missing
    missing_essentials = set([
        config.FLIGHT_DATE_COL, config.TAIL_NUM_COL, config.SCHED_DEP_TIME_COL,
        config.SCHED_ARR_TIME_COL, config.DEP_DELAY_COL, config.ARR_DELAY_COL,
        config.ORIGIN_COL, config.DEST_COL, config.CARRIER_CODE_COL
    ]) - set(df.columns)
    if missing_essentials:
         print(f"CRITICAL ERROR: Essential columns missing from data needed for FDPP-ML: {missing_essentials}")
         sys.exit(1)

    df = df[cols_to_keep].copy() # Select columns

    # 4. Drop rows with NaN in columns critical for path reconstruction and target variable
    critical_na_check = [
        config.TAIL_NUM_COL, config.FLIGHT_DATE_COL,
        config.SCHED_DEP_TIME_COL, config.SCHED_ARR_TIME_COL,
        config.ORIGIN_COL, config.DEST_COL,
        config.DEP_DELAY_COL, # Need delays for the target variable
        config.ARR_DELAY_COL
    ]
    # Filter check list based on actual columns present (should exist based on check above)
    critical_na_check = [col for col in critical_na_check if col in df.columns]
    df.dropna(subset=critical_na_check, inplace=True)
    print(f"Removed {current_rows - len(df)} rows with missing critical data ({critical_na_check}).")
    current_rows = len(df)

    # 5. Clean specific identifiers (Tail Number, Origin, Dest)
    df[config.TAIL_NUM_COL] = df[config.TAIL_NUM_COL].astype(str).str.strip().str.upper()
    invalid_tails = ['', '0', 'NONE', 'UNKN', 'UNKNOW', 'UNKNOWN', 'XXXXXX']
    df = df[~df[config.TAIL_NUM_COL].isin(invalid_tails)]
    df = df[df[config.TAIL_NUM_COL].notna()] # Ensure notna again after potential empty strings

    df[config.ORIGIN_COL] = df[config.ORIGIN_COL].astype(str).str.strip().str.upper()
    df = df[df[config.ORIGIN_COL].notna() & (df[config.ORIGIN_COL] != '')]
    df[config.DEST_COL] = df[config.DEST_COL].astype(str).str.strip().str.upper()
    df = df[df[config.DEST_COL].notna() & (df[config.DEST_COL] != '')]

    print(f"Removed {current_rows - len(df)} rows with invalid Tail Numbers/Origin/Dest.")
    current_rows = len(df) # Update count after cleaning identifiers

    if len(df) == 0:
        print("Error: No valid data remaining after basic cleaning.")
        sys.exit(1)

    print(f"Basic cleaning finished. Rows remaining: {len(df)}")
    return df

def parse_times_and_create_datetime(df):
    """Parses HHMM times and combines with FlightDate to create DateTime objects."""
    print("Parsing times and creating DateTime objects...")

    # 1. Convert FlightDate to datetime objects (date part only initially)
    try:
        # Attempt conversion, coercing errors to NaT (Not a Time)
        df['FlightDate_dt'] = pd.to_datetime(df[config.FLIGHT_DATE_COL], errors='coerce').dt.date
        rows_before_drop = len(df)
        df.dropna(subset=['FlightDate_dt'], inplace=True) # Drop rows where conversion failed
        if rows_before_drop > len(df):
             print(f"Dropped {rows_before_drop - len(df)} rows with invalid FlightDate format.")
        if df.empty:
            print("Error: No valid FlightDate entries remaining after parsing."); sys.exit(1)
    except KeyError:
        print(f"Error: Column '{config.FLIGHT_DATE_COL}' not found for parsing.")
        sys.exit(1)
    except Exception as e: # Catch other potential errors during conversion
        print(f"Error converting '{config.FLIGHT_DATE_COL}' to datetime: {e}")
        sys.exit(1)

    # 2. Define function to parse HHMM time strings/floats safely
    def parse_hhmm(time_col_series):
        # Handle potential floats (e.g., 830.0), convert to int string, zfill
        # Add check for NaN before conversion
        if time_col_series.isnull().any():
            time_col_series = time_col_series.fillna(-1) # Temp fill NaN to avoid error in astype(int)

        time_str = time_col_series.astype(float).astype(int).astype(str).str.zfill(4)
        # Handle '2400' -> '0000'
        time_str = time_str.replace('2400', '0000')
        # Revert temp fill to NaN so invalid times become NaT
        time_str = time_str.replace('-001', pd.NA)
        # Validate format HHMM (0000-2359) - important!
        valid_time_mask = time_str.str.match(r'^([01]\d|2[0-3])([0-5]\d)$', na=False) # na=False to treat NA as non-match
        # Convert valid times to time objects, invalid to NaT
        return pd.to_datetime(time_str.where(valid_time_mask), format='%H%M', errors='coerce').dt.time

    # 3. Parse Scheduled Departure and Arrival Times
    all_times_parsed = True
    for time_col_in, time_col_out in [
        (config.SCHED_DEP_TIME_COL, 'SchedDepTime_parsed'),
        (config.SCHED_ARR_TIME_COL, 'SchedArrTime_parsed')
    ]:
        if time_col_in in df.columns:
            df[time_col_out] = parse_hhmm(df[time_col_in])
        else:
             print(f"Warning: Column {time_col_in} not found. Cannot parse.")
             df[time_col_out] = pd.NaT # Assign NaT if column missing
             all_times_parsed = False # Mark that not all necessary times could be parsed

    if not all_times_parsed:
        print("Error: Could not parse all required scheduled times (Dep/Arr). Cannot proceed.")
        sys.exit(1)

    # 4. Drop rows where essential parsed times are NaT
    rows_before_drop = len(df)
    df.dropna(subset=['SchedDepTime_parsed', 'SchedArrTime_parsed'], inplace=True)
    if rows_before_drop > len(df):
         print(f"Dropped {rows_before_drop - len(df)} rows with invalid/unparseable scheduled time formats.")
    if df.empty:
        print("Error: No rows remaining after dropping invalid time formats."); sys.exit(1)

    # 5. Combine Date and Parsed Time into DateTime objects
    def combine_date_time(row, date_col, time_col):
        # Check for NaT/None in inputs before combining
        if pd.isna(row[date_col]) or pd.isna(row[time_col]):
            return pd.NaT
        try:
            # Combine using datetime.combine
            return pd.Timestamp.combine(row[date_col], row[time_col])
        except (TypeError, ValueError) as e:
            # print(f"Debug: Combine error for date {row[date_col]}, time {row[time_col]}: {e}") # Optional debug
            return pd.NaT # Return NaT if combination fails

    # Apply the combination function
    df[config.SCHED_DATETIME_COL + '_Dep'] = df.apply(
        combine_date_time, axis=1, date_col='FlightDate_dt', time_col='SchedDepTime_parsed'
    )
    df[config.SCHED_DATETIME_COL + '_Arr'] = df.apply(
        combine_date_time, axis=1, date_col='FlightDate_dt', time_col='SchedArrTime_parsed'
    )

    # 6. Handle overnight flights for scheduled arrival datetime
    # Ensure columns are actual datetime objects before comparison
    df[config.SCHED_DATETIME_COL + '_Dep'] = pd.to_datetime(df[config.SCHED_DATETIME_COL + '_Dep'], errors='coerce')
    df[config.SCHED_DATETIME_COL + '_Arr'] = pd.to_datetime(df[config.SCHED_DATETIME_COL + '_Arr'], errors='coerce')

    # Drop rows where conversion to full datetime failed
    rows_before_drop = len(df)
    df.dropna(subset=[config.SCHED_DATETIME_COL + '_Dep', config.SCHED_DATETIME_COL + '_Arr'], inplace=True)
    if rows_before_drop > len(df):
         print(f"Dropped {rows_before_drop - len(df)} rows where full datetime conversion failed (Dep/Arr).")
    if df.empty: print("Error: No valid schedule datetimes remaining after combination."); sys.exit(1)


    # If Arr DateTime is earlier than Dep DateTime, add one day to Arr DateTime
    overnight_mask = df[config.SCHED_DATETIME_COL + '_Arr'] < df[config.SCHED_DATETIME_COL + '_Dep']
    df.loc[overnight_mask, config.SCHED_DATETIME_COL + '_Arr'] += pd.Timedelta(days=1)
    print(f"Adjusted {overnight_mask.sum()} scheduled arrival times for overnight flights.")

    # 7. Final drop for any remaining NaT datetimes (should be minimal now)
    rows_before_drop = len(df)
    df.dropna(subset=[config.SCHED_DATETIME_COL + '_Dep', config.SCHED_DATETIME_COL + '_Arr'], inplace=True)
    if rows_before_drop > len(df):
         print(f"Dropped {rows_before_drop - len(df)} additional rows with NaT datetimes.")

    # Clean up intermediate columns
    df = df.drop(columns=['FlightDate_dt', 'SchedDepTime_parsed', 'SchedArrTime_parsed'], errors='ignore')

    if df.empty: print("Error: No valid flights after datetime creation."); sys.exit(1)
    print("DateTime parsing and creation finished.")
    return df

def prepare_delays(df):
    """Ensures delay columns are numeric and handles NaNs (fills with 0)."""
    print("Preparing delay columns...")
    delay_cols_to_process = [config.DEP_DELAY_COL, config.ARR_DELAY_COL]
    all_delays_present = True
    for delay_col in delay_cols_to_process:
        if delay_col in df.columns:
            initial_nan_count = df[delay_col].isnull().sum()
            df[delay_col] = pd.to_numeric(df[delay_col], errors='coerce')
            # Check NaNs created by coercion + initial NaNs
            nan_count = df[delay_col].isnull().sum()
            if nan_count > 0:
                 print(f"Filling {nan_count} NaN/invalid values in '{delay_col}' with 0.")
                 df[delay_col].fillna(0, inplace=True)
        else:
            print(f"Error: Required delay column '{delay_col}' not found. Cannot proceed.")
            all_delays_present = False
            # sys.exit(1) # Exit immediately if critical delay info is missing

    if not all_delays_present:
         sys.exit(1) # Exit if any required delay col was missing

    return df

def reframe_to_points(df):
    """Implements Algorithm 1 logic using columns from config based on Table 1."""
    print("Reframing flights into Departure/Arrival points...")
    if df.empty:
        print("Input dataframe is empty, cannot reframe.")
        return df

    # Columns common to both points (Identifiers and flight details)
    common_cols_reframing = [
        config.CARRIER_CODE_COL, config.FLIGHT_NUM_COL, config.TAIL_NUM_COL,
        config.ORIGIN_COL, config.DEST_COL
    ]
    common_cols_reframing = [col for col in common_cols_reframing if col in df.columns]

    # Check if essential datetime and delay columns exist before proceeding
    if (config.SCHED_DATETIME_COL + '_Dep') not in df.columns or \
       (config.SCHED_DATETIME_COL + '_Arr') not in df.columns or \
       config.DEP_DELAY_COL not in df.columns or \
       config.ARR_DELAY_COL not in df.columns:
        print("Error: Missing necessary datetime or delay columns for reframing.")
        sys.exit(1)


    # --- Create Departure Points DataFrame ---
    df_dep = df[common_cols_reframing].copy()
    df_dep[config.ORIENTATION_COL] = 'Departure'
    df_dep[config.SCHED_DATETIME_COL] = df[config.SCHED_DATETIME_COL + '_Dep']
    df_dep[config.FLIGHT_DELAY_COL] = df[config.DEP_DELAY_COL]

    # --- Create Arrival Points DataFrame ---
    df_arr = df[common_cols_reframing].copy()
    df_arr[config.ORIENTATION_COL] = 'Arrival'
    df_arr[config.SCHED_DATETIME_COL] = df[config.SCHED_DATETIME_COL + '_Arr']
    df_arr[config.FLIGHT_DELAY_COL] = df[config.ARR_DELAY_COL]

    # --- Concatenate Departure and Arrival Points ---
    reframed_df = pd.concat([df_dep, df_arr], ignore_index=True)

    # --- Initialize FTD and PFD Columns ---
    reframed_df[config.FTD_COL] = 0.0
    reframed_df[config.PFD_COL] = 0.0

    # Drop rows where the crucial SCHED_DATETIME_COL ended up being NaT after concat
    rows_before_drop = len(reframed_df)
    reframed_df.dropna(subset=[config.SCHED_DATETIME_COL], inplace=True)
    if rows_before_drop > len(reframed_df):
        print(f"Dropped {rows_before_drop - len(reframed_df)} points with invalid Schedule_DateTime after reframing.")

    if reframed_df.empty:
        print("Error: No valid points remaining after reframing and NaT drop.")
        sys.exit(1)

    # --- Sort by Tail Number and Schedule DateTime (Crucial for next steps) ---
    reframed_df = reframed_df.sort_values(by=[config.TAIL_NUM_COL, config.SCHED_DATETIME_COL])
    reframed_df = reframed_df.reset_index(drop=True)

    print(f"Reframing complete. Created {len(reframed_df)} points (rows).")
    return reframed_df

def save_data(df, file_path):
    """Saves the processed dataframe."""
    print(f"Saving data to: {file_path}")
    if df is None or df.empty:
        print(f"Dataframe is empty or None. Skipping save for {file_path.name}.")
        return
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df_copy = df.copy() # Save a copy

        # Ensure datetime format is consistent for saving
        if config.SCHED_DATETIME_COL in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[config.SCHED_DATETIME_COL]):
                 df_copy[config.SCHED_DATETIME_COL] = df_copy[config.SCHED_DATETIME_COL].dt.strftime('%Y-%m-%d %H:%M:%S')
            else: # If not datetime, try converting before formatting
                 try:
                     df_copy[config.SCHED_DATETIME_COL] = pd.to_datetime(df_copy[config.SCHED_DATETIME_COL]).dt.strftime('%Y-%m-%d %H:%M:%S')
                 except Exception as fmt_e:
                      print(f"Warning: Could not format '{config.SCHED_DATETIME_COL}' for saving {file_path.name}: {fmt_e}. Saving as is.")

        df_copy.to_csv(file_path, index=False)
        print(f"Save complete for {file_path.name}. Shape: {df_copy.shape}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
        traceback.print_exc()
        # Decide if script should exit or continue
        # sys.exit(1)

def run_preprocess_reframe():
    """Orchestrates the loading, cleaning, parsing, and reframing steps."""
    df = load_data(config.MERGED_RAW_FLIGHTS)
    df_cleaned = basic_clean(df)
    df_parsed = parse_times_and_create_datetime(df_cleaned)
    df_delays_prepared = prepare_delays(df_parsed) # Renamed variable for clarity
    df_reframed = reframe_to_points(df_delays_prepared)

    if not df_reframed.empty:
        save_data(df_reframed, config.REFRAMED_POINTS)
        print("Preprocessing and reframing completed successfully.")
    else:
        print("Preprocessing and reframing failed: resulting dataframe is empty.")
        sys.exit(1) # Exit if reframing results in empty dataframe

if __name__ == "__main__":
    run_preprocess_reframe()
