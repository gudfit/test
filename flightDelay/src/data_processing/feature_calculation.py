# flightDelay/src/data_processing/feature_calculation.py
import pandas as pd
import numpy as np
import sys
import traceback
import os # Import os

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
    """Loads the reframed points data."""
    print(f"Loading reframed points data from: {file_path}")
    if not file_path.exists():
        print(f"Error: Input file not found at {file_path}")
        sys.exit(1)
    try:
        # Parse Schedule_DateTime back into datetime objects
        df = pd.read_csv(file_path, parse_dates=[config.SCHED_DATETIME_COL])
        print(f"Loaded {len(df)} points.")
        # Ensure it's sorted correctly (should be from previous step, but double-check)
        df = df.sort_values(by=[config.TAIL_NUM_COL, config.SCHED_DATETIME_COL]).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

def calculate_ftd(df):
    """Calculates Flight Time Duration (FTD) based on Algorithm 2, lines 1-9."""
    print("Calculating Flight Time Duration (FTD)...")
    if df.empty:
        print("Input DataFrame is empty, skipping FTD calculation.")
        return df

    if config.FTD_COL not in df.columns:
        df[config.FTD_COL] = 0.0 # Ensure column exists

    # Calculate time difference between consecutive points FOR THE SAME TAIL NUMBER
    # Use transform for potentially better performance on large groups
    df['Prev_Sched_DateTime'] = df.groupby(config.TAIL_NUM_COL)[config.SCHED_DATETIME_COL].transform(lambda x: x.shift(1))

    # Calculate the difference in minutes
    time_diff = df[config.SCHED_DATETIME_COL] - df['Prev_Sched_DateTime']

    # Convert timedelta to total minutes, fill NaT results (first points) with 0
    df[config.FTD_COL] = (time_diff.dt.total_seconds() / 60.0).fillna(0.0)

    # Clean up temporary column
    df = df.drop(columns=['Prev_Sched_DateTime'])

    # Ensure FTD is non-negative
    df[config.FTD_COL] = df[config.FTD_COL].clip(lower=0)

    print("FTD calculation complete.")
    print(f"FTD stats:\n{df[config.FTD_COL].describe()}")
    return df

def partition_data(df):
    """Partitions data into historical and future sets based on config.PARTITION_DATETIME."""
    print(f"Partitioning data based on timestamp <= {config.PARTITION_DATETIME}")
    if df.empty:
         print("Warning: Cannot partition empty dataframe.")
         return pd.DataFrame(), pd.DataFrame()

    historical_df = df[df[config.SCHED_DATETIME_COL] <= config.PARTITION_DATETIME].copy()
    future_df = df[df[config.SCHED_DATETIME_COL] > config.PARTITION_DATETIME].copy()

    print(f"Historical data points: {len(historical_df)}")
    print(f"Future data points: {len(future_df)}")

    if historical_df.empty:
        print("Warning: No historical data found based on partition time. Check PARTITION_DATETIME and data range.")
    if future_df.empty:
        print("Warning: No future data found based on partition time. Check PARTITION_DATETIME and data range.")

    return historical_df, future_df

def calculate_pfd_historical(historical_df):
    """Calculates Previous Flight Delay (PFD) for historical data (Algorithm 3, lines 1-6)."""
    print("Calculating Previous Flight Delay (PFD) for historical data...")
    if historical_df.empty:
         print("Historical dataframe is empty, skipping PFD calculation.")
         return historical_df

    if config.PFD_COL not in historical_df.columns:
        historical_df[config.PFD_COL] = 0.0 # Ensure column exists

    # PFD for point 'i' is the actual Flight_Delay of point 'i-1' within the same Tail_Number path
    # Use transform for potential performance gain
    historical_df[config.PFD_COL] = historical_df.groupby(config.TAIL_NUM_COL)[config.FLIGHT_DELAY_COL].transform(lambda x: x.shift(1))

    # Fill NaN values for the first point of each path with 0 (no previous delay)
    nan_pfd_count = historical_df[config.PFD_COL].isnull().sum()
    # Use direct assignment instead of inplace=True for modern pandas
    historical_df[config.PFD_COL] = historical_df[config.PFD_COL].fillna(0.0)
    print(f"Calculated PFD for historical data. Filled {nan_pfd_count} NaN PFD values (first points) with 0.")
    print(f"PFD (historical) stats:\n{historical_df[config.PFD_COL].describe()}")

    return historical_df

def save_data(df, file_path):
    """Saves the processed dataframe, formatting datetime."""
    print(f"Saving data to: {file_path}")
    if df is None or df.empty:
        print(f"Dataframe is empty or None. Skipping save for {file_path.name}.")
        return
    try:
        df_copy = df.copy() # Save a copy
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if config.SCHED_DATETIME_COL in df_copy.columns:
             if pd.api.types.is_datetime64_any_dtype(df_copy[config.SCHED_DATETIME_COL]):
                  df_copy[config.SCHED_DATETIME_COL] = df_copy[config.SCHED_DATETIME_COL].dt.strftime('%Y-%m-%d %H:%M:%S')
             else:
                  try: df_copy[config.SCHED_DATETIME_COL] = pd.to_datetime(df_copy[config.SCHED_DATETIME_COL]).dt.strftime('%Y-%m-%d %H:%M:%S')
                  except Exception as fmt_e: print(f"Warning: Could not format {config.SCHED_DATETIME_COL} for saving {file_path.name}: {fmt_e}")

        df_copy.to_csv(file_path, index=False)
        print(f"Save complete for {file_path.name}. Shape: {df_copy.shape}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
        traceback.print_exc()

def run_feature_calculation():
    """
    Orchestrates FTD calculation, partitioning, historical PFD calculation,
    and saves BOTH historical (with PFD) and initial future (with FTD, no PFD) data.
    """
    df_reframed = load_data(config.REFRAMED_POINTS)
    df_with_ftd = calculate_ftd(df_reframed)

    historical_df_initial, future_df_initial = partition_data(df_with_ftd)

    # Calculate PFD only for the historical partition
    historical_df_with_pfd = calculate_pfd_historical(historical_df_initial)

    # --- Save the final historical data WITH PFD calculated (used for training) ---
    save_data(historical_df_with_pfd, config.HISTORICAL_POINTS)

    # --- Save the initial future data partition (used as input for partition_prepare) ---
    # This dataframe has FTD calculated but PFD is still the initial 0.0
    # Define a new config variable for this intermediate future file
    INITIAL_FUTURE_FILE = config.PROCESSED_DATA_DIR / '3a_future_points_initial.csv' # New intermediate file name
    save_data(future_df_initial, INITIAL_FUTURE_FILE)

    print("Feature calculation (FTD & historical PFD) and partitioning completed.")
    print(f"Saved final historical data to: {config.HISTORICAL_POINTS}")
    print(f"Saved initial future data to: {INITIAL_FUTURE_FILE}")


if __name__ == "__main__":
    run_feature_calculation()
