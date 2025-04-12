# flightDelay/src/data_processing/partition_prepare.py
import pandas as pd
import sys
import traceback
import os

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


def load_partitioned_data():
    """Loads the final historical data (with PFD) and the initial future data."""
    print(f"Loading final historical data from: {config.HISTORICAL_POINTS}")
    if not config.HISTORICAL_POINTS.exists():
        print(f"Error: Historical data file not found at {config.HISTORICAL_POINTS}")
        print("Ensure 'feature_calculation.py' ran successfully.")
        sys.exit(1)
    try:
        hist_df = pd.read_csv(config.HISTORICAL_POINTS, parse_dates=[config.SCHED_DATETIME_COL])
        print(f"Loaded {len(hist_df)} historical points.")
    except Exception as e:
        print(f"Error loading historical data: {e}"); sys.exit(1)

    # --- Load the CORRECT initial future file ---
    print(f"Loading initial future data from: {config.INITIAL_FUTURE_FILE}")
    if not config.INITIAL_FUTURE_FILE.exists():
         print(f"Error: Initial future data file not found at {config.INITIAL_FUTURE_FILE}")
         print("Ensure 'feature_calculation.py' ran successfully and saved this file.")
         sys.exit(1)
    try:
        future_df = pd.read_csv(config.INITIAL_FUTURE_FILE, parse_dates=[config.SCHED_DATETIME_COL])
        # Ensure PFD column exists (should be 0.0 from initial calculation)
        if config.PFD_COL not in future_df.columns and not future_df.empty:
            future_df[config.PFD_COL] = 0.0
        print(f"Loaded {len(future_df)} initial future points.")

    except Exception as e:
        print(f"Error loading initial future data: {e}"); sys.exit(1)

    # Basic validation
    if hist_df.empty and future_df.empty:
        print("Warning: Both loaded historical and future datasets are empty.")
    elif hist_df.empty:
        print("Warning: Loaded historical dataset is empty.")
    elif future_df.empty:
        print("Warning: Loaded initial future dataset is empty.")


    return hist_df, future_df

def extract_last_historical_points(historical_df):
    """
    Extracts the last point of each flight path (Tail Number group) from historical data.
    As per Algorithm 3, line 7.
    """
    print("Extracting last point for each historical path...")
    if historical_df.empty:
        print("Historical data is empty. Cannot extract last points.")
        return pd.DataFrame() # Return empty df

    # Ensure sorted (should be, but safer)
    historical_df = historical_df.sort_values(by=[config.TAIL_NUM_COL, config.SCHED_DATETIME_COL])

    # Use groupby().tail(1) to get the last row for each group efficiently
    last_points_df = historical_df.groupby(config.TAIL_NUM_COL).tail(1).copy()

    print(f"Extracted {len(last_points_df)} last points from {len(historical_df)} historical points.")
    return last_points_df


def merge_and_prepare_future(future_df, last_points_df):
    """
    Merges the extracted last historical points with the initial future data.
    Sorts the result. (Algorithm 3, line 10)
    """
    print("Merging last historical points with future data...")
    if future_df.empty and last_points_df.empty:
        print("Both future and last historical points are empty. Result is empty.")
        return pd.DataFrame()
    elif last_points_df.empty:
         print("No last historical points to merge. Using original future data.")
         # Still need to sort future data if it exists
         if not future_df.empty:
              future_df = future_df.sort_values(by=[config.TAIL_NUM_COL, config.SCHED_DATETIME_COL]).reset_index(drop=True)
         return future_df
    elif future_df.empty:
        print("Initial future data is empty. Using only last historical points.")
        # Sort the last points data
        last_points_df = last_points_df.sort_values(by=[config.TAIL_NUM_COL, config.SCHED_DATETIME_COL]).reset_index(drop=True)
        return last_points_df

    # Concatenate
    prepared_future_df = pd.concat([future_df, last_points_df], ignore_index=True)

    # Sort the combined future data by Tail Number and Schedule DateTime
    prepared_future_df = prepared_future_df.sort_values(by=[config.TAIL_NUM_COL, config.SCHED_DATETIME_COL])
    prepared_future_df = prepared_future_df.reset_index(drop=True)

    print(f"Merging complete. Prepared future dataset has {len(prepared_future_df)} points.")
    # Check PFD values - points from 'future_df' should have PFD=0 initially,
    # points from 'last_points_df' should have their calculated PFD.
    print("PFD stats in prepared future data (includes seeded historical PFDs):")
    if config.PFD_COL in prepared_future_df.columns:
        print(prepared_future_df[config.PFD_COL].describe())
    else:
        print("PFD column not found in prepared future data.")


    return prepared_future_df

def save_data(df, file_path):
    """Saves the processed dataframe, formatting datetime."""
    print(f"Saving data to: {file_path}")
    if df is None or df.empty:
        print(f"Dataframe is empty or None. Skipping save for {file_path.name}.")
        return
    try:
        df_copy = df.copy() # Work on a copy
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


def run_partition_prepare():
    """Orchestrates loading partitions and preparing the final future dataset."""
    historical_df, initial_future_df = load_partitioned_data()
    last_hist_points = extract_last_historical_points(historical_df)
    prepared_future = merge_and_prepare_future(initial_future_df, last_hist_points)

    # Save the final prepared future dataset (input for prediction step)
    save_data(prepared_future, config.FUTURE_PREPARED)

    print("Preparation of future dataset completed.")
    print(f"Saved prepared future data to: {config.FUTURE_PREPARED}")


if __name__ == "__main__":
    run_partition_prepare()
