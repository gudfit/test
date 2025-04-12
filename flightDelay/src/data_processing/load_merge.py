# flightDelay/src/data_processing/load_merge.py
import pandas as pd
import pathlib
import sys
import traceback
import glob # Use glob directly if needed

# Import config
try:
    from . import config
except ImportError: # Handle running script directly
    import config

def merge_csv_files(input_dir: pathlib.Path, file_pattern: str, output_file: pathlib.Path) -> bool:
    """
    Merges CSV files matching a given pattern within a specified directory
    into a single output CSV file.

    Args:
        input_dir: Path object for the directory containing input CSVs.
        file_pattern: Glob pattern to match input CSV files (relative to input_dir).
        output_file: Path object for the merged output CSV file.

    Returns:
        True if merge was successful, False otherwise.
    """
    print(f"\n--- Starting merge process ---")
    print(f"Input directory: {input_dir}")
    print(f"Looking for files matching: {file_pattern}")
    print(f"Output file: {output_file}")

    # Find the input files using glob directly for simplicity here
    # Alternatively, use input_dir.glob(file_pattern)
    search_path = str(input_dir / file_pattern)
    input_files = sorted(glob.glob(search_path))

    if not input_files:
        print(f"Error: No files found matching the pattern '{file_pattern}' in {input_dir}")
        # Provide specific expected file examples if helpful
        # print("Example expected files: On_Time_Reporting_..._2022_3.csv, ..._2022_6.csv, etc.")
        return False

    print(f"Found {len(input_files)} files to merge:")
    for f_path in input_files:
        print(f" - {pathlib.Path(f_path).name}")

    df_list = []
    read_error = False
    for file_path_str in input_files:
        file_path = pathlib.Path(file_path_str)
        print(f"Reading {file_path.name}...")
        try:
            # Read CSV, handle potential type inconsistencies
            # Specify dtypes for potentially problematic columns if known
            # Example: dtype={'Tail_Number': str, 'Flight_Number...': str}
            df = pd.read_csv(file_path, low_memory=False)

            # Drop empty 'Unnamed:' columns often added by spreadsheet software
            unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
            if unnamed_cols:
                # Check if ALL unnamed columns are entirely empty (all NaN)
                all_unnamed_empty = True
                for col in unnamed_cols:
                    if not df[col].isnull().all():
                        all_unnamed_empty = False
                        print(f"  -> Warning: Column '{col}' starts with 'Unnamed:' but contains data. Not dropping.")
                        break
                if all_unnamed_empty:
                    print(f"  -> Dropping empty unnamed columns: {unnamed_cols}")
                    df = df.drop(columns=unnamed_cols)

            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            read_error = True
            # Optional: break here if one file error should stop the merge
            # break

    if read_error:
        print("Errors occurred while reading some files. Aborting merge.")
        return False

    if not df_list:
        print("Error: No dataframes were successfully read. Cannot merge.")
        return False

    print("Concatenating dataframes...")
    try:
        merged_df = pd.concat(df_list, ignore_index=True)
    except pd.errors.InvalidIndexError as e:
         print(f"Error during concatenation, likely due to duplicate column names across files: {e}")
         # Add more diagnostics here if needed, e.g., print columns of each df
         return False
    except Exception as e:
         print(f"Error during concatenation: {e}")
         return False


    print(f"Saving merged data to: {output_file}")
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        merged_df.to_csv(output_file, index=False)
        print(f"Successfully merged files and saved '{output_file.name}'.")
        print(f"Merged DataFrame shape: {merged_df.shape}")
        return True
    except Exception as e:
        print(f"Error saving merged file to {output_file}: {e}")
        traceback.print_exc()
        return False

def run_merge():
    """Runs the merge operation using paths from config."""
    success = merge_csv_files(
        input_dir=config.DATA_DIR,
        file_pattern=config.RAW_FLIGHT_FILES_PATTERN,
        output_file=config.MERGED_RAW_FLIGHTS
    )
    if not success:
        print("Merging raw flight data failed.")
        sys.exit(1)
    else:
        print("Merging raw flight data completed successfully.")

if __name__ == "__main__":
    # This allows running the script directly for just merging
    run_merge()
