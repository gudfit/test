# src/utils/data_loader.py
import pandas as pd
import glob
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the absolute path of the directory containing the current script
_CURRENT_DIR = Path(__file__).resolve().parent

def load_csv_files_from_dir(directory: Path, pattern: str, usecols: list = None) -> pd.DataFrame:
    """Loads and concatenates CSV files matching a pattern in a directory."""
    all_files = glob.glob(os.path.join(directory, pattern))
    if not all_files:
        logging.warning(f"No files found matching pattern '{pattern}' in directory '{directory}'")
        return pd.DataFrame()

    logging.info(f"Found {len(all_files)} files matching pattern '{pattern}'. Loading...")
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, low_memory=False, usecols=usecols) # low_memory=False for mixed types
            df_list.append(df)
            logging.debug(f"Loaded {f}")
        except Exception as e:
            logging.error(f"Error loading file {f}: {e}")
            continue # Skip problematic files

    if not df_list:
        logging.warning("No data loaded after processing files.")
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Successfully concatenated {len(df_list)} files into a DataFrame with {len(combined_df)} rows.")
    return combined_df

def save_dataframe(df: pd.DataFrame, file_path: Path, format: str = 'csv'):
    """Saves a DataFrame to a specified format (csv or parquet)."""
    try:
        if format == 'csv':
            df.to_csv(file_path, index=False)
            logging.info(f"DataFrame saved to {file_path} (CSV)")
        elif format == 'parquet':
            df.to_parquet(file_path, index=False)
            logging.info(f"DataFrame saved to {file_path} (Parquet)")
        else:
            logging.error(f"Unsupported save format: {format}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to {file_path}: {e}")

def load_dataframe(file_path: Path, format: str = 'csv') -> pd.DataFrame:
    """Loads a DataFrame from a specified format (csv or parquet)."""
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()
    try:
        if format == 'csv':
            df = pd.read_csv(file_path, low_memory=False)
            logging.info(f"DataFrame loaded from {file_path} (CSV)")
            return df
        elif format == 'parquet':
            df = pd.read_parquet(file_path)
            logging.info(f"DataFrame loaded from {file_path} (Parquet)")
            return df
        else:
            logging.error(f"Unsupported load format: {format}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading DataFrame from {file_path}: {e}")
        return pd.DataFrame()
