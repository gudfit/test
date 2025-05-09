# eu_flight_predictor/eu_merge_data.py
import pandas as pd
import glob
import os

# Import specific variables needed from eu_config
from eu_config import (
    EU_INDIVIDUAL_CLEANED_CSVS_DIR,  # This is what we need for the source
    MERGED_EU_DATA_FILE,  # This is the output file path
    EU_COLUMN_MAPPING_RAW,
    TARGET_COL_INTERNAL_NAME,
)


def merge_eu_flight_data():
    """Merges all cleaned EU flight CSVs into a single file."""

    csv_files = glob.glob(
        os.path.join(EU_INDIVIDUAL_CLEANED_CSVS_DIR, "cleaned_Flights_*.csv")
    )
    if not csv_files:
        print(
            f"No CSV files found in {EU_INDIVIDUAL_CLEANED_CSVS_DIR}. Please check the path and filenames."
        )
        return

    print(f"Found {len(csv_files)} files to merge: {csv_files}")
    df_list = []
    for file in csv_files:
        try:
            df_list.append(pd.read_csv(file, low_memory=False))
            print(f"Successfully read {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not df_list:
        print("No dataframes were loaded. Exiting merge.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"Merged dataframe shape: {merged_df.shape}")

    target_internal_name_for_check = TARGET_COL_INTERNAL_NAME
    target_csv_name_from_mapping = EU_COLUMN_MAPPING_RAW.get(
        target_internal_name_for_check
    )

    if (
        target_csv_name_from_mapping
        and target_csv_name_from_mapping not in merged_df.columns
    ):
        print(
            f"WARNING: Expected target column '{target_csv_name_from_mapping}' (mapped from internal name '{target_internal_name_for_check}') not found in merged EU data!"
        )
        print(f"Available columns in merged data: {merged_df.columns.tolist()}")
    elif not target_csv_name_from_mapping:
        print(
            f"WARNING: The internal target column name '{target_internal_name_for_check}' is not mapped to an actual CSV column name in EU_COLUMN_MAPPING_RAW (in eu_config.py)."
        )
    else:
        print(
            f"Target column check: Internal name '{target_internal_name_for_check}' maps to CSV column '{target_csv_name_from_mapping}', which is present in merged data."
        )

    # Ensure the directory for the merged file exists
    # MERGED_EU_DATA_FILE already contains the full path including the directory from eu_config.py
    output_dir_for_merged_file = os.path.dirname(MERGED_EU_DATA_FILE)
    os.makedirs(output_dir_for_merged_file, exist_ok=True)

    merged_df.to_csv(MERGED_EU_DATA_FILE, index=False)
    print(f"Successfully merged EU data to {MERGED_EU_DATA_FILE}")


if __name__ == "__main__":
    merge_eu_flight_data()
