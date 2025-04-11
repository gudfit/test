# src/1_dataPreprocessing/weatherPredictionProcessing/merge_data.py
import pandas as pd
import os
import logging
from pathlib import Path
from datetime import timedelta, datetime

# Import config only if necessary for paths, avoid if logic is self-contained
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_merge_notebook_pair(weather_filepath: Path, flight_filepath: Path) -> pd.DataFrame:
    """
    Loads one pair of weather/flight files and merges based on notebook logic.
    Fixed to properly handle airport codes and timestamps.
    """
    logging.info(f"Processing pair: {weather_filepath.name} & {flight_filepath.name}")
    try:
        # --- Check if files exist ---
        if not weather_filepath.exists() or not flight_filepath.exists():
            logging.error(f"Files not found: {weather_filepath} or {flight_filepath}")
            return pd.DataFrame()

        # --- Load weather data with better error handling ---
        try:
            weather = pd.read_csv(weather_filepath)
            logging.info(f"Weather data loaded: {len(weather)} rows from {weather_filepath.name}")
        except Exception as e:
            logging.error(f"Error loading weather file {weather_filepath}: {e}")
            return pd.DataFrame()
            
        # --- Load flight data more selectively ---
        try:
            # Load only potentially relevant flight columns specified in config
            cols_to_load = config.FLIGHT_COLS_TO_LOAD
            logging.debug(f"Loading flight columns: {cols_to_load}")
            
            flight = pd.read_csv(flight_filepath, low_memory=False, 
                               usecols=lambda c: c in cols_to_load or 'Unnamed' in c)
            flight = flight.loc[:, ~flight.columns.str.contains('^Unnamed')]
            logging.info(f"Flight data loaded: {len(flight)} rows from {flight_filepath.name}")
        except Exception as e:
            logging.error(f"Error loading flight file {flight_filepath}: {e}")
            return pd.DataFrame()

        # --- Process weather data ---
        logging.info("Processing weather data...")
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather['timestamp_hour'] = weather['timestamp'].dt.floor('h')
        
        # Clean airport codes - strip whitespace and ensure uppercase
        weather['airport_code_iata'] = weather['airport_code_iata'].str.strip().str.upper()
        
        # Log weather airports for debugging
        unique_weather_airports = weather['airport_code_iata'].unique()
        logging.info(f"Weather data contains {len(unique_weather_airports)} unique airports: {unique_weather_airports}")
        
        # Keep only necessary weather cols specified in config + merge keys
        weather_cols_needed = list(set(config.NOTEBOOK_FEATURES_TO_LOAD_WEATHER + ['timestamp_hour', 'airport_code_iata']))
        weather_cols_available = [col for col in weather_cols_needed if col in weather.columns]
        weather = weather[weather_cols_available].copy()
        logging.debug(f"Weather data shape after selection: {weather.shape}")

        # --- Process flight data ---
        logging.info("Processing flight data...")
        flight = flight.dropna(subset=['CRSDepTime']).copy()
        flight['FlightDate'] = pd.to_datetime(flight['FlightDate'])
        
        def format_crs_dep_time(x):
            try: 
                return str(int(x)).zfill(4)
            except: 
                return None
                
        flight['CRSDepTime'] = flight['CRSDepTime'].apply(format_crs_dep_time)
        flight = flight.dropna(subset=['CRSDepTime'])

        flight['CRSDepHour'] = flight['CRSDepTime'].str[:2].astype(int)
        
        # Create proper timestamp for merging
        flight['timestamp_hour'] = pd.to_datetime(
            flight['FlightDate'].dt.strftime('%Y-%m-%d') + ' ' + 
            flight['CRSDepHour'].astype(str).str.zfill(2) + ':00:00'
        )
        
        # Clean airport codes - strip whitespace and ensure uppercase
        flight['Origin'] = flight['Origin'].str.strip().str.upper()
        flight['Dest'] = flight['Dest'].str.strip().str.upper()
        
        # Log flight airports for debugging
        unique_origin_airports = flight['Origin'].unique()
        logging.info(f"Flight data contains {len(unique_origin_airports)} unique origin airports (first 10): {unique_origin_airports[:10]}")
        
        # Calculate overlap between weather and flight airports
        overlap_airports = set(unique_weather_airports) & set(unique_origin_airports)
        logging.info(f"Overlap between weather and flight origins: {len(overlap_airports)} airports: {overlap_airports}")
        
        if len(overlap_airports) == 0:
            logging.error("NO OVERLAP between weather and flight airports! Check airport codes.")
        
        logging.debug(f"Flight data shape after processing: {flight.shape}")

        # --- Merge to origin airport weather ---
        logging.info("Merging flight data with origin airport weather...")
        if 'Origin' not in flight.columns or 'timestamp_hour' not in flight.columns or \
           'airport_code_iata' not in weather.columns or 'timestamp_hour' not in weather.columns:
            logging.error("Merge key columns missing from DataFrames.")
            return pd.DataFrame()

        merged = pd.merge(
            flight,
            weather,
            left_on=['Origin', 'timestamp_hour'],
            right_on=['airport_code_iata', 'timestamp_hour'],
            how='left',
            suffixes=('', '_origin')
        )
        
        # Rename weather columns to indicate they're for origin
        for col in weather_cols_available:
            if col not in ['airport_code_iata', 'timestamp_hour'] and col in merged.columns:
                merged.rename(columns={col: f"{col}_origin"}, inplace=True)
                
        # Drop redundant columns
        if 'airport_code_iata' in merged.columns:
            merged = merged.drop('airport_code_iata', axis=1)
        
        # --- Also merge to destination airport weather ---
        # For each flight, we'll also get the weather at the destination airport
        logging.info("Adding destination airport weather...")
        dest_merged = pd.merge(
            merged,
            weather,
            left_on=['Dest', 'timestamp_hour'],
            right_on=['airport_code_iata', 'timestamp_hour'],
            how='left',
            suffixes=('', '_dest')
        )
        
        # Rename weather columns for destination
        for col in weather_cols_available:
            if col not in ['airport_code_iata', 'timestamp_hour'] and col in dest_merged.columns and not col.endswith('_origin'):
                dest_merged.rename(columns={col: f"{col}_dest"}, inplace=True)
                
        # Drop redundant columns
        if 'airport_code_iata' in dest_merged.columns:
            dest_merged = dest_merged.drop('airport_code_iata', axis=1)
            
        # --- Add column with count of non-null weather features to check merge success ---
        weather_cols = [col for col in dest_merged.columns if any(
            col.startswith(f"{feat}_origin") or col.startswith(f"{feat}_dest") 
            for feat in config.WEATHER_FEATURES_BASE
        )]
        dest_merged['weather_data_count'] = dest_merged[weather_cols].notna().sum(axis=1)
        
        # Log merge statistics
        rows_with_weather = (dest_merged['weather_data_count'] > 0).sum()
        logging.info(f"Merged shape: {dest_merged.shape}, Rows with weather data: {rows_with_weather} ({rows_with_weather/len(dest_merged)*100:.2f}%)")
        
        return dest_merged

    except Exception as e:
        logging.error(f"Error during load/merge for {flight_filepath.name}: {e}", exc_info=True)
        return pd.DataFrame()


def load_and_merge_with_lag_features(weather_filepath: Path, flight_filepath: Path) -> pd.DataFrame:
    """
    Enhanced version that adds lagged weather features.
    Loads, merges, and adds lag features for weather data.
    """
    logging.info(f"Processing pair with lags: {weather_filepath.name} & {flight_filepath.name}")
    try:
        # --- Load and process weather data ---
        if not weather_filepath.exists() or not flight_filepath.exists():
            logging.error(f"Files not found: {weather_filepath} or {flight_filepath}")
            return pd.DataFrame()
            
        weather = pd.read_csv(weather_filepath)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather['timestamp_hour'] = weather['timestamp'].dt.floor('h')
        weather['airport_code_iata'] = weather['airport_code_iata'].str.strip().str.upper()
        
        # --- Create lagged versions of weather data ---
        logging.info("Creating lagged weather features...")
        all_weather_dfs = [weather]
        
        for lag_hours in config.LAG_HOURS:
            # Create a copy with lagged timestamp
            lag_weather = weather.copy()
            lag_weather['original_timestamp_hour'] = lag_weather['timestamp_hour']
            lag_weather['timestamp_hour'] = lag_weather['timestamp_hour'] + pd.Timedelta(hours=lag_hours)
            
            # Rename columns to indicate lag
            for col in config.WEATHER_FEATURES_TO_LAG:
                if col in lag_weather.columns:
                    lag_weather.rename(columns={col: f"{col}_lag{lag_hours}h"}, inplace=True)
            
            all_weather_dfs.append(lag_weather)
        
        # Combine all weather dataframes
        combined_weather = all_weather_dfs[0]
        for i, lag_df in enumerate(all_weather_dfs[1:], 1):
            lag_hours = config.LAG_HOURS[i-1]
            lag_cols = [f"{col}_lag{lag_hours}h" for col in config.WEATHER_FEATURES_TO_LAG if col in lag_df.columns]
            combined_weather = pd.merge(
                combined_weather,
                lag_df[['airport_code_iata', 'timestamp_hour'] + lag_cols],
                on=['airport_code_iata', 'timestamp_hour'],
                how='left'
            )
        
        # --- Load and process flight data (same as original) ---
        flight = pd.read_csv(flight_filepath, low_memory=False, 
                            usecols=lambda c: c in config.FLIGHT_COLS_TO_LOAD or 'Unnamed' in c)
        flight = flight.loc[:, ~flight.columns.str.contains('^Unnamed')]
        flight = flight.dropna(subset=['CRSDepTime']).copy()
        flight['FlightDate'] = pd.to_datetime(flight['FlightDate'])
        flight['CRSDepTime'] = flight['CRSDepTime'].apply(lambda x: str(int(x)).zfill(4) if pd.notna(x) and isinstance(x, (int, float)) else None)
        flight = flight.dropna(subset=['CRSDepTime'])
        flight['CRSDepHour'] = flight['CRSDepTime'].str[:2].astype(int)
        flight['timestamp_hour'] = pd.to_datetime(
            flight['FlightDate'].dt.strftime('%Y-%m-%d') + ' ' + 
            flight['CRSDepHour'].astype(str).str.zfill(2) + ':00:00'
        )
        flight['Origin'] = flight['Origin'].str.strip().str.upper()
        flight['Dest'] = flight['Dest'].str.strip().str.upper()
        
        # --- Merge to origin airport weather (with lags) ---
        merged = pd.merge(
            flight,
            combined_weather,
            left_on=['Origin', 'timestamp_hour'],
            right_on=['airport_code_iata', 'timestamp_hour'],
            how='left',
            suffixes=('', '_origin')
        )
        
        # Rename weather columns to indicate they're for origin
        for col in config.WEATHER_FEATURES_BASE:
            if col in merged.columns and col not in ['timestamp_hour']:
                merged.rename(columns={col: f"{col}_origin"}, inplace=True)
                
        for lag_hours in config.LAG_HOURS:
            for col in config.WEATHER_FEATURES_TO_LAG:
                lag_col = f"{col}_lag{lag_hours}h"
                if lag_col in merged.columns:
                    merged.rename(columns={lag_col: f"{lag_col}_origin"}, inplace=True)
        
        # Drop redundant columns
        if 'airport_code_iata' in merged.columns:
            merged = merged.drop('airport_code_iata', axis=1)
            
        # --- Also merge to destination airport weather ---
        dest_merged = pd.merge(
            merged,
            combined_weather,
            left_on=['Dest', 'timestamp_hour'],
            right_on=['airport_code_iata', 'timestamp_hour'],
            how='left',
            suffixes=('', '_dest')
        )
        
        # Rename weather columns for destination
        for col in config.WEATHER_FEATURES_BASE:
            if col in dest_merged.columns and col not in ['timestamp_hour'] and not col.endswith('_origin'):
                dest_merged.rename(columns={col: f"{col}_dest"}, inplace=True)
                
        for lag_hours in config.LAG_HOURS:
            for col in config.WEATHER_FEATURES_TO_LAG:
                lag_col = f"{col}_lag{lag_hours}h"
                if lag_col in dest_merged.columns and not lag_col.endswith('_origin'):
                    dest_merged.rename(columns={lag_col: f"{lag_col}_dest"}, inplace=True)
                
        # Drop redundant columns
        if 'airport_code_iata' in dest_merged.columns:
            dest_merged = dest_merged.drop('airport_code_iata', axis=1)
            
        # Log merge statistics
        weather_cols = [col for col in dest_merged.columns if any(
            (col.startswith(f"{feat}_origin") or col.startswith(f"{feat}_dest"))
            for feat in config.WEATHER_FEATURES_BASE
        )]
        dest_merged['weather_data_count'] = dest_merged[weather_cols].notna().sum(axis=1)
        rows_with_weather = (dest_merged['weather_data_count'] > 0).sum()
        logging.info(f"Merged with lags shape: {dest_merged.shape}, Rows with weather data: {rows_with_weather} ({rows_with_weather/len(dest_merged)*100:.2f}%)")
        
        return dest_merged
            
    except Exception as e:
        logging.error(f"Error during load/merge with lags for {flight_filepath.name}: {e}", exc_info=True)
        return pd.DataFrame()
