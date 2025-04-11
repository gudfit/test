# src/data_preprocessing/merge_data.py
import pandas as pd
import os
import logging
from pathlib import Path
from datetime import timedelta # Needed for notebook logic

# Import config only if necessary for paths, avoid if logic is self-contained
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Keep previous merge functions if needed for other tasks ---
# def merge_flight_reports(...)
# def merge_weather_reports(...)
# def merge_single_weather_set(...)
# def merge_flights_with_weather_and_lags(...)

# --- NEW Function for Notebook Replication ---
def load_and_merge_notebook_pair(weather_filepath: Path, flight_filepath: Path) -> pd.DataFrame:
    """Loads one pair of weather/flight files and merges based on notebook logic."""
    logging.info(f"Processing pair: {weather_filepath.name} & {flight_filepath.name}")
    try:
        if not weather_filepath.exists() or not flight_filepath.exists():
            logging.error(f"Files not found: {weather_filepath} or {flight_filepath}")
            return pd.DataFrame()

        weather = pd.read_csv(weather_filepath)
        # Load only potentially relevant flight columns specified in config
        flight = pd.read_csv(flight_filepath, low_memory=False, usecols=lambda c: c in config.FLIGHT_COLS_TO_LOAD or 'Unnamed' in c)
        flight = flight.loc[:, ~flight.columns.str.contains('^Unnamed')]

        # --- Process weather ---
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather['timestamp_hour'] = weather['timestamp'].dt.floor('h')
        weather['airport_code_iata'] = weather['airport_code_iata'].str.strip()
        # Keep only necessary weather cols specified in config + merge keys
        weather_cols_needed = list(set(config.NOTEBOOK_FEATURES_TO_LOAD_WEATHER + ['timestamp_hour', 'airport_code_iata']))
        weather_cols_available = [col for col in weather_cols_needed if col in weather.columns]
        weather = weather[weather_cols_available].copy()
        logging.debug(f"Weather data shape after selection: {weather.shape}")

        # --- Process flight ---
        flight = flight.dropna(subset=['CRSDepTime']).copy()
        flight['FlightDate'] = pd.to_datetime(flight['FlightDate'])
        def format_crs_dep_time(x):
            try: return str(int(x)).zfill(4)
            except: return None
        flight['CRSDepTime'] = flight['CRSDepTime'].apply(format_crs_dep_time)
        flight = flight.dropna(subset=['CRSDepTime'])

        flight['CRSDepHour'] = flight['CRSDepTime'].str[:2].astype(int)
        flight['timestamp_hour'] = flight['FlightDate'] + pd.to_timedelta(flight['CRSDepHour'], unit='h')
        flight['Origin'] = flight['Origin'].str.strip()
        # Ensure Dest exists if it's in the load list (needed for later OHE)
        if 'Dest' in config.FLIGHT_COLS_TO_LOAD and 'Dest' in flight.columns:
            flight['Dest'] = flight['Dest'].str.strip()
        logging.debug(f"Flight data shape after processing: {flight.shape}")

        # --- Merge ---
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
            suffixes=('', '_w')
        )
        if 'airport_code_iata' in merged.columns:
            merged = merged.drop('airport_code_iata', axis=1)

        logging.info(f"Merged shape: {merged.shape}")
        return merged

    except Exception as e:
        logging.error(f"Error during load/merge for {flight_filepath.name}: {e}", exc_info=True) # Add exc_info
        return pd.DataFrame()
