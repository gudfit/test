#!/usr/bin/env python3
"""
flightDelay/src/4_data_processing.py

Cleans raw flight data, constructs chains of consecutive flights
for each aircraft, extracts features, bins delays into classes,
and writes train/validation/test splits.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from src.data_processing.config import (
    DATA_DIR,
    ML_DATA_DIR,
    AIRPORT_COORDS_FILE,
    FTD_COL,
    PFD_COL,
    FLIGHT_DATE_COL,
    ORIGIN_COL,
    DEST_COL,
    CARRIER_CODE_COL,
    TAIL_NUM_COL,
    FLIGHT_NUM_COL,
    SCHED_DEP_TIME_COL,
    SCHED_ARR_TIME_COL,
    DEP_DELAY_COL,
    ARR_DELAY_COL,
    ORIENTATION_COL,
    SCHED_DATETIME_COL,
    FLIGHT_DELAY_COL,
    DELAY_THRESHOLDS,
    NUM_CLASSES
)

def load_airport_data(filepath):
    """Load IATA→(lat,lon) from CSV for orientation lookup."""
    if not os.path.exists(filepath):
        print(f"Warning: coords file not found at {filepath}; skipping orientation.")
        return {}
    df = pd.read_csv(filepath, dtype={'iata': str}, usecols=['iata','latitude','longitude'])
    df.dropna(subset=['iata','latitude','longitude'], inplace=True)
    return {
        row['iata'].strip().upper(): (float(row['latitude']), float(row['longitude']))
        for _, row in df.iterrows()
    }

def calculate_orientation(origin, dest, airport_data):
    """Return 'North-South', 'East-West', 'Same Location', or 'Unknown'."""
    o, d = origin.strip().upper(), dest.strip().upper()
    if o in airport_data and d in airport_data:
        lat1, lon1 = airport_data[o]
        lat2, lon2 = airport_data[d]
        dlat, dlon = lat2 - lat1, lon2 - lon1
        if abs(dlat) < 1e-6 and abs(dlon) < 1e-6:
            return 'Same Location'
        return 'North-South' if abs(dlat) > abs(dlon) else 'East-West'
    return 'Unknown'

def clean_and_process_flight_data(input_file, coords_file):
    """
    1) Remove cancelled/diverted flights
    2) Drop missing critical fields
    3) Parse dates/times and build scheduled datetimes
    4) Compute flight durations & unified delay
    5) Compute orientation
    Returns a DataFrame with columns:
       [ TAIL_NUM_COL, SCHED_DATETIME_COL, 'Flight_Duration_Minutes',
         'Flight_Delay', 'Orientation' ]
    """
    df = pd.read_csv(input_file, low_memory=False)

    # --- 1. Remove cancelled/diverted ---
    for col in (CANCELLED_COL, DIVERTED_COL):
        if col in df.columns:
            df = df[df[col] == 0]

    # --- 2. Drop missing critical ---
    critical = [TAIL_NUM_COL, ORIGIN_COL, DEST_COL, FLIGHT_DATE_COL,
                SCHED_DEP_TIME_COL, SCHED_ARR_TIME_COL]
    df.dropna(subset=[c for c in critical if c in df.columns], inplace=True)

    # --- Clean identifiers ---
    df[TAIL_NUM_COL] = df[TAIL_NUM_COL].astype(str).str.strip().str.upper()
    df[[ORIGIN_COL, DEST_COL]] = df[[ORIGIN_COL, DEST_COL]] \
        .apply(lambda s: s.astype(str).str.strip().str.upper())

    # --- Parse FlightDate ---
    df['FlightDate_dt'] = pd.to_datetime(df[FLIGHT_DATE_COL],
                                         errors='coerce').dt.date
    df.dropna(subset=['FlightDate_dt'], inplace=True)

    # --- Parse HHMM times helper ---
    def parse_hhmm(col):
        s = col.fillna(-1).astype(float).astype(int).astype(str).str.zfill(4)
        s = s.replace('2400', '0000')
        valid = s.str.match(r'^([01]\d|2[0-3])([0-5]\d)$')
        return pd.to_datetime(s.where(valid),
                              format='%H%M', errors='coerce').dt.time

    df['DepTime_parsed'] = parse_hhmm(df[SCHED_DEP_TIME_COL])
    df['ArrTime_parsed'] = parse_hhmm(df[SCHED_ARR_TIME_COL])
    df.dropna(subset=['DepTime_parsed', 'ArrTime_parsed'], inplace=True)

    # --- Combine into datetimes ---
    df[SCHED_DATETIME_COL] = pd.to_datetime(
        df.apply(lambda r: pd.Timestamp.combine(r['FlightDate_dt'],
                                          r['DepTime_parsed']), axis=1)
    )
    df['SchedArr_dt'] = pd.to_datetime(
        df.apply(lambda r: pd.Timestamp.combine(r['FlightDate_dt'],
                                          r['ArrTime_parsed']), axis=1)
    )
    # Overnight adjustment
    ov_mask = df['SchedArr_dt'] < df[SCHED_DATETIME_COL]
    df.loc[ov_mask, 'SchedArr_dt'] += timedelta(days=1)

    # --- Flight duration & delay ---
    df['Flight_Duration_Minutes'] = (
        df['SchedArr_dt'] - df[SCHED_DATETIME_COL]
    ).dt.total_seconds() / 60
    df = df[df['Flight_Duration_Minutes'] > 0]

    if ARR_DELAY_COL in df.columns:
        df['Flight_Delay'] = pd.to_numeric(df[ARR_DELAY_COL],
                                           errors='coerce')
        df['Flight_Delay'].fillna(
            pd.to_numeric(df[DEP_DELAY_COL], errors='coerce'),
            inplace=True
        )
    else:
        df['Flight_Delay'] = pd.to_numeric(df[DEP_DELAY_COL],
                                           errors='coerce')
    df['Flight_Delay'].fillna(0, inplace=True)

    # --- Orientation ---
    airport_data = load_airport_data(coords_file)
    df['Orientation'] = df.apply(
        lambda r: calculate_orientation(r[ORIGIN_COL],
                                         r[DEST_COL],
                                         airport_data),
        axis=1
    )

    # --- Final trim ---
    out_cols = [
        TAIL_NUM_COL,
        SCHED_DATETIME_COL,
        'Flight_Duration_Minutes',
        'Flight_Delay',
        'Orientation'
    ]
    return df[out_cols].copy()

def create_flight_chains(df, output_dir,
                         chain_length=3,
                         max_time_diff_hours=24):
    """
    Build consecutive‐flight chains per aircraft,
    extract sliding‐window features, bin delays into classes,
    and write train/val/test CSVs into output_dir.
    """
    # Sort & group by tail number
    df = df.sort_values([TAIL_NUM_COL, SCHED_DATETIME_COL]) \
           .reset_index(drop=True)

    raw_chains = []
    current_tail = None
    for _, row in df.iterrows():
        if row[TAIL_NUM_COL] != current_tail:
            raw_chains.append([])
            current_tail = row[TAIL_NUM_COL]
        raw_chains[-1].append(row)

    # Slide over each chain
    sequences = []
    for chain in raw_chains:
        for i in range(len(chain) - chain_length + 1):
            window = chain[i : i + chain_length]
            valid = True
            for j in range(1, chain_length):
                prev = window[j-1]
                curr = window[j]
                prev_arr = prev[SCHED_DATETIME_COL] + \
                           timedelta(minutes=prev['Flight_Duration_Minutes'])
                gap = (curr[SCHED_DATETIME_COL] - prev_arr).total_seconds() / 60
                if gap < 1 or gap > max_time_diff_hours * 60:
                    valid = False
                    break
            if valid:
                sequences.append(window)

    # Extract features + labels
    features = []
    labels   = []
    for window in sequences:
        feat = {}
        for idx, flight in enumerate(window, start=1):
            prefix = f'flight{idx}_'
            feat[f'{prefix}Schedule_DateTime']    = flight[SCHED_DATETIME_COL]
            feat[f'{prefix}Flight_Duration_Minutes'] = flight['Flight_Duration_Minutes']
            feat[f'{prefix}Flight_Delay']           = flight['Flight_Delay']
            feat[f'{prefix}Orientation']           = flight['Orientation']
            if idx > 1:
                prev = window[idx-2]
                prev_arr = prev[SCHED_DATETIME_COL] + \
                           timedelta(minutes=prev['Flight_Duration_Minutes'])
                gap = (flight[SCHED_DATETIME_COL] - prev_arr).total_seconds() / 60
                feat[f'{prefix}FTD'] = max(0, gap)
                feat[f'{prefix}PFD'] = prev['Flight_Delay']
        features.append(feat)
        labels.append(window[-1]['Flight_Delay'])

    df_feat = pd.DataFrame(features)
    df_feat['delay_label'] = labels

    # --- Bin into classes ---
    def categorize_delay(delay):
        for cls, (low, high) in enumerate(zip(DELAY_THRESHOLDS[:-1],
                                              DELAY_THRESHOLDS[1:])):
            if low < delay <= high:
                return cls
        return NUM_CLASSES - 1

    df_feat['delay_category'] = df_feat['delay_label'] \
                                .apply(categorize_delay)

    # --- Split into train/val/test ---
    train = df_feat.sample(frac=0.7, random_state=42)
    rest  = df_feat.drop(train.index)
    val   = rest.sample(frac=0.5, random_state=42)
    test  = rest.drop(val.index)

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'train_set.csv'),
                 index=False)
    val.to_csv(os.path.join(output_dir, 'validation_set.csv'),
               index=False)
    test.to_csv(os.path.join(output_dir, 'test_set.csv'),
                index=False)

    print(f"Generated {len(df_feat)} windows; "
          f"train={len(train)}, val={len(val)}, test={len(test)}")
    return df_feat

def main():
    raw_input  = os.path.join(ML_DATA_DIR, 'report_carrier_top10.csv')
    coords_file= str(AIRPORT_COORDS_FILE)
    output_dir = os.path.join(ML_DATA_DIR, 'processedDataTest')

    print("Cleaning raw flight data…")
    df_clean = clean_and_process_flight_data(raw_input, coords_file)

    print("Creating flight chains + extracting features…")
    create_flight_chains(df_clean, output_dir)

if __name__ == "__main__":
    main()
