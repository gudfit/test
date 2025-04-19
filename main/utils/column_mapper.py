import pandas as pd
import master_config as config


def transform_cleaned_flights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a cleaned flights DataFrame to the expected format for MasterPredictor.
    """
    # Rename columns from cleaned schema to model schema
    mapping = {
        'flight_date': 'FlightDate',
        'tail_number': 'Tail_Number',
        'airline': 'Reporting_Airline',
        # If your cleaned data includes a flight number column, map it here
        'flight_number': 'Flight_Number_Reporting_Airline',
        'depart_from_iata': 'Origin',
        'arrive_at_iata': 'Dest',
        'scheduled_departure_local': 'CRSDepTime',
        'scheduled_arrival_local': 'CRSArrTime',
        'actual_departure_local': 'DepTime',
        'actual_arrival_local': 'ArrTime',
        'departure_delay': 'DepDelayMinutes',
        'arrival_delay': 'ArrDelayMinutes',
        'scheduled_duration': 'CRSElapsedTime',
        'actual_duration': 'ActualElapsedTime',
        'distance': 'Distance',
        'weather_delay': 'WeatherDelay'
    }
    df = df.rename(columns=mapping)

    # Format HHMM time strings for scheduled/actual times
    for col in ['CRSDepTime', 'CRSArrTime', 'DepTime', 'ArrTime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%H%M')

    # Convert durations (timedelta strings or pd.Timedelta) to minutes
    for col in ['CRSElapsedTime', 'ActualElapsedTime']:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds() / 60

    # Add columns expected by the model but missing in cleaned data
    df['Cancelled'] = 0.0
    df['Diverted'] = 0.0
    df['TaxiOut'] = pd.NA
    df['TaxiIn'] = pd.NA

    # Ensure all expected columns exist
    for col in config.EXPECTED_INPUT_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # Reorder columns to match model input
    return df[config.EXPECTED_INPUT_COLS]


def load_cleaned_and_transform(path: str) -> list:
    """
    Reads a cleaned CSV file, applies the transformation, and returns a list of dicts
    ready for MasterPredictor.predict().
    """
    df = pd.read_csv(path)
    df_transformed = transform_cleaned_flights(df)
    return df_transformed.to_dict(orient='records')

