# master_predictor/utils/datetime_helpers.py
import pandas as pd
from datetime import datetime, date, time # <-- ADD THIS IMPORT

def parse_hhmm_to_time(time_val):
    """Safely parses HHMM format (float, int, or string) to a time object."""
    if pd.isna(time_val):
        return None
    try:
        # Convert to string, remove potential .0, zfill
        time_str = str(int(float(time_val))).zfill(4)
        # Handle '2400' -> '0000'
        if time_str == '2400':
            time_str = '0000'
        # Validate HHMM format
        if not (len(time_str) == 4 and '0000' <= time_str <= '2359' and time_str[2:4] <= '59'):
             return None
        # Parse valid time string using pandas for consistency
        # Use pd.to_datetime which handles format checks well
        dt_obj = pd.to_datetime(time_str, format='%H%M', errors='coerce')
        return dt_obj.time() if pd.notna(dt_obj) else None # Extract time part
    except (ValueError, TypeError):
        return None

def combine_date_time_flex(date_input, time_input):
    """Combines various date/time inputs into a pandas Timestamp."""
    if pd.isna(date_input) or pd.isna(time_input):
        return pd.NaT

    # Ensure date_input is a date object
    try:
        if isinstance(date_input, str):
            date_obj = pd.to_datetime(date_input).date()
        elif isinstance(date_input, pd.Timestamp):
            date_obj = date_input.date()
        elif isinstance(date_input, date): # Use imported 'date'
             date_obj = date_input
        else:
            # Fallback attempt, might raise error if unparseable
            date_obj = pd.to_datetime(date_input).date()
    except Exception:
        # print(f"Debug: Failed to parse date_input: {date_input}") # Optional
        return pd.NaT # Cannot parse date

    # Ensure time_input is a time object
    if isinstance(time_input, pd.Timestamp): # Can happen if already processed
        time_obj = time_input.time()
    elif not isinstance(time_input, time): # Use imported 'time'
        time_obj = parse_hhmm_to_time(time_input)
        if time_obj is None:
            # print(f"Debug: Failed to parse time_input: {time_input}") # Optional
            return pd.NaT # Cannot parse time
    else:
        time_obj = time_input # Already a time object

    # Combine date and time objects
    try:
        # Combine using the imported datetime object
        return pd.Timestamp(datetime.combine(date_obj, time_obj))
    except (TypeError, ValueError) as e:
        # print(f"Debug: Combine error for date {date_obj}, time {time_obj}: {e}") # Optional
        return pd.NaT # Return NaT if combination fails


def calculate_scheduled_datetimes(flight_dict):
    """Calculates scheduled departure and arrival Timestamps for a flight dictionary."""
    sched_dep_dt = combine_date_time_flex(flight_dict.get('FlightDate'), flight_dict.get('CRSDepTime'))
    sched_arr_dt = combine_date_time_flex(flight_dict.get('FlightDate'), flight_dict.get('CRSArrTime'))

    # Ensure results are Timestamps before comparison/adjustment
    if pd.notna(sched_dep_dt) and pd.notna(sched_arr_dt) and isinstance(sched_arr_dt, pd.Timestamp) and isinstance(sched_dep_dt, pd.Timestamp):
        if sched_arr_dt < sched_dep_dt:
            sched_arr_dt += pd.Timedelta(days=1)

    return sched_dep_dt, sched_arr_dt

def calculate_actual_datetimes(flight_dict):
    """Calculates actual departure and arrival Timestamps for a flight dictionary."""
    act_dep_dt = combine_date_time_flex(flight_dict.get('FlightDate'), flight_dict.get('DepTime'))
    act_arr_dt = combine_date_time_flex(flight_dict.get('FlightDate'), flight_dict.get('ArrTime'))

    # Ensure results are Timestamps before comparison/adjustment
    if pd.notna(act_dep_dt) and pd.notna(act_arr_dt) and isinstance(act_arr_dt, pd.Timestamp) and isinstance(act_dep_dt, pd.Timestamp):
        if act_arr_dt < act_dep_dt:
            act_arr_dt += pd.Timedelta(days=1)

    return act_dep_dt, act_arr_dt
