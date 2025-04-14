# master_predictor/utils/datetime_helpers.py
import pandas   as pd
from   datetime import datetime, date, time

def parse_hhmm_to_time(time_val):
    """Safely parses HHMM format (float, int, or string) to a time object."""
    if pd.isna(time_val):
        return None
    try:
        # Convert to string, remove potential .0, and ensure zero-padding to 4 digits
        time_str = str(int(float(time_val))).zfill(4)
        # Handle '2400' as a special case by converting it to '0000'
        if time_str == '2400':
            time_str = '0000'
        # Validate HHMM format: 4 digits, within '0000' to '2359', with minutes less than '60'
        if not (len(time_str) == 4 and '0000' <= time_str <= '2359' and time_str[2:4] <= '59'):
            return None
        # Parse the valid time string using pandas for consistency
        dt_obj = pd.to_datetime(time_str, format='%H%M', errors='coerce')
        # Extract and return the time part if valid; otherwise, return None
        return dt_obj.time() if pd.notna(dt_obj) else None
    except (ValueError, TypeError):
        return None

def combine_date_time_flex(date_input, time_input):
    """Combines various date/time inputs into a pandas Timestamp."""
    if pd.isna(date_input) or pd.isna(time_input):
        return pd.NaT

    # Convert date_input to a date object
    try:
        if isinstance(date_input, str):
            date_obj = pd.to_datetime(date_input).date()
        elif isinstance(date_input, pd.Timestamp):
            date_obj = date_input.date()
        elif isinstance(date_input, date):
            date_obj = date_input
        else:
            date_obj = pd.to_datetime(date_input).date()
    except Exception:
        return pd.NaT  # Cannot parse date

    # Ensure time_input is a time object
    if isinstance(time_input, pd.Timestamp):
        time_obj = time_input.time()
    elif not isinstance(time_input, time):
        time_obj = parse_hhmm_to_time(time_input)
        if time_obj is None:
            return pd.NaT  # Cannot parse time
    else:
        time_obj = time_input

    # Combine the date and time objects into a pandas Timestamp
    try:
        return pd.Timestamp(datetime.combine(date_obj, time_obj))
    except (TypeError, ValueError):
        return pd.NaT

def calculate_scheduled_datetimes(flight_dict):
    """Calculates scheduled departure and arrival Timestamps for a flight dictionary."""
    sched_dep_dt = combine_date_time_flex(
        flight_dict.get('FlightDate'),
        flight_dict.get('CRSDepTime')
    )
    sched_arr_dt = combine_date_time_flex(
        flight_dict.get('FlightDate'),
        flight_dict.get('CRSArrTime')
    )

    # Adjust the arrival day if the scheduled arrival time is earlier than departure time
    if (pd.notna(sched_dep_dt) and pd.notna(sched_arr_dt) and 
        isinstance(sched_dep_dt, pd.Timestamp) and isinstance(sched_arr_dt, pd.Timestamp)):
        if sched_arr_dt < sched_dep_dt:
            sched_arr_dt += pd.Timedelta(days=1)

    return sched_dep_dt, sched_arr_dt

def calculate_actual_datetimes(flight_dict):
    """Calculates actual departure and arrival Timestamps for a flight dictionary."""
    act_dep_dt = combine_date_time_flex(
        flight_dict.get('FlightDate'),
        flight_dict.get('DepTime')
    )
    act_arr_dt = combine_date_time_flex(
        flight_dict.get('FlightDate'),
        flight_dict.get('ArrTime')
    )

    # Adjust the arrival day if the actual arrival time is earlier than departure time
    if (pd.notna(act_dep_dt) and pd.notna(act_arr_dt) and 
        isinstance(act_dep_dt, pd.Timestamp) and isinstance(act_arr_dt, pd.Timestamp)):
        if act_arr_dt < act_dep_dt:
            act_arr_dt += pd.Timedelta(days=1)

    return act_dep_dt, act_arr_dt
