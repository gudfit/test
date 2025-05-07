# eu_flight_predictor/utils/datetime_helpers.py
import pandas as pd
from datetime import datetime, time


def parse_hhmm_to_time(time_val):
    """Safely parses HHMM format (string or float) to datetime.time object."""
    if pd.isna(time_val):
        return None
    time_str = str(time_val)
    if "." in time_str:  # Handle float like 1230.0
        time_str = time_str.split(".")[0]

    if (
        not time_str or not time_str.isdigit()
    ):  # Ensure it's purely digits after handling float
        return None

    if len(time_str) < 3 or len(time_str) > 4:
        return None  # Invalid length

    time_str_padded = time_str.zfill(4)

    if time_str_padded == "2400":  # Midnight case
        return time(0, 0)
    try:
        return datetime.strptime(time_str_padded, "%H%M").time()
    except ValueError:
        return None


def combine_date_time_objects(date_obj, time_obj):
    """Combines a date object and a time object into a pd.Timestamp."""
    if pd.isna(date_obj) or pd.isna(time_obj):
        return pd.NaT
    try:
        # Ensure date_obj is a date (not datetime) if it comes from pd.to_datetime().dt.date
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        return pd.Timestamp.combine(date_obj, time_obj)
    except (TypeError, ValueError):
        return pd.NaT


def calculate_scheduled_datetimes(flight_data_dict):
    """
    Calculates SchedDepDateTime and SchedArrDateTime from a flight data dictionary.
    Expects 'FlightDate', 'CRSDepTime', 'CRSArrTime'.
    """
    flight_date_str = flight_data_dict.get("FlightDate")
    crs_dep_time_str = flight_data_dict.get("CRSDepTime")
    crs_arr_time_str = flight_data_dict.get("CRSArrTime")

    if not flight_date_str or pd.isna(flight_date_str):
        return pd.NaT, pd.NaT

    try:
        flight_date_obj = pd.to_datetime(flight_date_str).date()
    except (ValueError, TypeError):
        return pd.NaT, pd.NaT

    crs_dep_time_obj = parse_hhmm_to_time(crs_dep_time_str)
    crs_arr_time_obj = parse_hhmm_to_time(crs_arr_time_str)

    sched_dep_datetime = combine_date_time_objects(flight_date_obj, crs_dep_time_obj)
    sched_arr_datetime = combine_date_time_objects(flight_date_obj, crs_arr_time_obj)

    # Handle overnight flights for scheduled times
    if (
        pd.notna(sched_dep_datetime)
        and pd.notna(sched_arr_datetime)
        and sched_arr_datetime < sched_dep_datetime
    ):
        sched_arr_datetime += pd.Timedelta(days=1)

    return sched_dep_datetime, sched_arr_datetime


def calculate_actual_datetimes(flight_data_dict):
    """
    Calculates ActualDepDateTime and ActualArrDateTime from a flight data dictionary.
    Expects 'FlightDate', 'DepTime', 'ArrTime'.
    Uses scheduled departure date as the base for actual departure if DepTime causes day roll-over.
    """
    flight_date_str = flight_data_dict.get("FlightDate")  # Date of the flight operation
    actual_dep_time_str = flight_data_dict.get("DepTime")
    actual_arr_time_str = flight_data_dict.get("ArrTime")
    # For context, to determine if actual departure moved to next day relative to scheduled
    crs_dep_time_str = flight_data_dict.get("CRSDepTime")

    if not flight_date_str or pd.isna(flight_date_str):
        return pd.NaT, pd.NaT
    try:
        flight_date_obj = pd.to_datetime(flight_date_str).date()
    except (ValueError, TypeError):
        return pd.NaT, pd.NaT

    actual_dep_time_obj = parse_hhmm_to_time(actual_dep_time_str)
    actual_arr_time_obj = parse_hhmm_to_time(actual_arr_time_str)
    crs_dep_time_obj = parse_hhmm_to_time(crs_dep_time_str)

    actual_dep_datetime = combine_date_time_objects(
        flight_date_obj, actual_dep_time_obj
    )

    # Adjust actual departure day if it's significantly earlier than CRSDepTime (e.g. CRSDep 0010, ActualDep 2350)
    # This implies the actual departure was on the next day relative to the FlightDate's CRSDepTime.
    if (
        pd.notna(actual_dep_datetime)
        and pd.notna(crs_dep_time_obj)
        and actual_dep_time_obj < crs_dep_time_obj
        and (
            crs_dep_time_obj.hour - actual_dep_time_obj.hour > 12
            or (crs_dep_time_obj.hour == 0 and actual_dep_time_obj.hour > 12)
        )
    ):  # Crude check for day rollover
        actual_dep_datetime += pd.Timedelta(days=1)

    # Base the actual arrival on the (potentially adjusted) actual departure date
    if pd.notna(actual_dep_datetime) and pd.notna(actual_arr_time_obj):
        actual_arr_datetime_base_date = actual_dep_datetime.date()
        actual_arr_datetime = combine_date_time_objects(
            actual_arr_datetime_base_date, actual_arr_time_obj
        )

        # Handle overnight flights for actual times
        if pd.notna(actual_arr_datetime) and actual_arr_datetime < actual_dep_datetime:
            actual_arr_datetime += pd.Timedelta(days=1)
    else:
        actual_arr_datetime = pd.NaT

    return actual_dep_datetime, actual_arr_datetime
