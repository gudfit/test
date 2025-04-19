#!/usr/bin/env python3
"""
Inference script: accepts snake_case JSON input via file or CLI,
normalizes keys to MasterPredictor’s expected schema, and prints the prediction.
On failure, falls back to the FDPP-ML regressor pipeline.
Removes probability output and applies direct threshold categorization for regressor.
Extracts cancelled/diverted status from input’s `status` field when present.
Transforms full datetime strings into HHMM format for time fields.
Exits non-zero only if both primary and fallback fail.
"""
import argparse
import json
import sys
import warnings
import os
import re
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure FDPP-ML pipeline is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
fdpp_src = os.path.join(script_dir, 'flightDelay', 'src')
if fdpp_src not in sys.path:
    sys.path.insert(0, fdpp_src)

try:
    from modeling.predict import run_prediction as fdpp_run_prediction
except ImportError:
    try:
        from src.modeling.predict import run_prediction as fdpp_run_prediction
    except ImportError:
        fdpp_run_prediction = None

from master_predictor import MasterPredictor

# Delay status map for threshold categorization
DELAY_STATUS_MAP = [
    "On Time / Slight Delay (<= 15 min)",
    "Delayed (15-60 min)",
    "Significantly Delayed (60-120 min)",
    "Severely Delayed (120-240 min)",
    "Extremely Delayed (> 240 min)"
]

def drop_alpha_prefix(s: str) -> str:
    """Remove any leading letters from a flight‐number string."""
    return re.sub(r'^[A-Za-z]+', '', s or '')

def extract_hhmm(dt_str):
    """
    Extracts HHMM string from a full datetime string like 'YYYY-MM-DD HH:MM'.
    Returns None if input is falsy or cannot be parsed.
    """
    if not dt_str or not isinstance(dt_str, str):
        return None
    parts = dt_str.strip().split()
    if len(parts) < 2:
        return None
    time_part = parts[-1]
    hhmm = time_part.replace(':', '')
    if len(hhmm) == 3:
        hhmm = '0' + hhmm
    return hhmm if len(hhmm) == 4 else None

def compute_delay(sched_str, actual_str):
    """Return (actual – sched) in minutes, or None if we can’t parse."""
    if not sched_str or not actual_str:
        return None
    try:
        sched_dt  = datetime.fromisoformat(sched_str)
        actual_dt = datetime.fromisoformat(actual_str)
        return (actual_dt - sched_dt).total_seconds() / 60.0
    except Exception:
        return None

def compute_duration(start_str, end_str):
    """
    Return (end – start) in minutes, or None if either input is falsy or unparsable.
    """
    if not start_str or not end_str:
        return None
    try:
        start_dt = datetime.fromisoformat(start_str)
        end_dt   = datetime.fromisoformat(end_str)
        return (end_dt - start_dt).total_seconds() / 60.0
    except Exception:
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MasterPredictor with fallback to FDPP-ML regressor."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--flights-file",
        dest="flights_file",
        type=str,
        help="Path to JSON file containing flight context (snake_case list of dicts)."
    )
    group.add_argument(
        "--flights-cli",
        dest="flights_cli",
        type=str,
        help="JSON string containing flight context (snake_case list of dicts)."
    )
    return parser.parse_args()


def load_and_normalize_from_list(data):
    if not isinstance(data, list) or not data:
        print("Error: Input must be a non-empty list.", file=sys.stderr)
        sys.exit(1)

    normalized = []
    for rec in data:
        # --- cancelled/diverted flags from status ---
        raw_status   = rec.get("status") or ""
        status   = raw_status.lower()
        canceled = 1.0 if status == "cancelled" else 0.0
        diverted = 1.0 if status == "diverted" else 0.0

        # --- delays (as before) ---
        dep_delay = rec.get("dep_delay_minutes")
        if dep_delay is None:
            dep_delay = compute_duration(
                rec.get("scheduled_departure_utc"),
                rec.get("actual_departure_utc")
            )

        arr_delay = rec.get("arr_delay_minutes")
        if arr_delay is None:
            arr_delay = compute_duration(
                rec.get("scheduled_arrival_utc"),
                rec.get("actual_arrival_utc")
            )

        # --- schedule_duration (CRSElapsedTime) ---
        sched_dur = rec.get("schedule_duration")
        if sched_dur is None:
            sched_dur = compute_duration(
                rec.get("scheduled_departure_utc"),
                rec.get("scheduled_arrival_utc")
            )

        # --- actual_duration (ActualElapsedTime) ---
        actual_dur = rec.get("actual_duration")
        if actual_dur is None:
            actual_dur = compute_duration(
                rec.get("actual_departure_utc"),
                rec.get("actual_arrival_utc")
            )

        normalized.append({
            "FlightDate": rec.get("flight_date"),
            "Tail_Number": rec.get("tail_number","").strip(),
            "Reporting_Airline": rec.get("airline_code"),
            "Flight_Number_Reporting_Airline":drop_alpha_prefix(rec.get("flight_number_iata")),
            "Origin": rec.get("depart_from_iata"),
            "Dest": rec.get("arrive_at_iata"),
            "CRSDepTime": extract_hhmm(rec.get("scheduled_departure_utc")),
            "CRSArrTime": extract_hhmm(rec.get("scheduled_arrival_utc")),
            "DepDelayMinutes": dep_delay,
            "ArrDelayMinutes": arr_delay,
            "DepTime": extract_hhmm(rec.get("actual_departure_utc")),
            "ArrTime": extract_hhmm(rec.get("actual_arrival_utc")),
            # here we use the computed-or-provided durations:
            "CRSElapsedTime": sched_dur,
            "ActualElapsedTime": actual_dur,
            "AirTime": rec.get("air_time"),
            "Distance": rec.get("distance"),
            "WeatherDelay": rec.get("weather_delay"),
            "Cancelled": canceled,
            "Diverted": diverted,
            "TaxiOut": rec.get("taxi_out"),
            "TaxiIn": rec.get("taxi_in"),
        })
    return normalized

def load_and_normalize(args):
    try:
        if args.flights_file:
            with open(args.flights_file) as f:
                data = json.load(f)
        else:
            data = json.loads(args.flights_cli)
    except Exception as e:
        print(f"Error loading JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    return load_and_normalize_from_list(data)


def categorize_delay(minutes):
    if minutes is None:
        return None
    try:
        m = float(minutes)
    except:
        return None
    if m <= 15:
        return DELAY_STATUS_MAP[0]
    if m <= 60:
        return DELAY_STATUS_MAP[1]
    if m <= 120:
        return DELAY_STATUS_MAP[2]
    if m <= 240:
        return DELAY_STATUS_MAP[3]
    return DELAY_STATUS_MAP[4]


def run_fdpp_fallback():
    if not fdpp_run_prediction:
        print("Error: FDPP-ML fallback not available.", file=sys.stderr)
        sys.exit(1)
    print("Falling back to FDPP-ML pipeline...", file=sys.stderr)
    fdpp_run_prediction()


def main():
    args = parse_args()
    flights = load_and_normalize(args)
    predictor = MasterPredictor()
    result = predictor.predict(flights)

    if result.get('status') == 'success' and result.get('value') is not None:
        if result.get('prediction_type') == 'regressor':
            msg = result.get('message', '')
            m = re.search(r'Predicted delay ([\d\.]+) min', msg)
            if m:
                delay_min = float(m.group(1))
                result['value'] = categorize_delay(delay_min)
        # remove any probabilities
        result.pop('probabilities', None)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    msg = result.get('message', 'No message')
    print(f"MasterPredictor fallback triggered: {msg}", file=sys.stderr)
    run_fdpp_fallback()


if __name__ == "__main__":
    main()

