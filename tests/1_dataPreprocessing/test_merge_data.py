# tests/1_dataPreprocessing/test_merge_data.py
import pytest
import pandas as pd
from pathlib import Path
import os
import shutil # To clean up test artifacts

# Mock the config paths to point to test-specific data/results directories if needed
# Or ensure the test runner executes from the project root

# Assuming the project root is the current working directory when running pytest
# If not, adjust paths accordingly
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
TEST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "test_data" # Create this dir
TEST_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "test_results"

# Create dummy data/results dirs for testing
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Mock config paths (this is one way, could also use pytest fixtures with monkeypatch)
# Important: Ensure the modules use the mocked config or pass paths explicitly
MOCK_CONFIG = {
    "MERGED_FLIGHTS_FILE": TEST_RESULTS_DIR / "merged_flights.csv",
    "MERGED_WEATHER_FILE": TEST_RESULTS_DIR / "merged_weather.csv",
    "FINAL_MERGED_FILE": TEST_RESULTS_DIR / "final_merged_data.csv",
    "FLIGHT_PERFORMANCE_DIR": TEST_DATA_DIR,
    "WEATHER_DIR": TEST_DATA_DIR,
    "FLIGHT_CSV_PATTERN": "test_flights*.csv",
    "WEATHER_US_CSV_PATTERN": "test_weather*.csv",
    "WEATHER_EU_CSV_PATTERN": "non_existent*.csv", # Test handling missing files
    "FLIGHT_COLS_TO_KEEP": ['FlightDate', 'CRSDepTime', 'WeatherDelay', 'Origin', 'Dest'], # Simplified
    "WEATHER_COLS_TO_KEEP": ['airport_code_iata', 'timestamp', 'temp_c'], # Simplified
    # Add other necessary mocked config values here
}

# Create dummy CSV files for testing in TEST_DATA_DIR
# flights 1
pd.DataFrame({
    'FlightDate': ['2022-03-01', '2022-03-01'],
    'CRSDepTime': ['0800', '1000'],
    'WeatherDelay': [0.0, 20.0],
    'Origin': ['ATL', 'JFK'],
    'Dest': ['JFK', 'LAX']
}).to_csv(TEST_DATA_DIR / "test_flights_1.csv", index=False)
# flights 2
pd.DataFrame({
    'FlightDate': ['2022-03-02'],
    'CRSDepTime': ['1200'],
    'WeatherDelay': [5.0],
    'Origin': ['ATL'],
    'Dest': ['MIA']
}).to_csv(TEST_DATA_DIR / "test_flights_2.csv", index=False)
# weather 1
pd.DataFrame({
    'airport_code_iata': ['ATL', 'JFK', 'LAX', 'MIA'],
    'timestamp': ['2022-03-01 08:00:00', '2022-03-01 10:00:00', '2022-03-01 12:00:00', '2022-03-02 13:00:00'],
    'temp_c': [10, 15, 20, 25]
}).to_csv(TEST_DATA_DIR / "test_weather_1.csv", index=False)


# Import the functions to test AFTER potentially mocking dependencies
# Need to ensure they use the mocked config
# For simplicity here, we assume the functions can accept paths/config dict,
# or we use monkeypatching in fixtures. Let's assume monkeypatching for config.
# Define this in a conftest.py or at the top of the test file

@pytest.fixture(autouse=True) # Apply automatically to all tests in this module
def mock_config_paths(monkeypatch):
    # Mock the config module attributes used by the functions
    from src import config as actual_config
    for key, value in MOCK_CONFIG.items():
        monkeypatch.setattr(actual_config, key, value)
    # Also mock directories if needed
    monkeypatch.setattr(actual_config, "PROCESSED_DATA_DIR", TEST_RESULTS_DIR)


# Import AFTER mocking
from src.1_dataPreprocessing import merge_data

# Clean up generated files after tests run
@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield # Run tests
    # Teardown: remove test data/results dirs
    # Use shutil.rmtree cautiously
    # if TEST_DATA_DIR.exists():
    #     shutil.rmtree(TEST_DATA_DIR)
    if TEST_RESULTS_DIR.exists():
       shutil.rmtree(TEST_RESULTS_DIR)


def test_merge_flight_reports():
    # Ensure results dir is clean before test
    if MOCK_CONFIG["MERGED_FLIGHTS_FILE"].exists():
        os.remove(MOCK_CONFIG["MERGED_FLIGHTS_FILE"])

    df = merge_data.merge_flight_reports(force_reload=True) # Force run
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3 # 2 from file 1 + 1 from file 2
    assert MOCK_CONFIG["MERGED_FLIGHTS_FILE"].exists()
    # Optional: check content

def test_merge_weather_reports():
     if MOCK_CONFIG["MERGED_WEATHER_FILE"].exists():
        os.remove(MOCK_CONFIG["MERGED_WEATHER_FILE"])

     df = merge_data.merge_weather_reports(force_reload=True)
     assert isinstance(df, pd.DataFrame)
     assert len(df) == 4 # Only from weather_1.csv, EU pattern matches none
     assert 'timestamp' in df.columns
     assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
     assert MOCK_CONFIG["MERGED_WEATHER_FILE"].exists()

# Add more tests for merge_flights_with_weather (more complex setup)
# and tests for feature_engineering, preprocessing, train, evaluate etc.
