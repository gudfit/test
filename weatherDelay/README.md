# Weather Delay Prediction

This project is a Python-based machine learning pipeline designed to predict the duration (in minutes) of weather-related flight delays.

## What it does

1.  Loads Data: Reads historical flight performance data and corresponding weather data for specific airports and time periods.
2.  Merges Data: Combines flight and weather information based on airport and time.
3.  Engineers Features: Creates new predictive features from the raw data, including:
    *   Lagged weather conditions (weather from previous hours).
    *   Time-based cyclical features (hour, day of week, month).
    *   Weather condition flags (e.g., fog, snow, high wind).
    *   Interaction terms between weather variables.
4.  Trains Models: Trains machine learning regression models (currently supports LightGBM, XGBoost, CatBoost) to predict the `WeatherDelay` target variable. Options for weighted training (to handle imbalanced data) and hyperparameter tuning (for LGBM) are included.
5.  Evaluates Models: Assesses the performance of the trained models using various regression metrics (RMSE, MAE, R2) and classification-proxy metrics (accuracy/precision/recall based on delay thresholds).
6.  Saves Results: Stores the trained models, evaluation plots, and a summary of performance metrics.

## How to Run

Prerequisites:

*   Python 3.8+
*   The required CSV data files listed in `src/config.py` placed inside a `data/` directory in the project root.

Setup:

1.  Navigate to the project's root directory (`weatherDelay`) in your terminal.
2.  Create and activate a virtual environment (Recommended using `uv`):
    ```bash
    # Create environment
    uv venv <name>
    # Activate (Linux/macOS)
    source <name>/bin/activate
    # Activate (Windows CMD/PowerShell)
    .\<name>\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Execution:

1.  Make sure your virtual environment is active and you are in the project root directory.
2.  Run the main pipeline script:
    ```bash
    python -m src.main
    ```
3.  (Optional) Use command-line arguments to customize the run:
    ```bash
    # See available options
    python -m src.main --help

    # Example: Run without hyperparameter tuning
    python -m src.main --no-tune

    # Example: Run only the XGBoost model
    python -m src.main --models XGB
    ```

## Output

*   **Models:** Saved model files (`.joblib`) in `results/models/`.
*   **Plots:** Evaluation plots (`.png`) in `results/plots/`.
*   **Evaluation Summary:** A CSV file (`evaluation_summary_*.csv`) with performance metrics in `results/evaluation/`.
*   **Logs:** A detailed log file (`pipeline_*.log`) in `results/`.
*   **(If tuned)** Best hyperparameters (`*_best_params.json`) in `results/evaluation/`.

## Configuration

Core settings like file paths, feature engineering parameters, model configurations, and evaluation thresholds can be modified in `src/config.py`.
