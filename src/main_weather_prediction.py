# src/main_weather_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import time
from tqdm import tqdm # Added tqdm

# Project modules
from src import config
from src.utils import file_handler
from src.data_preprocessing import merge_data, preprocessing
from src.weather_prediction import train, evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_enhanced_feature_pipeline():
    """Runs pipeline with enhanced features, tuning, balancing, multiple models, and plotting."""
    logging.info(f"--- Starting Enhanced Feature Pipeline ({config.MODEL_SUFFIX}) ---")
    start_pipeline_time = time.time()

    # --- Load and Merge Data ---
    logging.info("Loading and merging all file pairs...")
    all_merged_list = []
    file_pairs = zip(config.WEATHER_FILES, config.FLIGHT_FILES)
    # Use tqdm for progress bar on file loading
    for weather_file, flight_file in tqdm(file_pairs, total=len(config.WEATHER_FILES), desc="Loading Data"):
        weather_fp = config.DATA_DIR / weather_file
        flight_fp = config.DATA_DIR / flight_file
        merged_df = merge_data.load_and_merge_notebook_pair(weather_fp, flight_fp)
        if not merged_df.empty:
            all_merged_list.append(merged_df)

    if not all_merged_list:
        logging.error("No data loaded. Exiting.")
        return

    combined_df = pd.concat(all_merged_list, ignore_index=True)
    logging.info(f"Combined DataFrame shape before engineering: {combined_df.shape}")
    del all_merged_list

    # --- Preprocessing including Feature Engineering ---
    logging.info("Starting Preprocessing and Feature Engineering...")
    X, y, final_feature_names = preprocessing.preprocess_notebook_style(combined_df)
    del combined_df
    if X.empty:
        logging.error("Data preparation resulted in empty features. Exiting.")
        return

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE)
    logging.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")

    # --- Calculate sample weights ---
    # Always calculate, but only use if config.USE_BALANCING is True later
    sample_weights = train.calculate_sample_weights(y_train)

    # --- Train and Evaluate All Models ---
    models_to_try = ['LGBM', 'XGB', 'CatBoost']
    all_results = []
    best_model_info = {'model': None, 'metric': np.inf, 'name': 'None'} # Track best based on RMSE

    # Use tqdm for model training loop
    for model_type in tqdm(models_to_try, desc="Training Models"):
        tune_this_model = (model_type == 'LGBM' and config.TUNING_ENABLED) # Only tune LGBM for this example

        # --- Unbalanced Run ---
        model_unbalanced = train.train_single_model(
            model_type, X_train, y_train, sample_weight=None, tune=False # Never tune unbalanced here
        )
        if model_unbalanced:
            results_unbalanced = evaluate.evaluate_single_run(
                model_unbalanced, X_test, y_test, f"{model_type}_unbalanced", final_feature_names # Pass feature names
            )
            if results_unbalanced:
                all_results.append(results_unbalanced)
                current_metric = results_unbalanced[config.BEST_MODEL_METRIC]
                if not np.isnan(current_metric) and current_metric < best_model_info['metric']:
                    best_model_info['model'] = model_unbalanced
                    best_model_info['metric'] = current_metric
                    best_model_info['name'] = results_unbalanced['model_name']
                    logging.info(f"*** New best UNBALANCED model found: {best_model_info['name']} ({config.BEST_MODEL_METRIC}: {best_model_info['metric']:.4f}) ***")

        # --- Balanced Run (potentially tuned if LGBM) ---
        model_balanced = train.train_single_model(
            model_type, X_train, y_train,
            sample_weight=sample_weights if config.USE_BALANCING else None,
            tune=tune_this_model # Tune only if enabled AND it's LGBM
        )
        if model_balanced:
            # Use balanced name, add tuned suffix if tuning was applied
            model_name = f"{model_type}{'_balanced' if config.USE_BALANCING else ''}{'_tuned' if tune_this_model else ''}"
            results_balanced = evaluate.evaluate_single_run(
                model_balanced, X_test, y_test, model_name, final_feature_names # Pass feature names
            )
            if results_balanced:
                all_results.append(results_balanced)
                current_metric = results_balanced[config.BEST_MODEL_METRIC]
                # Decide if balanced model should compete for overall 'best' or just best 'balanced'
                # Current logic compares against the single best_model_info
                if not np.isnan(current_metric) and current_metric < best_model_info['metric']:
                    best_model_info['model'] = model_balanced
                    best_model_info['metric'] = current_metric
                    best_model_info['name'] = results_balanced['model_name']
                    logging.info(f"*** New best BALANCED model found: {best_model_info['name']} ({config.BEST_MODEL_METRIC}: {best_model_info['metric']:.4f}) ***")


    # --- Save the Overall Best Model ---
    if best_model_info['model'] is not None:
        # Use the name of the best model found for the filename
        save_path = config.MODELS_DIR / f"best_overall_model_{best_model_info['name']}.joblib"
        logging.info(f"--- Saving overall best model: {best_model_info['name']} ({config.BEST_MODEL_METRIC}={best_model_info['metric']:.4f}) ---")
        file_handler.save_model(best_model_info['model'], save_path)
    else:
        logging.warning("No best model found to save.")

    # --- Save results summary ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_path = config.EVALUATION_DIR / f"evaluation_summary_{config.MODEL_SUFFIX}.csv"
        try:
            results_df = results_df.sort_values(by=config.BEST_MODEL_METRIC) # Sort by best metric
            results_df.to_csv(summary_path, index=False)
            logging.info(f"Saved evaluation summary (sorted by {config.BEST_MODEL_METRIC}) to {summary_path}")
        except Exception as e:
            logging.error(f"Failed to save evaluation summary: {e}")

    end_pipeline_time = time.time()
    logging.info(f"--- Enhanced Feature Pipeline Finished in {end_pipeline_time - start_pipeline_time:.2f} seconds ---")


if __name__ == "__main__":
    run_enhanced_feature_pipeline()
