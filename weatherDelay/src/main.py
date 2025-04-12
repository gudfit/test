# FILE: src/main.py
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import time
from tqdm import tqdm
import gc
import argparse # For command-line arguments

# Project modules
from src import config, utils, data_preprocessing, train, evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(tune_lgbm: bool = config.TUNE_LGBM,
         run_balanced: bool = config.RUN_BALANCED,
         run_unbalanced: bool = config.RUN_UNBALANCED,
         model_types: list = config.MODEL_TYPES_TO_RUN):
    """
    Main pipeline execution function.

    Args:
        tune_lgbm (bool): Whether to perform hyperparameter tuning for LGBM.
        run_balanced (bool): Whether to train models with sample weighting.
        run_unbalanced (bool): Whether to train models without sample weighting.
        model_types (list): List of model types to train (e.g., ['LGBM', 'XGB']).
    """
    pipeline_start_time = time.time()
    logging.info(f"--- Starting Weather Prediction Pipeline: {config.PIPELINE_SUFFIX} ---")
    logging.info(f"Configuration: Tune LGBM={tune_lgbm}, Run Balanced={run_balanced}, Run Unbalanced={run_unbalanced}, Models={model_types}")

    # --- 1. Load and Merge Data ---
    combined_df = data_preprocessing.load_all_data()
    if combined_df.empty:
        logging.error("Failed to load data. Exiting.")
        return

    # --- 2. Preprocessing (including Feature Engineering) ---
    # Process in chunks if necessary
    X, y, final_feature_names = data_preprocessing.process_in_chunks(combined_df)
    del combined_df # Free memory
    gc.collect()

    if X.empty or y.empty or not final_feature_names:
        logging.error("Preprocessing failed or resulted in empty data. Exiting.")
        return

    # --- 3. Split Data ---
    # Consider splitting *before* potential memory-intensive OHE if dataset is huge
    # But splitting after ensures consistent features after OHE across train/test
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.RANDOM_STATE,
        # Stratify might be useful if the target is treated as classification later
        # stratify= (y > config.CLASSIFICATION_THRESHOLDS[0]).astype(int) if config.STRATIFY_SPLIT else None
    )
    logging.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")
    del X, y # Free memory
    gc.collect()

    # --- 4. Calculate Sample Weights (if needed) ---
    sample_weights = None
    if run_balanced and config.USE_BALANCING_WEIGHTS:
        sample_weights = train.calculate_sample_weights(y_train)
        if sample_weights is None:
            logging.warning("Failed to calculate sample weights, balanced run will proceed without weights.")

    # --- 5. Train and Evaluate Models ---
    all_results = []
    best_overall_model = {'model': None, 'metric': np.inf, 'name': 'None'}

    for model_type in model_types:
        logging.info(f"--- Processing Model Type: {model_type} ---")

        # --- Unbalanced Run ---
        if run_unbalanced:
            model_ub, _ = train.train_single_model(
                model_type, X_train, y_train, X_val=X_test, y_val=y_test, # Use test as val for early stopping
                sample_weight=None, tune=False # Never tune unbalanced run here
            )
            if model_ub:
                results_ub = evaluate.evaluate_model(
                    model_ub, X_test, y_test, final_feature_names, f"{model_type}_unbalanced"
                )
                if results_ub:
                    all_results.append(results_ub)
                    current_metric = results_ub.get(config.BEST_MODEL_METRIC)
                    if current_metric is not None and np.isfinite(current_metric) and current_metric < best_overall_model['metric']:
                        best_overall_model = {'model': model_ub, 'metric': current_metric, 'name': results_ub['model_name']}
                        logging.info(f"*** New best overall model (unbalanced): {best_overall_model['name']} ({config.BEST_MODEL_METRIC}: {current_metric:.4f}) ***")
            del model_ub # Free memory
            gc.collect()

        # --- Balanced Run ---
        if run_balanced:
            should_tune = (model_type == 'LGBM' and tune_lgbm)
            model_b, best_params = train.train_single_model(
                model_type, X_train, y_train, X_val=X_test, y_val=y_test, # Use test as val for early stopping
                sample_weight=sample_weights, # Use calculated weights if available
                tune=should_tune
            )
            if model_b:
                model_name_b = f"{model_type}{'_balanced' if sample_weights is not None else ''}{'_tuned' if should_tune else ''}"
                results_b = evaluate.evaluate_model(
                    model_b, X_test, y_test, final_feature_names, model_name_b
                )
                if results_b:
                    all_results.append(results_b)
                    current_metric = results_b.get(config.BEST_MODEL_METRIC)
                    if current_metric is not None and np.isfinite(current_metric) and current_metric < best_overall_model['metric']:
                        best_overall_model = {'model': model_b, 'metric': current_metric, 'name': results_b['model_name']}
                        logging.info(f"*** New best overall model (balanced): {best_overall_model['name']} ({config.BEST_MODEL_METRIC}: {current_metric:.4f}) ***")
                    # Save best params if tuned
                    if should_tune and best_params:
                         params_path = config.EVALUATION_DIR / f"{model_name_b}_best_params.json"
                         utils.save_dict_to_json(best_params, params_path)

            del model_b # Free memory
            gc.collect()

    # --- 6. Save Results & Best Model ---
    logging.info("--- Saving Results ---")
    summary_path = config.EVALUATION_DIR / f"evaluation_summary_{config.PIPELINE_SUFFIX}.csv"
    utils.save_evaluation_summary(all_results, summary_path)

    if best_overall_model['model'] is not None:
        best_model_name = best_overall_model['name']
        best_metric_val = best_overall_model['metric']
        logging.info(f"Saving overall best model: {best_model_name} ({config.BEST_MODEL_METRIC}={best_metric_val:.4f})")
        save_path = config.MODELS_DIR / f"best_model_{config.PIPELINE_SUFFIX}_{best_model_name}.joblib"
        utils.save_model(best_overall_model['model'], save_path)
    else:
        logging.warning("No valid models were trained successfully. No best model to save.")

    pipeline_end_time = time.time()
    logging.info(f"--- Weather Prediction Pipeline Finished in {pipeline_end_time - pipeline_start_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Weather Prediction Pipeline")
    parser.add_argument("--no-tune", action="store_false", dest="tune_lgbm", default=config.TUNE_LGBM, help="Disable LGBM hyperparameter tuning")
    parser.add_argument("--no-balance", action="store_false", dest="run_balanced", default=config.RUN_BALANCED, help="Disable training balanced models")
    parser.add_argument("--no-unbalanced", action="store_false", dest="run_unbalanced", default=config.RUN_UNBALANCED, help="Disable training unbalanced models")
    parser.add_argument("--models", nargs='+', default=config.MODEL_TYPES_TO_RUN, choices=['LGBM', 'XGB', 'CatBoost'], help="Specify model types to run")

    args = parser.parse_args()

    main(tune_lgbm=args.tune_lgbm,
         run_balanced=args.run_balanced,
         run_unbalanced=args.run_unbalanced,
         model_types=args.models)
