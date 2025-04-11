# src/main_weather_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import time
from tqdm import tqdm
import gc  # Add garbage collector

# Project modules
from src import config
from src.utils import file_handler
from src.dataPreprocessing.weatherPredictionProcessing import merge_data, preprocessing
from src.weatherPrediction import train, evaluate

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
        
        # Use the enhanced merge function that adds lag features
        merged_df = merge_data.load_and_merge_with_lag_features(weather_fp, flight_fp)
        
        if not merged_df.empty:
            # Calculate percentage of rows with weather data
            if 'weather_data_count' in merged_df.columns:
                rows_with_weather = (merged_df['weather_data_count'] > 0).sum()
                pct_with_weather = rows_with_weather / len(merged_df) * 100
                logging.info(f"File pair has {pct_with_weather:.2f}% rows with weather data")
                
                # Drop rows without any weather data to improve quality
                if pct_with_weather < 90:  # If we have poor weather coverage
                    original_len = len(merged_df)
                    merged_df = merged_df[merged_df['weather_data_count'] > 0].copy()
                    logging.info(f"Filtered to only rows with weather data: {len(merged_df)} of {original_len} rows kept")
                
                # Drop the helper column
                merged_df = merged_df.drop('weather_data_count', axis=1)
            
            # MEMORY OPTIMIZATION: Handle categorical columns early
            # Convert categorical columns to category type to save memory
            for col in ['Reporting_Airline', 'Origin', 'Dest']:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].astype('category')
            
            all_merged_list.append(merged_df)
        
        # Force garbage collection after each file pair
        gc.collect()

    if not all_merged_list:
        logging.error("No data loaded. Exiting.")
        return

    # Combine all data
    combined_df = pd.concat(all_merged_list, ignore_index=True)
    logging.info(f"Combined DataFrame shape before engineering: {combined_df.shape}")
    
    # Free up memory
    del all_merged_list
    gc.collect()

    # --- Data Quality Check ---
    # Count non-null values in weather columns to ensure proper merge
    weather_cols = [col for col in combined_df.columns if any(
        col.startswith(f"{feat}_origin") or col.startswith(f"{feat}_dest") 
        for feat in config.WEATHER_FEATURES_BASE
    )]
    
    if weather_cols:
        weather_cols_pct = combined_df[weather_cols].notna().mean() * 100
        avg_completeness = weather_cols_pct.mean()
        logging.info(f"Weather data completeness: {avg_completeness:.2f}% on average")
        
        # Log a few examples of the weather columns
        sample_cols = weather_cols[:min(5, len(weather_cols))]
        for col in sample_cols:
            pct_present = weather_cols_pct[col]
            logging.info(f"  - {col}: {pct_present:.2f}% non-null")
    else:
        logging.warning("No weather columns found in merged data!")

    # --- MEMORY OPTIMIZATION: Apply chunking for large datasets ---
    max_rows_per_chunk = 400000  # Adjust based on your available memory
    
    if len(combined_df) > max_rows_per_chunk:
        logging.info(f"Dataset too large ({len(combined_df)} rows), processing in chunks")
        
        # Find a threshold to split the data
        n_chunks = int(np.ceil(len(combined_df) / max_rows_per_chunk))
        chunk_results = []
        
        for i in range(n_chunks):
            start_idx = i * max_rows_per_chunk
            end_idx = min((i + 1) * max_rows_per_chunk, len(combined_df))
            
            logging.info(f"Processing chunk {i+1}/{n_chunks} (rows {start_idx}-{end_idx})")
            chunk_df = combined_df.iloc[start_idx:end_idx].copy()
            
            # Process this chunk
            X_chunk, y_chunk, _ = preprocessing.preprocess_notebook_style(chunk_df)
            
            # Store results
            if not X_chunk.empty:
                chunk_results.append((X_chunk, y_chunk))
            
            # Free memory
            del chunk_df
            gc.collect()
        
        # Combine results from all chunks
        if chunk_results:
            X = pd.concat([res[0] for res in chunk_results], ignore_index=True)
            y = pd.concat([res[1] for res in chunk_results], ignore_index=True)
            
            # Make sure we have all features from all chunks
            final_feature_names = X.columns.tolist()
            
            logging.info(f"Combined chunks: X={X.shape}, y={y.shape}")
        else:
            logging.error("No valid data after chunk processing")
            return
            
        # Free memory
        del chunk_results
        gc.collect()
        
    else:
        # For smaller datasets, process normally
        logging.info("Starting Preprocessing and Feature Engineering...")
        X, y, final_feature_names = preprocessing.preprocess_notebook_style(combined_df)
    
    # Free up memory
    del combined_df
    gc.collect()
    
    if X.empty:
        logging.error("Data preparation resulted in empty features. Exiting.")
        return

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE)
    logging.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")
    
    # Free up memory
    del X, y
    gc.collect()

    # --- Feature Importance Check ---
    # Log which features made it to the training set
    categorical_features = [f for f in final_feature_names if '_' in f and any(cat in f for cat in ['Origin', 'Dest', 'Reporting_Airline'])]
    weather_features = [f for f in final_feature_names if any(base in f for base in config.WEATHER_FEATURES_BASE)]
    engineered_features = [f for f in final_feature_names if any(f in l for l in [
        config.WEATHER_DESC_FEATURES_ALL, 
        config.CYCLICAL_FEATURES_ALL,
        config.THRESHOLD_FEATURES_ALL,
        config.TREND_FEATURES_ALL,
        config.INTERACTION_FEATURES_ALL
    ])]
    
    logging.info(f"Feature categories in training data:")
    logging.info(f"  - Categorical features: {len(categorical_features)}")
    logging.info(f"  - Weather features: {len(weather_features)}")
    logging.info(f"  - Engineered features: {len(engineered_features)}")
    
    # Log delay statistics
    non_zero_delays = (y_train > 0).sum()
    delay_ratio = non_zero_delays / len(y_train)
    logging.info(f"Target variable statistics:")
    logging.info(f"  - Non-zero delays: {non_zero_delays} ({delay_ratio:.2%})")
    logging.info(f"  - Zero delays: {len(y_train) - non_zero_delays} ({1-delay_ratio:.2%})")
    logging.info(f"  - Class imbalance ratio (zeros/non-zeros): {(len(y_train) - non_zero_delays) / non_zero_delays:.2f}")

    # --- Calculate sample weights ---
    # Make sure target is float64 before weight calculation to avoid float16 error
    if y_train.dtype == np.float16:
        y_train = y_train.astype(np.float64)
        
    sample_weights = train.calculate_sample_weights(y_train)

    # --- Train and Evaluate All Models ---
    # Start with just one model to avoid memory issues
    models_to_try = ['LGBM']  # Start with just LGBM, can add 'XGB', 'CatBoost' later
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
