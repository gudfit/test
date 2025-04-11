# src/weather_prediction/train.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import logging
import time
import re # Import re here

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Sample Weight Calculation ---
def calculate_sample_weights(y_train: pd.Series) -> np.ndarray:
    """Calculates sample weights giving higher weight to non-zero target values."""
    total_samples = len(y_train)
    non_zero_samples = (y_train > 0).sum()
    zero_samples = total_samples - non_zero_samples

    if non_zero_samples == 0 or zero_samples == 0:
        logging.warning("Cannot calculate weights: All samples are zero or non-zero.")
        return np.ones(total_samples)

    weights = np.ones(total_samples)
    weights[y_train > 0] = config.NON_ZERO_WEIGHT_MULTIPLIER
    logging.info(f"Calculated sample weights. Non-zero weight multiplier: {config.NON_ZERO_WEIGHT_MULTIPLIER if weights[y_train > 0].max() > 1 else 1}")
    return weights

# --- Model Training Function (includes tuning option for LGBM) ---
def train_single_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series, sample_weight=None, tune: bool = False):
    """Trains a single model instance, optionally tuning LGBM."""
    model_name = f"{model_type}{'_balanced' if sample_weight is not None else ''}{'_tuned' if tune else ''}"
    logging.info(f"--- Training {model_name} ---")

    fit_params = {}
    if sample_weight is not None and model_type != 'CatBoost':
        fit_params['sample_weight'] = sample_weight

    model = None
    start_time = time.time()
    X_train_processed = X_train.copy() # Work on a copy for potential renaming

    try:
        # --- Model Instantiation & Pre-fit Processing ---
        if model_type == 'LGBM':
            model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
        elif model_type == 'XGB':
            model = xgb.XGBRegressor(**config.XGB_PARAMS)
            # Sanitize column names for XGBoost
            regex = re.compile(r"[^A-Za-z0-9_]", re.IGNORECASE)
            X_train_processed.columns = [regex.sub("_", str(x)) for x in X_train_processed.columns]
            logging.debug("Sanitized column names for XGBoost training.")
        elif model_type == 'CatBoost':
            model = cb.CatBoostRegressor(**config.CATBOOST_PARAMS)
            if sample_weight is not None:
                 fit_params['sample_weight'] = sample_weight # Pass directly to fit
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # --- Tuning (only for LGBM in this example) ---
        if model_type == 'LGBM' and tune and config.TUNING_ENABLED:
            param_distributions = { # Define distributions inline
                'n_estimators': randint(100, 600), 'learning_rate': uniform(0.01, 0.15),
                'num_leaves': randint(20, 80), 'max_depth': [-1, 10, 15, 20, 25],
                'reg_alpha': uniform(0, 1.5), 'reg_lambda': uniform(0, 1.5),
                'colsample_bytree': uniform(0.5, 0.5), # Range 0.5 to 1.0
                'objective': ['regression', 'regression_l1']
            }
            logging.info(f"Starting RandomizedSearch for LGBM with n_iter={config.TUNING_N_ITER}, cv={config.TUNING_CV}, scoring='{config.TUNING_SCORING}'")
            # Use the base model instance here, not the pipeline
            random_search = RandomizedSearchCV(
                estimator=model, # Tune the LGBMRegressor directly
                param_distributions=param_distributions,
                n_iter=config.TUNING_N_ITER, cv=config.TUNING_CV,
                scoring=config.TUNING_SCORING, n_jobs=-1,
                random_state=config.RANDOM_STATE, verbose=1
            )
            random_search.fit(X_train_processed, y_train, **fit_params) # Fit tuner
            logging.info(f"Tuning Best Score ({config.TUNING_SCORING}): {random_search.best_score_:.4f}")
            logging.info(f"Tuning Best Params: {random_search.best_params_}")
            model = random_search.best_estimator_ # This IS the best trained LGBM model
        else:
            # --- Standard Fitting ---
            model.fit(X_train_processed, y_train, **fit_params)

        end_time = time.time()
        logging.info(f"Training {model_name} completed in {end_time - start_time:.2f} seconds.")
        return model

    except Exception as e:
        logging.error(f"Error training {model_name}: {e}", exc_info=True)
        return None
