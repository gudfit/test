# FILE: src/train.py
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
import logging
import time
import re
import gc

from src import config, utils # Import necessary modules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Sample Weight Calculation ---
def calculate_sample_weights(y_train: pd.Series) -> np.ndarray | None:
    """ Calculates sample weights based on configured method. """
    if not config.USE_BALANCING_WEIGHTS:
        logging.info("Sample weighting disabled by config.")
        return None

    method = config.BALANCING_METHOD
    logging.info(f"Calculating sample weights using method: '{method}'")
    total_samples = len(y_train)
    weights = np.ones(total_samples, dtype=np.float32) # Use float32 for weights

    if method == 'binary':
        weights[y_train > 0] = config.NON_ZERO_WEIGHT_MULTIPLIER
        logging.info(f"Applied binary weighting. Non-zero weight: {config.NON_ZERO_WEIGHT_MULTIPLIER}")

    elif method == 'balanced':
        non_zero_count = (y_train > 0).sum()
        zero_count = total_samples - non_zero_count
        if non_zero_count == 0 or zero_count == 0:
            logging.warning("Cannot calculate balanced weights: All samples are zero or non-zero.")
            return None # Return None if calculation fails
        imbalance_ratio = zero_count / non_zero_count
        weights[y_train > 0] = imbalance_ratio
        logging.info(f"Applied class-balanced weighting. Non-zero weight: {imbalance_ratio:.4f}")

    elif method == 'quantile':
        non_zero_mask = y_train > 0
        non_zero_count = non_zero_mask.sum()
        if non_zero_count < 10: # Need sufficient samples for meaningful quantiles
            logging.warning(f"Not enough non-zero samples ({non_zero_count}) for quantile weighting. Falling back to 'balanced'.")
            return calculate_sample_weights(y_train) # Re-call with fallback - careful with infinite recursion if balanced fails

        y_train_nonzero = y_train[non_zero_mask].astype(np.float64) # Use float64 for qcut stability
        try:
            # Use 3 quantiles (low, medium, high delay)
            quantiles, bins = pd.qcut(y_train_nonzero, q=3, labels=False, retbins=True, duplicates='drop')
            logging.debug(f"Quantile bins for non-zero delays: {bins}")

            # Assign weights: Higher weight for higher delay quantiles
            # Base weight similar to 'balanced' method
            zero_count = total_samples - non_zero_count
            base_weight = zero_count / non_zero_count if non_zero_count > 0 else 1.0

            quantile_weights = {
                0: base_weight * 1.0, # Low delay
                1: base_weight * 1.5, # Medium delay
                2: base_weight * 2.0  # High delay
            }
            weights[non_zero_mask] = quantiles.map(quantile_weights).values.astype(np.float32)
            logging.info(f"Applied quantile-based weights. Weights range: {weights[non_zero_mask].min():.2f} to {weights[non_zero_mask].max():.2f}")

        except Exception as e:
            logging.error(f"Error calculating quantile weights: {e}. Falling back to 'balanced'.", exc_info=True)
            return calculate_sample_weights(y_train) # Re-call with fallback

    else:
        logging.error(f"Unknown balancing method: '{method}'. No weights applied.")
        return None

    # Normalize weights to have an average of 1 (optional, but can sometimes help stability)
    # weights /= np.mean(weights)
    return weights

# --- Hyperparameter Tuning Setup ---
def _get_tuning_params(model_type: str) -> dict | None:
    """ Returns hyperparameter distributions for RandomizedSearchCV. """
    if model_type == 'LGBM':
        # Loguniform is good for rates, powers of 2/10
        # Uniform for fractions/additive params
        # randint for counts/indices
        return {
            'n_estimators': randint(200, 1000),
            'learning_rate': loguniform(0.005, 0.1),
            'num_leaves': randint(20, 150),
            'max_depth': randint(5, 25), # Or list like [-1, 10, 15, 20]
            'min_child_samples': randint(10, 100),
            'subsample': uniform(0.6, 0.4), # loc=0.6, scale=0.4 -> range [0.6, 1.0)
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': loguniform(1e-3, 10.0),
            'reg_lambda': loguniform(1e-3, 10.0),
            #'boosting_type': ['gbdt', 'dart'] # Optional
        }
    # Add distributions for XGBoost or CatBoost if tuning is implemented for them
    elif model_type == 'XGB':
         logging.warning("Tuning parameters not defined for XGBoost in this setup.")
         return None
    elif model_type == 'CatBoost':
         logging.warning("Tuning parameters not defined for CatBoost in this setup.")
         return None
    else:
        return None

# --- Model Training ---
def train_single_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None,
                       sample_weight: np.ndarray | None = None,
                       tune: bool = False):
    """ Trains a single model instance (LGBM, XGB, CatBoost), with optional tuning (LGBM only currently)."""

    model_name = f"{model_type}"
    if sample_weight is not None: model_name += "_balanced"
    if tune: model_name += "_tuned"
    logging.info(f"--- Training {model_name} ---")

    start_time = time.time()
    model = None
    best_params = None # Store best params if tuned

    # --- Data Type Checks & Preparation ---
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        logging.error("X_train must be a DataFrame and y_train a Series.")
        return None, None
    if y_train.dtype not in [np.float32, np.float64]:
        logging.warning(f"y_train dtype is {y_train.dtype}. Converting to float32.")
        y_train = y_train.astype(np.float32)
    if X_val is not None and y_val is not None:
        if y_val.dtype not in [np.float32, np.float64]:
             y_val = y_val.astype(np.float32)

    # Prepare fit parameters (including early stopping if validation set provided)
    fit_params = {}
    if sample_weight is not None:
        # CatBoost uses 'sample_weight' directly in fit
        fit_params['sample_weight'] = sample_weight

    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        fit_params["eval_set"] = eval_set
        fit_params["callbacks"] = [lgb.early_stopping(100, verbose=False)] # LGBM specific callback

    X_train_processed = X_train.copy() # Work on copy for potential modifications

    try:
        # --- Model Initialization ---
        if model_type == 'LGBM':
            base_model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
        elif model_type == 'XGB':
            base_model = xgb.XGBRegressor(**config.XGB_PARAMS)
            # XGBoost requires sanitized feature names
            regex = re.compile(r"[^A-Za-z0-9_]", re.IGNORECASE)
            X_train_processed.columns = [regex.sub("_", str(x)) for x in X_train_processed.columns]
            if X_val is not None:
                X_val.columns = [regex.sub("_", str(x)) for x in X_val.columns]
                fit_params["eval_set"] = [(X_val, y_val)] # Update eval set with renamed cols
            # XGBoost early stopping uses 'early_stopping_rounds' in fit()
            if eval_set:
                fit_params.pop("callbacks", None) # Remove LGBM callback
                fit_params["early_stopping_rounds"] = 100
                fit_params["eval_set"] = [(X_train_processed, y_train)] + fit_params["eval_set"] # Add train set for eval metrics
                fit_params["verbose"] = False # Suppress XGBoost verbosity during fit

        elif model_type == 'CatBoost':
            base_model = cb.CatBoostRegressor(**config.CATBOOST_PARAMS)
            # CatBoost early stopping uses 'early_stopping_rounds' in fit()
            if eval_set:
                fit_params.pop("callbacks", None) # Remove LGBM callback
                fit_params["early_stopping_rounds"] = 100
                # CatBoost needs eval_set directly in fit(), not fit_params dict
                fit_params.pop("eval_set", None)
            # Sample weight handled directly in fit_params for CatBoost

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # --- Tuning (Currently LGBM only) ---
        if model_type == 'LGBM' and tune:
            param_distributions = _get_tuning_params(model_type)
            if param_distributions:
                logging.info(f"Starting RandomizedSearch for {model_type} (n_iter={config.TUNING_N_ITER}, cv={config.TUNING_CV})")
                # Note: CV uses StratifiedKFold for classification, KFold for regression by default
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_distributions,
                    n_iter=config.TUNING_N_ITER,
                    cv=config.TUNING_CV,
                    scoring=config.TUNING_SCORING,
                    n_jobs=-1, # Use all available cores
                    random_state=config.RANDOM_STATE,
                    verbose=1 # Show progress
                )
                # Fit the tuner (uses base model fit method internally)
                # Pass sample weights for balanced tuning
                random_search.fit(X_train_processed, y_train,
                                  sample_weight=fit_params.get('sample_weight')) # Pass weight if exists

                logging.info(f"Tuning Best Score ({config.TUNING_SCORING}): {random_search.best_score_:.4f}")
                best_params = random_search.best_params_
                logging.info(f"Tuning Best Params: {best_params}")
                # Use the best estimator found by the search
                model = random_search.best_estimator_
                # Tuned model is already fitted by RandomizedSearchCV

            else:
                logging.warning(f"Tuning requested for {model_type}, but no distributions found. Training with default params.")
                model = base_model
                if model_type == 'CatBoost' and eval_set:
                     model.fit(X_train_processed, y_train, eval_set=eval_set, **fit_params)
                else:
                     model.fit(X_train_processed, y_train, **fit_params)
        else:
            # --- Standard Fitting (No Tuning or Tuner Disabled) ---
            logging.info(f"Training {model_type} with default/base parameters.")
            model = base_model
            # CatBoost needs eval_set passed directly, others via fit_params
            if model_type == 'CatBoost' and eval_set:
                model.fit(X_train_processed, y_train, eval_set=eval_set, **fit_params)
            else:
                model.fit(X_train_processed, y_train, **fit_params)


        end_time = time.time()
        logging.info(f"Training {model_name} completed in {end_time - start_time:.2f} seconds.")
        return model, best_params # Return model and best params (if tuned)

    except Exception as e:
        logging.error(f"Error training {model_name}: {e}", exc_info=True)
        return None, None
    finally:
        # Clean up memory
        del X_train_processed, fit_params, eval_set
        if 'random_search' in locals():
            del random_search
        gc.collect()
