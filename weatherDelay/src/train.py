# FILE: weatherDelay/src/train.py
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

from src import config, utils

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
    weights = np.ones(total_samples, dtype=np.float32)

    if method == 'binary':
        weights[y_train > 0] = config.NON_ZERO_WEIGHT_MULTIPLIER
        logging.info(f"Applied binary weighting. Non-zero weight: {config.NON_ZERO_WEIGHT_MULTIPLIER}")

    elif method == 'balanced':
        non_zero_count = (y_train > 0).sum()
        zero_count = total_samples - non_zero_count
        if non_zero_count == 0 or zero_count == 0:
            logging.warning("Cannot calculate balanced weights: All samples are zero or non-zero.")
            return None
        imbalance_ratio = zero_count / non_zero_count
        weights[y_train > 0] = imbalance_ratio
        logging.info(f"Applied class-balanced weighting. Non-zero weight: {imbalance_ratio:.4f}")

    elif method == 'quantile':
        non_zero_mask = y_train > 0
        non_zero_count = non_zero_mask.sum()
        if non_zero_count < 10:
            logging.warning(f"Not enough non-zero samples ({non_zero_count}) for quantile weighting. Falling back to 'balanced'.")
            return calculate_sample_weights(y_train)  # fallback

        y_train_nonzero = y_train[non_zero_mask].astype(np.float64)
        try:
            quantiles, bins = pd.qcut(y_train_nonzero, q=3, labels=False, retbins=True, duplicates='drop')
            logging.debug(f"Quantile bins for non-zero delays: {bins}")

            zero_count = total_samples - non_zero_count
            base_weight = zero_count / non_zero_count if non_zero_count > 0 else 1.0

            quantile_weights = {
                0: base_weight * 1.0,
                1: base_weight * 1.5,
                2: base_weight * 2.0
            }
            weights[non_zero_mask] = quantiles.map(quantile_weights).values.astype(np.float32)
            logging.info(f"Applied quantile-based weights. Weights range: {weights[non_zero_mask].min():.2f} to {weights[non_zero_mask].max():.2f}")

        except Exception as e:
            logging.error(f"Error calculating quantile weights: {e}. Falling back to 'balanced'.", exc_info=True)
            return calculate_sample_weights(y_train)
    else:
        logging.error(f"Unknown balancing method: '{method}'. No weights applied.")
        return None

    return weights

# --- Hyperparameter Tuning Setup ---
def _get_tuning_params(model_type: str) -> dict | None:
    """
    Returns hyperparameter distributions for RandomizedSearchCV.
    Updated to include XGB for demonstration with scale_pos_weight, etc.
    """
    if model_type == 'LGBM':
        return {
            'n_estimators': randint(200, 1000),
            'learning_rate': loguniform(0.005, 0.1),
            'num_leaves': randint(20, 150),
            'max_depth': randint(5, 25),
            'min_child_samples': randint(10, 100),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': loguniform(1e-3, 10.0),
            'reg_lambda': loguniform(1e-3, 10.0),
        }
    elif model_type == 'XGB':
        # EXAMPLE param distributions for XGBoost
        # This includes scale_pos_weight for imbalance
        return {
            'n_estimators': randint(200, 800),
            'learning_rate': loguniform(0.005, 0.1),
            'max_depth': randint(3, 15),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': loguniform(1e-3, 10.0),
            'reg_lambda': loguniform(1e-3, 10.0),
            'scale_pos_weight': loguniform(0.5, 30)  # <--- Tuning the imbalance weighting
        }
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
    """ Trains a single model instance (LGBM, XGB, CatBoost), with optional tuning. """
    model_name = f"{model_type}"
    if sample_weight is not None:
        model_name += "_balanced"
    if tune:
        model_name += "_tuned"
    logging.info(f"--- Training {model_name} ---")

    start_time = time.time()
    model = None
    best_params = None

    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        logging.error("X_train must be a DataFrame and y_train a Series.")
        return None, None
    if y_train.dtype not in [np.float32, np.float64]:
        y_train = y_train.astype(np.float32)
    if X_val is not None and y_val is not None:
        if y_val.dtype not in [np.float32, np.float64]:
            y_val = y_val.astype(np.float32)

    fit_params = {}
    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight

    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        # LGBM or CatBoost approach to early stopping
        fit_params["eval_set"] = eval_set
        fit_params["callbacks"] = [lgb.early_stopping(100, verbose=False)]  # LGBM callback (harmless if not LGBM)

    X_train_processed = X_train.copy()

    try:
        # --- Initialize Base Model ---
        if model_type == 'LGBM':
            base_model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
        elif model_type == 'XGB':
            base_model = xgb.XGBRegressor(**config.XGB_PARAMS)
            # XGBoost requires sanitized columns
            regex = re.compile(r"[^A-Za-z0-9_]", re.IGNORECASE)
            X_train_processed.columns = [regex.sub("_", str(x)) for x in X_train_processed.columns]
            if X_val is not None:
                X_val.columns = [regex.sub("_", str(x)) for x in X_val.columns]
            if eval_set:
                fit_params.pop("callbacks", None)
                fit_params["eval_set"] = [(X_train_processed, y_train)] + fit_params["eval_set"]
                fit_params["verbose"] = False
        elif model_type == 'CatBoost':
            base_model = cb.CatBoostRegressor(**config.CATBOOST_PARAMS)
            if eval_set:
                fit_params.pop("callbacks", None)
                fit_params["early_stopping_rounds"] = 100
                fit_params.pop("eval_set", None)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # --- Tuning if needed ---
        if tune:
            param_distributions = _get_tuning_params(model_type)
            if param_distributions:
                logging.info(f"Starting RandomizedSearch for {model_type}")
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_distributions,
                    n_iter=config.TUNING_N_ITER,
                    cv=config.TUNING_CV,
                    scoring=config.TUNING_SCORING,
                    n_jobs=-1,
                    random_state=config.RANDOM_STATE,
                    verbose=1
                )
                random_search.fit(X_train_processed, y_train,
                                  sample_weight=fit_params.get('sample_weight'))
                logging.info(f"Tuning Best Score ({config.TUNING_SCORING}): {random_search.best_score_:.4f}")
                best_params = random_search.best_params_
                logging.info(f"Tuning Best Params: {best_params}")
                model = random_search.best_estimator_
            else:
                logging.warning(f"Tuning requested for {model_type}, but no distributions found. Training with defaults.")
                model = base_model
                if model_type == 'CatBoost' and eval_set:
                    model.fit(X_train_processed, y_train, eval_set=eval_set, **fit_params)
                else:
                    model.fit(X_train_processed, y_train, **fit_params)
        else:
            logging.info(f"Training {model_type} with default/base parameters.")
            model = base_model
            if model_type == 'CatBoost' and eval_set:
                model.fit(X_train_processed, y_train, eval_set=eval_set, **fit_params)
            else:
                model.fit(X_train_processed, y_train, **fit_params)

        end_time = time.time()
        logging.info(f"Training {model_name} completed in {end_time - start_time:.2f} seconds.")
        return model, best_params

    except Exception as e:
        logging.error(f"Error training {model_name}: {e}", exc_info=True)
        return None, None
    finally:
        del X_train_processed, fit_params, eval_set
        gc.collect()

