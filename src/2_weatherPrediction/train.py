# src/2_weatherPrediction/train.py
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

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Improved Sample Weight Calculation ---
def calculate_sample_weights(y_train: pd.Series) -> np.ndarray:
    """
    Calculates sample weights based on target value distribution.
    Uses a more sophisticated approach than simple binary weighting.
    """
    total_samples = len(y_train)
    non_zero_samples = (y_train > 0).sum()
    zero_samples = total_samples - non_zero_samples
    
    if non_zero_samples == 0 or zero_samples == 0:
        logging.warning("Cannot calculate weights: All samples are zero or non-zero.")
        return np.ones(total_samples)
    
    # Calculate imbalance ratio
    imbalance_ratio = zero_samples / non_zero_samples
    logging.info(f"Class imbalance ratio (zeros/non-zeros): {imbalance_ratio:.2f}")
    
    # Initialize weights
    weights = np.ones(total_samples)
    
    # Option 1: Simple binary weighting (original approach)
    if config.BALANCING_METHOD == 'binary':
        weights[y_train > 0] = config.NON_ZERO_WEIGHT_MULTIPLIER
        logging.info(f"Applied binary weighting with multiplier: {config.NON_ZERO_WEIGHT_MULTIPLIER}")
        
    # Option 2: Class-balanced weighting
    elif config.BALANCING_METHOD == 'balanced':
        # Set weight for each class inversely proportional to class frequency
        weights[y_train == 0] = 1.0
        weights[y_train > 0] = imbalance_ratio
        logging.info(f"Applied class-balanced weighting. Non-zero weight: {imbalance_ratio:.2f}")
        
    # Option 3: Quantile-based weighting (for regression)
    elif config.BALANCING_METHOD == 'quantile':
        # Group non-zero values into quantiles and assign weights
        if non_zero_samples >= 3:  # Need enough samples for quantiles
            non_zero_mask = y_train > 0
            
            # Make a copy of the non-zero values to avoid modifying original
            y_train_nonzero = y_train[non_zero_mask].copy()
            
            # Convert to float64 to avoid pandas qcut error with float16
            if y_train_nonzero.dtype != np.float64:
                y_train_nonzero = y_train_nonzero.astype(np.float64)
            
            try:
                # Check for duplicates and add tiny noise if needed
                value_counts = y_train_nonzero.value_counts()
                has_duplicates = any(count > 1 for count in value_counts)
                
                if has_duplicates and len(y_train_nonzero) > 3:
                    # Add tiny random noise to break ties
                    logging.info("Adding small noise to break ties in quantile calculation")
                    noise = np.random.uniform(-0.001, 0.001, size=len(y_train_nonzero))
                    y_train_nonzero = y_train_nonzero + noise
                
                # Try with 3 quantiles
                num_quantiles = min(3, len(y_train_nonzero.unique()))
                if num_quantiles < 3:
                    logging.warning(f"Too few unique non-zero values ({num_quantiles}), using binary weighting")
                    weights[y_train > 0] = config.NON_ZERO_WEIGHT_MULTIPLIER
                    return weights
                
                quantiles = pd.qcut(y_train_nonzero, num_quantiles, duplicates='drop')
                
                # Map quantiles to weights (higher quantiles get higher weights)
                quantile_weights = {}
                quantile_values = sorted(quantiles.unique())
                
                for i, q in enumerate(quantile_values):
                    # Use progressively higher weights for higher delay quantiles
                    quantile_weights[q] = imbalance_ratio * (1 + i * 0.5)
                
                # Apply weights to non-zero samples based on their quantile
                weights[non_zero_mask] = quantiles.map(quantile_weights).values
                
                logging.info(f"Applied quantile-based weights ranging from {min(weights):.2f} to {max(weights):.2f}")
                
            except Exception as e:
                logging.warning(f"Error calculating quantile weights: {str(e)}")
                # Fall back to binary if quantiles fail
                weights[y_train > 0] = config.NON_ZERO_WEIGHT_MULTIPLIER
                logging.info(f"Falling back to binary weighting with multiplier: {config.NON_ZERO_WEIGHT_MULTIPLIER}")
        else:
            # Fall back to binary if not enough samples
            weights[y_train > 0] = config.NON_ZERO_WEIGHT_MULTIPLIER
            logging.info(f"Not enough non-zero samples for quantile weighting, using binary method")
    
    else:
        logging.warning(f"Unknown balancing method '{config.BALANCING_METHOD}', using binary weights")
        weights[y_train > 0] = config.NON_ZERO_WEIGHT_MULTIPLIER
    
    return weights

# --- Hyperparameter Tuning Distributions ---
def get_tuning_params(model_type: str) -> dict:
    """Returns hyperparameter distributions for tuning based on model type."""
    
    if model_type == 'LGBM':
        return {
            'n_estimators': randint(100, 800),
            'learning_rate': loguniform(0.005, 0.2),
            'num_leaves': randint(20, 150),
            'max_depth': [-1, 5, 10, 15, 20, 25],
            'min_child_samples': randint(10, 100),
            'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
            'reg_alpha': loguniform(1e-3, 10),
            'reg_lambda': loguniform(1e-3, 10),
            'objective': ['regression', 'regression_l1', 'huber']
        }
    
    elif model_type == 'XGB':
        return {
            'n_estimators': randint(100, 800),
            'learning_rate': loguniform(0.005, 0.2),
            'max_depth': randint(3, 15),
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 1),
            'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
            'reg_alpha': loguniform(1e-3, 10),
            'reg_lambda': loguniform(1e-3, 10),
            'objective': ['reg:squarederror', 'reg:pseudohubererror']
        }
    
    elif model_type == 'CatBoost':
        return {
            'iterations': randint(100, 800),
            'learning_rate': loguniform(0.005, 0.2),
            'depth': randint(4, 12),
            'l2_leaf_reg': loguniform(1, 100),
            'border_count': randint(32, 255),
            'bagging_temperature': uniform(0, 1),
            'random_strength': uniform(0, 1),
            'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
            'loss_function': ['RMSE', 'MAE', 'Quantile:alpha=0.5']
        }
    
    else:
        logging.warning(f"No tuning parameters defined for model type: {model_type}")
        return {}

# --- Model Training Function (includes tuning for all model types) ---
def train_single_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series, sample_weight=None, tune: bool = False):
    """Trains a single model instance, with tuning support for all model types."""
    model_name = f"{model_type}{'_balanced' if sample_weight is not None else ''}{'_tuned' if tune else ''}"
    logging.info(f"--- Training {model_name} ---")

    fit_params = {}
    if sample_weight is not None:
        if model_type != 'CatBoost':
            fit_params['sample_weight'] = sample_weight
        else:
            # CatBoost uses a different parameter name for sample weights
            fit_params['sample_weight'] = sample_weight

    model = None
    start_time = time.time()
    X_train_processed = X_train.copy()  # Work on a copy for potential renaming
    
    # Make sure target value is float64 for stable training
    if y_train.dtype != np.float64:
        y_train = y_train.astype(np.float64)

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

        # --- Tuning (now available for all models) ---
        if tune and config.TUNING_ENABLED:
            param_distributions = get_tuning_params(model_type)
            
            if not param_distributions:
                logging.warning(f"Tuning enabled but no parameters defined for {model_type}. Using default parameters.")
                model.fit(X_train_processed, y_train, **fit_params)
                
            else:
                logging.info(f"Starting RandomizedSearch for {model_type} with n_iter={config.TUNING_N_ITER}, cv={config.TUNING_CV}")
                
                # Create RandomizedSearchCV object
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_distributions,
                    n_iter=config.TUNING_N_ITER,
                    cv=config.TUNING_CV,
                    scoring=config.TUNING_SCORING,
                    n_jobs=-1,
                    random_state=config.RANDOM_STATE,
                    verbose=1
                )
                
                # Fit with appropriate parameters
                random_search.fit(X_train_processed, y_train, **fit_params)
                
                logging.info(f"Tuning Best Score ({config.TUNING_SCORING}): {random_search.best_score_:.4f}")
                logging.info(f"Tuning Best Params: {random_search.best_params_}")
                
                # Use the best model as our final model
                model = random_search.best_estimator_
        else:
            # --- Standard Fitting ---
            model.fit(X_train_processed, y_train, **fit_params)

        end_time = time.time()
        logging.info(f"Training {model_name} completed in {end_time - start_time:.2f} seconds.")
        return model

    except Exception as e:
        logging.error(f"Error training {model_name}: {e}", exc_info=True)
        return None
