# FILE: src/evaluate.py
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, median_absolute_error,
                             mean_absolute_percentage_error,
                             confusion_matrix, classification_report, accuracy_score,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path
import re
import gc
from typing import List, Dict, Any, Tuple
import time

# Model specific types for feature importance check (optional, good for type hinting)
try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    from statsmodels.nonparametric.smoothers_lowess import lowess # For residuals plot
    from scipy import stats # For error distribution plot
    _PLOT_EXTRAS = True
except ImportError:
    lgb = xgb = cb = None # Set to None if optional dependencies missing
    lowess = stats = None
    _PLOT_EXTRAS = False
    logging.warning("Optional plotting dependencies (statsmodels, scipy) or model libraries not found. Some plot features disabled.")


from src import config, utils # Import necessary modules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Plotting Helpers ---

def _create_plot_filename(base_name: str, plot_type: str) -> Path:
    """ Creates a standardized plot filename. """
    plot_dir = config.PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / f"{base_name}_{plot_type}.png"

def _safe_plot_save(func):
    """ Decorator to handle errors during plot generation/saving. """
    def wrapper(*args, **kwargs):
        plot_path = kwargs.get('file_path', args[2] if len(args) > 2 and isinstance(args[2], Path) else None)
        plot_desc = kwargs.get('title_suffix', args[3] if len(args) > 3 else "")
        try:
            func(*args, **kwargs)
            if plot_path:
                logging.info(f"Plot saved: {plot_path.name}")
        except Exception as e:
            logging.error(f"Failed to generate/save plot '{plot_desc}' ({plot_path}): {e}", exc_info=False) # Less verbose log
        finally:
            plt.close('all') # Ensure plot is closed even on error
            gc.collect()
    return wrapper

@_safe_plot_save
def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    """ Plots predicted vs actual values using hexbin for density. """
    logging.debug(f"Generating Predicted vs Actual plot {title_suffix}...")
    if len(y_test) == 0: return
    sample_size = min(len(y_test), config.PLOT_SAMPLE_SIZE)
    indices = np.random.choice(np.arange(len(y_test)), sample_size, replace=False)
    y_test_sample = y_test.iloc[indices]
    y_pred_sample = y_pred[indices]

    plt.figure(figsize=(10, 8))
    plt.hexbin(y_test_sample, y_pred_sample, gridsize=50, cmap=config.PLOT_CMAP, mincnt=1, alpha=0.8)
    plt.colorbar(label='Count in Bin')

    min_val = min(0, y_test_sample.min(), y_pred_sample.min()) # Ensure axis starts at 0 or below
    max_val = max(y_test_sample.max(), y_pred_sample.max()) * 1.05 # Add padding
    plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Ideal (y=x)')

    plt.title(f'Predicted vs. Actual {title_suffix}', fontsize=14)
    plt.xlabel('Actual Weather Delay (Minutes)', fontsize=12)
    plt.ylabel('Predicted Weather Delay (Minutes)', fontsize=12)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=config.PLOT_DPI)

@_safe_plot_save
def plot_residuals_vs_predicted(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    """ Plots residuals vs. predicted values with LOWESS trend. """
    logging.debug(f"Generating Residuals vs Predicted plot {title_suffix}...")
    if len(y_test) == 0: return
    residuals = y_test - y_pred
    sample_size = min(len(y_test), config.PLOT_SAMPLE_SIZE)
    indices = np.random.choice(np.arange(len(y_test)), sample_size, replace=False)
    y_pred_sample = y_pred[indices]
    residuals_sample = residuals.iloc[indices]

    plt.figure(figsize=(10, 8))
    plt.hexbin(y_pred_sample, residuals_sample, gridsize=50, cmap='coolwarm', mincnt=1, alpha=0.7)
    plt.colorbar(label='Count in Bin')
    plt.axhline(0, color='black', linestyle='--', linewidth=2, label='Zero Residual')

    # Add LOWESS trend line if statsmodels is available
    if _PLOT_EXTRAS and lowess:
        try:
            smoothed = lowess(residuals_sample, y_pred_sample, frac=0.3)
            plt.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=3, label='LOWESS Trend')
        except Exception as lowess_err:
            logging.warning(f"Could not compute LOWESS for residuals plot: {lowess_err}")

    plt.title(f'Residuals vs. Predicted {title_suffix}', fontsize=14)
    plt.xlabel('Predicted Weather Delay (Minutes)', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=config.PLOT_DPI)

@_safe_plot_save
def plot_error_distribution(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    """ Plots the distribution of prediction errors with stats. """
    logging.debug(f"Generating Error Distribution plot {title_suffix}...")
    if len(y_test) == 0: return
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True, stat="density")

    # Add Normal curve comparison if scipy.stats available
    if _PLOT_EXTRAS and stats:
        try:
            mu, sigma = stats.norm.fit(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=2,
                     label=f'Normal Fit (Î¼={mu:.2f}, Ïƒ={sigma:.2f})')
            kurt = stats.kurtosis(errors)
            skew = stats.skew(errors)
            stats_text = f"Mean: {errors.mean():.2f}\nMedian: {errors.median():.2f}\nStd Dev: {errors.std():.2f}\nSkew: {skew:.2f}\nKurtosis: {kurt:.2f}"
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        except Exception as stats_err:
            logging.warning(f"Could not compute stats/fit for error distribution plot: {stats_err}")

    plt.axvline(errors.mean(), color='r', linestyle='-', linewidth=1.5, label=f'Mean Error ({errors.mean():.2f})')
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.title(f"Prediction Error Distribution {title_suffix}", fontsize=14)
    plt.xlabel("Error (Actual - Predicted Minutes)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=config.PLOT_DPI)

@_safe_plot_save
def plot_feature_importance(model, feature_names: List[str], file_path: Path, title_suffix: str = ""):
    """ Plots feature importances for tree-based models. """
    logging.debug(f"Generating Feature Importance plot {title_suffix}...")
    importance = None
    fnames_used = feature_names # Default to provided names

    try:
        if lgb and isinstance(model, lgb.LGBMRegressor):
            importance = model.feature_importances_
            fnames_used = model.feature_name_ # Use names from model if available
            plot_title = f"LGBM Feature Importance {title_suffix}"
        elif xgb and isinstance(model, xgb.XGBRegressor):
            importance = model.feature_importances_
            # XGBoost doesn't store feature names in the same way by default after joblib load
            # We rely on the passed feature_names, ensuring they match the training order.
            plot_title = f"XGBoost Feature Importance {title_suffix}"
        elif cb and isinstance(model, cb.CatBoostRegressor):
            importance = model.get_feature_importance()
            fnames_used = model.feature_names_ if hasattr(model, 'feature_names_') else feature_names
            plot_title = f"CatBoost Feature Importance {title_suffix}"
        else:
            logging.warning(f"Feature importance plot not supported or library missing for model type: {type(model)}. Skipping.")
            return

        if importance is None:
            logging.warning("Could not retrieve feature importances.")
            return
        if len(fnames_used) != len(importance):
            logging.error(f"Feature name length mismatch! Got {len(fnames_used)} names, {len(importance)} importances. Cannot plot.")
            return

        feat_imp_df = pd.DataFrame({'feature': fnames_used, 'importance': importance})
        feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False).head(25) # Show top 25

        plt.figure(figsize=(10, 10))
        sns.barplot(x='importance', y='feature', data=feat_imp_df, hue='feature', palette='viridis', legend=False)
        plt.title(plot_title, fontsize=14)
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(file_path, dpi=config.PLOT_DPI)

    except Exception as e:
        logging.error(f"Error generating feature importance plot: {e}", exc_info=False)

# FILE: src/evaluate.py
# ... (other imports and plotting functions) ...

@_safe_plot_save
def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, threshold: float = 0, title_suffix: str = ""):
    """ Creates a visual confusion matrix heatmap for a given threshold. """
    logging.debug(f"Generating Confusion Matrix plot (Threshold > {threshold}) {title_suffix}...")
    if len(y_test) == 0: return

    # Convert to binary classification based on threshold
    y_test_binary = (y_test > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    # Calculate confusion matrix
    try:
        cm = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except ValueError: # Handle cases where only one class is present/predicted
        logging.warning(f"Could not unpack full confusion matrix for threshold {threshold}. Adjusting plot.")
        cm = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
        # Initialize to avoid errors later, logic might need adjustment based on desired plot for edge cases
        tn, fp, fn, tp = (cm[0,0], cm[0,1], cm[1,0], cm[1,1]) if cm.shape == (2,2) else (0,0,0,0)
        # A simple fallback if only one class exists (e.g., all TN or all TP)
        if cm.shape == (1,1):
             if y_test_binary.iloc[0] == 0: tn = cm[0,0]
             else: tp = cm[0,0]


    # Calculate metrics for annotation (optional, but useful on the plot)
    accuracy = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Create plot
    plt.figure(figsize=(10, 8)) # Slightly larger for annotations

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred No Delay', 'Pred Delay'],
                yticklabels=['True No Delay', 'True Delay'],
                annot_kws={"size": 12}) # Adjust annotation size

    # Add annotations for clarity
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Annotate each cell with percentage (optional)
    total = tn + fp + fn + tp
    if total > 0:
        for i, val_row in enumerate(cm):
            for j, val in enumerate(val_row):
                 plt.text(j + 0.5, i + 0.7, f'({val / total:.1%})',
                          ha='center', va='center', fontsize=10, color='black' if val < total/2 else 'white')

    plt.title(f'Confusion Matrix (Threshold > {threshold}) {title_suffix}', fontsize=14)

    # Add metrics as a text box outside the plot (optional)
    metrics_text = (f"Accuracy: {accuracy:.3f}\n"
                   f"Precision: {precision:.3f}\n"
                   f"Recall: {recall:.3f}\n"
                   f"F1 Score: {f1:.3f}\n"
                   f"Specificity: {specificity:.3f}")

    # Place text box to the right
    plt.text(1.1, 0.5, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    plt.subplots_adjust(right=0.75) # Make space for the text box
    plt.savefig(file_path, dpi=config.PLOT_DPI, bbox_inches='tight')
# --- Evaluation Metrics Calculation ---

def calculate_regression_metrics(y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """ Calculates standard regression metrics. """
    if len(y_test) == 0: return {}
    metrics = {}
    metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['mae'] = mean_absolute_error(y_test, y_pred)
    metrics['median_ae'] = median_absolute_error(y_test, y_pred)
    metrics['r2'] = r2_score(y_test, y_pred)

    # Accuracy within N minutes
    for mins in [5, 15, 30]:
         metrics[f'acc_{mins}min'] = np.mean(np.abs(y_test - y_pred) <= mins) * 100

    # MAPE - handle division by zero
    mask = y_test != 0
    if np.any(mask):
        mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100
        metrics['mape'] = mape if np.isfinite(mape) else None # Store None if MAPE is inf
    else:
        metrics['mape'] = None # Cannot calculate if all actuals are 0

    # Non-zero metrics
    non_zero_mask = y_test > 0
    metrics['count_non_zero'] = int(non_zero_mask.sum())
    if metrics['count_non_zero'] > 0:
        y_test_nz = y_test[non_zero_mask]
        y_pred_nz = y_pred[non_zero_mask]
        metrics['rmse_non_zero'] = np.sqrt(mean_squared_error(y_test_nz, y_pred_nz))
        metrics['mae_non_zero'] = mean_absolute_error(y_test_nz, y_pred_nz)
        metrics['r2_non_zero'] = r2_score(y_test_nz, y_pred_nz)
        mape_nz = mean_absolute_percentage_error(y_test_nz, y_pred_nz) * 100
        metrics['mape_non_zero'] = mape_nz if np.isfinite(mape_nz) else None
    else:
        metrics['rmse_non_zero'] = None
        metrics['mae_non_zero'] = None
        metrics['r2_non_zero'] = None
        metrics['mape_non_zero'] = None

    return metrics

def calculate_classification_proxy_metrics(y_test: pd.Series, y_pred: np.ndarray, threshold: float) -> Dict[str, Any]:
    """ Calculates classification metrics based on a delay threshold. """
    if len(y_test) == 0: return {}
    metrics = {}
    prefix = f"clf_thresh{threshold}" # Example: clf_thresh0

    y_test_binary = (y_test > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    metrics[f'{prefix}_accuracy'] = accuracy_score(y_test_binary, y_pred_binary)

    # --- Ensure these keys are being populated ---
    try:
        tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1]).ravel()
        metrics[f'{prefix}_tn'] = int(tn)
        metrics[f'{prefix}_fp'] = int(fp)
        metrics[f'{prefix}_fn'] = int(fn)
        metrics[f'{prefix}_tp'] = int(tp)
    except ValueError:
        # ... (handle edge case as before) ...
        cm = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
        # Ensure these keys are still added, even if potentially 0
        metrics[f'{prefix}_tn'] = int(cm[0,0]) if cm.shape == (2,2) else int(cm[0,0]) if cm.shape == (1,1) and y_test_binary.iloc[0] == 0 else 0
        metrics[f'{prefix}_fp'] = int(cm[0,1]) if cm.shape == (2,2) else 0
        metrics[f'{prefix}_fn'] = int(cm[1,0]) if cm.shape == (2,2) else 0
        metrics[f'{prefix}_tp'] = int(cm[1,1]) if cm.shape == (2,2) else int(cm[0,0]) if cm.shape == (1,1) and y_test_binary.iloc[0] == 1 else 0


    precision, recall, f1, _ = precision_recall_fscore_support(y_test_binary, y_pred_binary, labels=[0, 1], zero_division=0)
    metrics[f'{prefix}_precision_0'] = precision[0]
    metrics[f'{prefix}_recall_0'] = recall[0]      # Specificity is recall of class 0
    metrics[f'{prefix}_f1_0'] = f1[0]
    metrics[f'{prefix}_precision_1'] = precision[1] # Precision for delay class
    metrics[f'{prefix}_recall_1'] = recall[1]      # Recall for delay class
    metrics[f'{prefix}_f1_1'] = f1[1]              # F1 for delay class

    # Explicitly add specificity (same as recall_0) and prevalence
    metrics[f'{prefix}_specificity'] = recall[0]

    total = metrics.get(f'{prefix}_tp', 0) + metrics.get(f'{prefix}_fn', 0) + \
            metrics.get(f'{prefix}_fp', 0) + metrics.get(f'{prefix}_tn', 0)
    if total > 0:
        metrics[f'{prefix}_prevalence'] = (metrics.get(f'{prefix}_tp', 0) + metrics.get(f'{prefix}_fn', 0)) / total
    else:
        metrics[f'{prefix}_prevalence'] = None
    # --- End ensure ---

    return metrics
# --- Main Evaluation Function ---

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                   feature_names: List[str], model_name: str) -> Dict[str, Any] | None:
    """ Evaluates a trained model, calculates metrics, generates plots, and returns results. """
    logging.info(f"--- Evaluating Model: {model_name} ---")
    if model is None:
        logging.error("Model object is None. Cannot evaluate.")
        return None
    if X_test.empty or y_test.empty:
        logging.error("Test data (X_test or y_test) is empty. Cannot evaluate.")
        return None

    results = {'model_name': model_name}
    X_test_processed = X_test.copy() # Work on a copy

    try:
        # --- Pre-prediction Processing (e.g., XGBoost name sanitizing) ---
        if xgb and isinstance(model, xgb.XGBRegressor):
            logging.debug("Sanitizing feature names for XGBoost prediction...")
            regex = re.compile(r"[^A-Za-z0-9_]", re.IGNORECASE)
            sanitized_cols = [regex.sub("_", str(col)) for col in X_test_processed.columns]
            X_test_processed.columns = sanitized_cols
            # Ensure feature_names list matches if plotting importance later
            if len(feature_names) == len(sanitized_cols):
                 feature_names = sanitized_cols # Use sanitized names if lengths match
            else:
                 logging.warning("Original feature names list length doesn't match sanitized columns for XGB. Importance plot might be affected.")


        # --- Prediction ---
        start_pred = time.time()
        y_pred = model.predict(X_test_processed)
        # Ensure predictions are non-negative
        y_pred = np.maximum(y_pred, 0).astype(np.float32) # Clip at 0 and set type
        logging.info(f"Prediction finished in {time.time() - start_pred:.2f} seconds.")

        # Basic prediction stats
        results['test_set_size'] = len(y_test)
        results['actual_mean'] = y_test.mean()
        results['actual_std'] = y_test.std()
        results['predicted_mean'] = y_pred.mean()
        results['predicted_std'] = y_pred.std()

        # --- Calculate Metrics ---
        logging.info("Calculating evaluation metrics...")
        results.update(calculate_regression_metrics(y_test, y_pred))

        # Calculate classification proxy metrics for defined thresholds
        for threshold in config.CLASSIFICATION_THRESHOLDS:
            clf_metrics = calculate_classification_proxy_metrics(y_test, y_pred, float(threshold))
            results.update(clf_metrics)

        # --- Generate Plots ---
        logging.info("Generating evaluation plots...")
        plot_base_name = f"{model_name}_{config.PIPELINE_SUFFIX}"
        plot_predicted_vs_actual(y_test, y_pred, _create_plot_filename(plot_base_name, "pred_vs_actual"), title_suffix=f"({model_name})")
        plot_residuals_vs_predicted(y_test, y_pred, _create_plot_filename(plot_base_name, "residuals"), title_suffix=f"({model_name})")
        plot_error_distribution(y_test, y_pred, _create_plot_filename(plot_base_name, "error_dist"), title_suffix=f"({model_name})")
        plot_feature_importance(model, feature_names, _create_plot_filename(plot_base_name, "feature_importance"), title_suffix=f"({model_name})")

        # --- Plot Confusion Matrix (for primary threshold) --- <<< ADD THIS SECTION
        primary_threshold = config.CLASSIFICATION_THRESHOLDS[0] # Usually 0
        plot_confusion_matrix(
            y_test, y_pred,
            _create_plot_filename(plot_base_name, f"confusion_matrix_thresh{primary_threshold}"),
            threshold=primary_threshold,
            title_suffix=f"({model_name})"
        )

        logging.info(f"--- Evaluation Complete: {model_name} ---")
        # Log key metrics
        rmse_val = results.get('rmse', 'N/A')
        mae_val = results.get('mae', 'N/A')
        r2_val = results.get('r2', 'N/A')
        acc5_val = results.get('acc_5min', 'N/A')
        clf_acc_val = results.get('clf_thresh0_accuracy', 'N/A')

        log_rmse = f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else rmse_val
        log_mae = f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) else mae_val
        log_r2 = f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else r2_val
        log_acc5 = f"{acc5_val:.2f}%" if isinstance(acc5_val, (int, float)) else acc5_val
        log_clf_acc = f"{clf_acc_val:.4f}" if isinstance(clf_acc_val, (int, float)) else clf_acc_val

        logging.info(f"  RMSE: {log_rmse}, MAE: {log_mae}, R2: {log_r2}")
        logging.info(f"  Accuracy (±5min): {log_acc5}")
        logging.info(f"  Classification Acc (Thresh 0): {log_clf_acc}")
        return results

    except Exception as e:
        logging.error(f"Error during evaluation of {model_name}: {e}", exc_info=True)
        return None
    finally:
        # Clean up memory
        if 'X_test_processed' in locals() and X_test_processed is not None:
            del X_test_processed, y_pred
        if 'y_pred' in locals() and y_pred is not None: 
             del y_pred
        gc.collect()
