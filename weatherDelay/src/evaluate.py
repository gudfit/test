# FILE: weatherDelay/src/evaluate.py
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, median_absolute_error,
                             mean_absolute_percentage_error,
                             confusion_matrix, classification_report, accuracy_score,
                             precision_recall_fscore_support,
                             precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path
import re
import gc
from typing import List, Dict, Any, Tuple
import time

try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy import stats
    _PLOT_EXTRAS = True
except ImportError:
    lgb = xgb = cb = None
    lowess = stats = None
    _PLOT_EXTRAS = False
    logging.warning("Optional plotting dependencies not found. Some plots disabled.")

from src import config, utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --------------------------------------------------------------------------------
# Helpers for Plot Filenames & Safe-Save Decorator
# --------------------------------------------------------------------------------
def _create_plot_filename(base_name: str, plot_type: str) -> Path:
    plot_dir = config.PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / f"{base_name}_{plot_type}.png"

def _safe_plot_save(func):
    def wrapper(*args, **kwargs):
        plot_path = kwargs.get('file_path', args[2] if len(args) > 2 and isinstance(args[2], Path) else None)
        plot_desc = kwargs.get('title_suffix', args[3] if len(args) > 3 else "")
        try:
            func(*args, **kwargs)
            if plot_path:
                logging.info(f"Plot saved: {plot_path.name}")
        except Exception as e:
            logging.error(f"Failed to generate/save plot '{plot_desc}' ({plot_path}): {e}", exc_info=False)
        finally:
            plt.close('all')
            gc.collect()
    return wrapper


# --------------------------------------------------------------------------------
# Plot #1: Predicted vs Actual
# --------------------------------------------------------------------------------
@_safe_plot_save
def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    logging.debug(f"Generating Predicted vs Actual plot {title_suffix}...")
    if len(y_test) == 0:
        return
    sample_size = min(len(y_test), config.PLOT_SAMPLE_SIZE)
    indices = np.random.choice(np.arange(len(y_test)), sample_size, replace=False)
    y_test_sample = y_test.iloc[indices]
    y_pred_sample = y_pred[indices]

    plt.figure(figsize=(10, 8))
    plt.hexbin(y_test_sample, y_pred_sample, gridsize=50, cmap=config.PLOT_CMAP, mincnt=1, alpha=0.8)
    plt.colorbar(label='Count in Bin')

    min_val = min(0, y_test_sample.min(), y_pred_sample.min())
    max_val = max(y_test_sample.max(), y_pred_sample.max()) * 1.05
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

# --------------------------------------------------------------------------------
# Plot #2: Residuals vs. Predicted
# --------------------------------------------------------------------------------
@_safe_plot_save
def plot_residuals_vs_predicted(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    logging.debug(f"Generating Residuals vs Predicted plot {title_suffix}...")
    if len(y_test) == 0:
        return
    residuals = y_test - y_pred
    sample_size = min(len(y_test), config.PLOT_SAMPLE_SIZE)
    indices = np.random.choice(np.arange(len(y_test)), sample_size, replace=False)
    y_pred_sample = y_pred[indices]
    residuals_sample = residuals.iloc[indices]

    plt.figure(figsize=(10, 8))
    plt.hexbin(y_pred_sample, residuals_sample, gridsize=50, cmap='coolwarm', mincnt=1, alpha=0.7)
    plt.colorbar(label='Count in Bin')
    plt.axhline(0, color='black', linestyle='--', linewidth=2, label='Zero Residual')

    if _PLOT_EXTRAS and lowess:
        try:
            smoothed = lowess(residuals_sample, y_pred_sample, frac=0.3)
            plt.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=3, label='LOWESS Trend')
        except Exception as e:
            logging.warning(f"Could not compute LOWESS for residuals plot: {e}")

    plt.title(f'Residuals vs. Predicted {title_suffix}', fontsize=14)
    plt.xlabel('Predicted Weather Delay (Minutes)', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=config.PLOT_DPI)

# --------------------------------------------------------------------------------
# Plot #3: Error Distribution
# --------------------------------------------------------------------------------
@_safe_plot_save
def plot_error_distribution(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    logging.debug(f"Generating Error Distribution plot {title_suffix}...")
    if len(y_test) == 0:
        return
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True, stat="density")

    if _PLOT_EXTRAS and stats:
        try:
            mu, sigma = stats.norm.fit(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=2,
                     label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
            kurt = stats.kurtosis(errors)
            skew = stats.skew(errors)
            stats_text = (f"Mean: {errors.mean():.2f}\nMedian: {errors.median():.2f}\n"
                          f"Std Dev: {errors.std():.2f}\nSkew: {skew:.2f}\nKurtosis: {kurt:.2f}")
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        except Exception as e:
            logging.warning(f"Could not compute stats/fit for error distribution: {e}")

    plt.axvline(errors.mean(), color='r', linestyle='-', linewidth=1.5, label=f'Mean Error ({errors.mean():.2f})')
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.title(f"Prediction Error Distribution {title_suffix}", fontsize=14)
    plt.xlabel("Error (Actual - Predicted Minutes)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=config.PLOT_DPI)

# --------------------------------------------------------------------------------
# Plot #4: Feature Importance
# --------------------------------------------------------------------------------
@_safe_plot_save
def plot_feature_importance(model, feature_names: List[str], file_path: Path, title_suffix: str = ""):
    logging.debug(f"Generating Feature Importance plot {title_suffix}...")
    importance = None
    fnames_used = feature_names

    try:
        if lgb and isinstance(model, lgb.LGBMRegressor):
            importance = model.feature_importances_
            fnames_used = model.feature_name_
            plot_title = f"LGBM Feature Importance {title_suffix}"
        elif xgb and isinstance(model, xgb.XGBRegressor):
            importance = model.feature_importances_
            plot_title = f"XGBoost Feature Importance {title_suffix}"
        elif cb and isinstance(model, cb.CatBoostRegressor):
            importance = model.get_feature_importance()
            fnames_used = model.feature_names_ if hasattr(model, 'feature_names_') else feature_names
            plot_title = f"CatBoost Feature Importance {title_suffix}"
        else:
            logging.warning(f"Feature importance plot not supported for model type: {type(model)}. Skipping.")
            return

        if importance is None:
            logging.warning("Could not retrieve feature importances.")
            return
        if len(fnames_used) != len(importance):
            logging.error(f"Feature name length mismatch! Got {len(fnames_used)} names, {len(importance)} importances.")
            return

        feat_imp_df = pd.DataFrame({'feature': fnames_used, 'importance': importance})
        feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False).head(25)

        plt.figure(figsize=(10, 10))
        sns.barplot(x='importance', y='feature', data=feat_imp_df, hue='feature', palette='viridis', legend=False)
        plt.title(plot_title, fontsize=14)
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(file_path, dpi=config.PLOT_DPI)

    except Exception as e:
        logging.error(f"Error generating feature importance plot: {e}", exc_info=False)

# --------------------------------------------------------------------------------
# Plot #5: Confusion Matrix for a Single Threshold
# --------------------------------------------------------------------------------
@_safe_plot_save
def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, threshold: float = 0, title_suffix: str = ""):
    logging.debug(f"Generating Confusion Matrix plot (Threshold > {threshold}) {title_suffix}...")
    if len(y_test) == 0:
        return

    y_test_binary = (y_test > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    try:
        cm = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except ValueError:
        logging.warning(f"Could not unpack full confusion matrix for threshold {threshold}. Adjusting plot.")
        cm = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
        tn = fp = fn = tp = 0
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()

    accuracy = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred No Delay', 'Pred Delay'],
                yticklabels=['True No Delay', 'True Delay'],
                annot_kws={"size": 12})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    total = tn + fp + fn + tp
    if total > 0:
        for i, val_row in enumerate(cm):
            for j, val in enumerate(val_row):
                plt.text(j + 0.5, i + 0.7, f'({val / total:.1%})',
                         ha='center', va='center', fontsize=10,
                         color='black' if val < total/2 else 'white')

    plt.title(f'Confusion Matrix (Threshold > {threshold}) {title_suffix}', fontsize=14)
    metrics_text = (f"Accuracy: {accuracy:.3f}\n"
                    f"Precision: {precision:.3f}\n"
                    f"Recall: {recall:.3f}\n"
                    f"F1 Score: {f1:.3f}\n"
                    f"Specificity: {specificity:.3f}")
    plt.text(1.1, 0.5, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    plt.subplots_adjust(right=0.75)
    plt.savefig(file_path, dpi=config.PLOT_DPI, bbox_inches='tight')

# --------------------------------------------------------------------------------
# Plot #6: Precision–Recall Curve
# --------------------------------------------------------------------------------
@_safe_plot_save
def plot_precision_recall_curve(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    logging.debug(f"Generating Precision–Recall curve {title_suffix}...")
    if len(y_test) == 0:
        return

    y_true_binary = (y_test > 15).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.title(f"Precision–Recall Curve for Delay>15 {title_suffix}", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path, dpi=config.PLOT_DPI)

# --------------------------------------------------------------------------------
# NEW FUNCTION: find_best_threshold_for_f1
# --------------------------------------------------------------------------------
def find_best_threshold_for_f1(y_test: pd.Series, y_pred: np.ndarray, actual_delay_cutoff=15.0) -> float:
    """
    Sweeps possible thresholds on y_pred (the predicted minutes) to classify flights
    as "delayed" vs "not delayed" for an actual delay cutoff (e.g. 15 min).
    Returns the threshold that yields the highest F1 for the "delayed" class.
    """
    logging.info(f"Finding best threshold for F1 (delay > {actual_delay_cutoff} min) ...")

    # 1) Create binary ground truth for actual > 15
    y_true_binary = (y_test > actual_delay_cutoff).astype(int)

    # 2) We'll sort unique predictions as candidate thresholds
    #    (We could also use the precision_recall_curve thresholds, same idea.)
    unique_preds = np.unique(y_pred)
    # For big data, we might limit to e.g. 1000 sorted thresholds, but let's keep it simple here

    best_threshold = 0.0
    best_f1 = 0.0

    for thresh in unique_preds:
        y_pred_binary = (y_pred >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, labels=[1], average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    logging.info(f"Best F1={best_f1:.3f} at threshold={best_threshold:.3f} for Delay>={actual_delay_cutoff}min")
    return best_threshold

# --------------------------------------------------------------------------------
# Regression & Classification Metrics
# --------------------------------------------------------------------------------
def calculate_regression_metrics(y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_test) == 0:
        return {}
    metrics = {}
    metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['mae'] = mean_absolute_error(y_test, y_pred)
    metrics['median_ae'] = median_absolute_error(y_test, y_pred)
    metrics['r2'] = r2_score(y_test, y_pred)

    for mins in [5, 15, 30]:
        metrics[f'acc_{mins}min'] = np.mean(np.abs(y_test - y_pred) <= mins) * 100

    mask = y_test != 0
    if np.any(mask):
        mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100
        metrics['mape'] = mape if np.isfinite(mape) else None
    else:
        metrics['mape'] = None

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
    if len(y_test) == 0:
        return {}
    metrics = {}
    prefix = f"clf_thresh{threshold}"

    y_test_binary = (y_test > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    metrics[f'{prefix}_accuracy'] = accuracy_score(y_test_binary, y_pred_binary)
    try:
        tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1]).ravel()
        metrics[f'{prefix}_tn'] = int(tn)
        metrics[f'{prefix}_fp'] = int(fp)
        metrics[f'{prefix}_fn'] = int(fn)
        metrics[f'{prefix}_tp'] = int(tp)
    except ValueError:
        cm = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
        metrics[f'{prefix}_tn'] = 0
        metrics[f'{prefix}_fp'] = 0
        metrics[f'{prefix}_fn'] = 0
        metrics[f'{prefix}_tp'] = 0
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            metrics[f'{prefix}_tn'] = int(tn)
            metrics[f'{prefix}_fp'] = int(fp)
            metrics[f'{prefix}_fn'] = int(fn)
            metrics[f'{prefix}_tp'] = int(tp)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test_binary, y_pred_binary, labels=[0, 1], zero_division=0)
    metrics[f'{prefix}_precision_0'] = precision[0]
    metrics[f'{prefix}_recall_0']   = recall[0]
    metrics[f'{prefix}_f1_0']       = f1[0]
    metrics[f'{prefix}_precision_1'] = precision[1]
    metrics[f'{prefix}_recall_1']    = recall[1]
    metrics[f'{prefix}_f1_1']        = f1[1]
    metrics[f'{prefix}_specificity'] = recall[0]

    total = (metrics.get(f'{prefix}_tp', 0) + metrics.get(f'{prefix}_fn', 0) +
             metrics.get(f'{prefix}_fp', 0) + metrics.get(f'{prefix}_tn', 0))
    if total > 0:
        metrics[f'{prefix}_prevalence'] = (metrics.get(f'{prefix}_tp', 0) + metrics.get(f'{prefix}_fn', 0)) / total
    else:
        metrics[f'{prefix}_prevalence'] = None

    return metrics

# --------------------------------------------------------------------------------
# Main Evaluation Function
# --------------------------------------------------------------------------------
def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                   feature_names: List[str], model_name: str) -> Dict[str, Any] | None:
    logging.info(f"--- Evaluating Model: {model_name} ---")
    if model is None:
        logging.error("Model object is None. Cannot evaluate.")
        return None
    if X_test.empty or y_test.empty:
        logging.error("Test data (X_test or y_test) is empty. Cannot evaluate.")
        return None

    results = {'model_name': model_name}
    X_test_processed = X_test.copy()

    try:
        # If XGB, sanitize columns
        if xgb and isinstance(model, xgb.XGBRegressor):
            logging.debug("Sanitizing feature names for XGBoost prediction...")
            regex = re.compile(r"[^A-Za-z0-9_]", re.IGNORECASE)
            sanitized_cols = [regex.sub("_", str(col)) for col in X_test_processed.columns]
            X_test_processed.columns = sanitized_cols
            if len(feature_names) == len(sanitized_cols):
                feature_names = sanitized_cols
            else:
                logging.warning("Feature name mismatch for XGB. Feature importance plot might be off.")

        # --- Prediction ---
        start_pred = time.time()
        y_pred = model.predict(X_test_processed)
        y_pred = np.maximum(y_pred, 0).astype(np.float32)
        logging.info(f"Prediction finished in {time.time() - start_pred:.2f} seconds.")

        results['test_set_size']   = len(y_test)
        results['actual_mean']     = y_test.mean()
        results['actual_std']      = y_test.std()
        results['predicted_mean']  = y_pred.mean()
        results['predicted_std']   = y_pred.std()

        # --- Calculate Regression Metrics ---
        logging.info("Calculating evaluation metrics (regression + classification proxy)...")
        results.update(calculate_regression_metrics(y_test, y_pred))

        # Classification Proxy for thresholds in config
        for threshold in config.CLASSIFICATION_THRESHOLDS:
            clf_metrics = calculate_classification_proxy_metrics(y_test, y_pred, float(threshold))
            results.update(clf_metrics)

        # --- Plots ---
        logging.info("Generating evaluation plots...")
        plot_base_name = f"{model_name}_{config.PIPELINE_SUFFIX}"
        plot_predicted_vs_actual(y_test, y_pred, _create_plot_filename(plot_base_name, "pred_vs_actual"), title_suffix=f"({model_name})")
        plot_residuals_vs_predicted(y_test, y_pred, _create_plot_filename(plot_base_name, "residuals"), title_suffix=f"({model_name})")
        plot_error_distribution(y_test, y_pred, _create_plot_filename(plot_base_name, "error_dist"), title_suffix=f"({model_name})")
        plot_feature_importance(model, feature_names, _create_plot_filename(plot_base_name, "feature_importance"), title_suffix=f"({model_name})")

        # Confusion matrix for the first threshold in config (often 15)
        if config.CLASSIFICATION_THRESHOLDS:
            primary_threshold = config.CLASSIFICATION_THRESHOLDS[0]
            plot_confusion_matrix(
                y_test, y_pred,
                _create_plot_filename(plot_base_name, f"confusion_matrix_thresh{primary_threshold}"),
                threshold=primary_threshold,
                title_suffix=f"({model_name})"
            )

        # PR curve for "delay > 15"
        plot_precision_recall_curve(
            y_test, y_pred,
            _create_plot_filename(plot_base_name, "precision_recall_curve"),
            title_suffix=f"({model_name})"
        )

        # --------------------------------------------------------------------------------
        # NEW CODE BLOCK: find the best threshold for F1 & plot a confusion matrix for it
        # --------------------------------------------------------------------------------
        best_f1_threshold = find_best_threshold_for_f1(y_test, y_pred, actual_delay_cutoff=15.0)
        # Let's do a confusion matrix at that threshold too:
        plot_confusion_matrix(
            y_test, y_pred,
            _create_plot_filename(plot_base_name, f"confusion_matrix_bestF1"),
            threshold=best_f1_threshold,
            title_suffix=f"({model_name}_bestF1th={best_f1_threshold:.1f})"
        )

        logging.info(f"--- Evaluation Complete: {model_name} ---")

        # Log a summary
        rmse_val = results.get('rmse', 'N/A')
        mae_val  = results.get('mae', 'N/A')
        r2_val   = results.get('r2', 'N/A')
        acc5_val = results.get('acc_5min', 'N/A')
        clf_acc15_val = results.get('clf_thresh15.0_accuracy', 'N/A')

        log_rmse = f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else rmse_val
        log_mae  = f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) else mae_val
        log_r2   = f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else r2_val
        log_acc5 = f"{acc5_val:.2f}%" if isinstance(acc5_val, (int, float)) else acc5_val
        log_clf_acc15 = f"{clf_acc15_val:.4f}" if isinstance(clf_acc15_val, (int, float)) else clf_acc15_val

        logging.info(f"  RMSE: {log_rmse}, MAE: {log_mae}, R2: {log_r2}")
        logging.info(f"  Accuracy (±5min): {log_acc5}")
        logging.info(f"  Classification Acc (Thresh 15): {log_clf_acc15}")
        logging.info(f"  Best-F1 threshold for Delay>15 found at: {best_f1_threshold:.1f}")

        return results

    except Exception as e:
        logging.error(f"Error during evaluation of {model_name}: {e}", exc_info=True)
        return None
    finally:
        del X_test_processed
        gc.collect()
