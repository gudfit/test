 # src/weather_prediction/evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             confusion_matrix, classification_report, accuracy_score) # Added classification metrics
import matplotlib.pyplot as plt
import seaborn as sns # Added seaborn
import logging
import json
from pathlib import Path
import re

# Model specific types for feature importance check
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from src import config
# from src.utils import file_handler # Not directly needed in functions here

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Plotting Functions ---
def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = "", sample_size: int = 1000):
    """Plots predicted vs actual values (with optional sampling)."""
    logging.info(f"Generating Predicted vs Actual plot {title_suffix}...")
    try:
        plt.figure(figsize=(8, 6)) # Adjusted size slightly
        indices = np.arange(len(y_test))
        if len(y_test) > sample_size:
            indices = np.random.choice(indices, sample_size, replace=False)

        y_test_sample = y_test.iloc[indices]
        y_pred_sample = y_pred[indices]

        plt.scatter(y_test_sample, y_pred_sample, alpha=0.3, label='Sampled Data')
        min_val = min(y_test_sample.min(), y_pred_sample.min())
        max_val = max(y_test_sample.max(), y_pred_sample.max())
        plot_lim = max(abs(min_val), abs(max_val)) # Make axis symmetric around ideal line if needed
        ideal_min = 0 if min_val >= 0 else min_val # Start ideal line at 0 if no negative values
        ideal_max = max_val
        plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], '--r', linewidth=2, label='Ideal Line (y=x)')
        plt.title(f'Predicted vs. Actual Weather Delay {title_suffix}')
        plt.xlabel('Actual Weather Delay (Minutes)')
        plt.ylabel('Predicted Weather Delay (Minutes)')
        # Optional: Zoom on lower values might be useful
        # plt.xlim(right=max(50, y_test_sample.quantile(0.9))) # Example zoom
        # plt.ylim(top=max(50, y_pred_sample.quantile(0.9)))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        logging.info(f"Plot saved to {file_path}")
    except Exception as e:
        logging.error(f"Error generating Predicted vs Actual plot {title_suffix}: {e}", exc_info=True)

def plot_residuals(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = "", sample_size: int = 1000):
    """Plots residuals (actual - predicted) vs predicted values."""
    logging.info(f"Generating Residuals plot {title_suffix}...")
    try:
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6)) # Adjusted size slightly

        indices = np.arange(len(y_test))
        if len(y_test) > sample_size:
            indices = np.random.choice(indices, sample_size, replace=False)

        y_pred_sample = y_pred[indices]
        residuals_sample = residuals.iloc[indices]

        plt.scatter(y_pred_sample, residuals_sample, alpha=0.3, label='Sampled Residuals')
        min_pred = y_pred_sample.min()
        max_pred = y_pred_sample.max()
        plt.hlines(0, xmin=min_pred, xmax=max_pred, colors='red', linestyles='--', label='Zero Residual Line')
        plt.title(f'Residuals vs. Predicted Weather Delay {title_suffix}')
        plt.xlabel('Predicted Weather Delay (Minutes)')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        logging.info(f"Plot saved to {file_path}")
    except Exception as e:
        logging.error(f"Error generating Residuals plot {title_suffix}: {e}", exc_info=True)

def plot_error_distribution(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    """Plots the distribution of prediction errors."""
    logging.info(f"Generating Error Distribution plot {title_suffix}...")
    try:
        errors = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(errors, bins=50, kde=True)
        plt.title(f"Prediction Error Distribution {title_suffix}")
        plt.xlabel("Error (Actual - Predicted Minutes)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        logging.info(f"Plot saved to {file_path}")
    except Exception as e:
        logging.error(f"Error generating Error Distribution plot {title_suffix}: {e}", exc_info=True)

def plot_feature_importance(model, feature_names: list, file_path: Path, title_suffix: str = ""):
    """Plots feature importances for tree-based models."""
    logging.info(f"Generating Feature Importance plot {title_suffix}...")
    try:
        if isinstance(model, lgb.LGBMRegressor):
            ax = lgb.plot_importance(model, max_num_features=20, figsize=(10, 8))
            plt.title(f"LGBM Feature Importance {title_suffix}")
        elif isinstance(model, xgb.XGBRegressor):
            fig, ax = plt.subplots(figsize=(10, 8))
            xgb.plot_importance(model, max_num_features=20, ax=ax)
            plt.title(f"XGBoost Feature Importance {title_suffix}")
        elif isinstance(model, cb.CatBoostRegressor):
            importances = model.get_feature_importance()
            # Need feature names from the model if possible, or use input list
            fnames = model.feature_names_ if hasattr(model, 'feature_names_') and model.feature_names_ else feature_names
            if len(fnames) == len(importances):
                 feat_imp_df = pd.DataFrame({'feature': fnames, 'importance': importances})
                 feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False).head(20)
                 plt.figure(figsize=(10, 8))
                 sns.barplot(x='importance', y='feature', data=feat_imp_df)
                 plt.title(f"CatBoost Feature Importance {title_suffix}")
            else:
                 logging.warning(f"CatBoost feature name length mismatch ({len(fnames)} vs {len(importances)}). Skipping plot.")
                 return # Cannot plot without matching names
        else:
            logging.warning(f"Feature importance plot not supported for model type: {type(model)}. Skipping.")
            return

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        logging.info(f"Plot saved to {file_path}")

    except Exception as e:
        logging.error(f"Error generating Feature Importance plot {title_suffix}: {e}", exc_info=True)


# --- Classification Proxy Evaluation ---
def evaluate_regression_as_classification(y_test: pd.Series, y_pred: np.ndarray, threshold: float, model_name: str) -> dict | None:
    """Evaluates regression predictions using classification metrics based on a threshold."""
    logging.info(f"--- Evaluating {model_name} as Classification (Threshold > {threshold}) ---")
    try:
        y_test_binary = (y_test > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)

        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
        class_report = classification_report(y_test_binary, y_pred_binary, output_dict=True, zero_division=0)

        tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.shape == (2, 2) else (0, 0, 0, 0) # Handle cases with only one class predicted

        logging.info(f"Classification Accuracy: {accuracy:.4f}")
        logging.info(f"Confusion Matrix:\n[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")
        # logging.info(f"Classification Report:\n{classification_report(y_test_binary, y_pred_binary, zero_division=0)}") # Less verbose

        results = {
            f'clf_accuracy_thresh{threshold}': accuracy,
            f'clf_tn_thresh{threshold}': tn,
            f'clf_fp_thresh{threshold}': fp,
            f'clf_fn_thresh{threshold}': fn,
            f'clf_tp_thresh{threshold}': tp,
            f'clf_precision_0_thresh{threshold}': class_report['0']['precision'],
            f'clf_recall_0_thresh{threshold}': class_report['0']['recall'],
            f'clf_f1_0_thresh{threshold}': class_report['0']['f1-score'],
            f'clf_precision_1_thresh{threshold}': class_report['1']['precision'],
            f'clf_recall_1_thresh{threshold}': class_report['1']['recall'],
            f'clf_f1_1_thresh{threshold}': class_report['1']['f1-score'],
        }
        return results

    except Exception as e:
        logging.error(f"Error during classification proxy evaluation for {model_name}: {e}", exc_info=True)
        return None


# --- Main Evaluation Function ---
def evaluate_single_run(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, feature_names: list) -> dict | None:
    """Evaluates a single trained model run, returns metrics, and generates plots."""
    logging.info(f"--- Evaluating {model_name} ---")
    X_test_processed = X_test.copy()
    results = {}
    try:
        # --- Sanitize for XGBoost ---
        if 'XGB' in model_name:
             try:
                 if hasattr(X_test_processed, 'columns'):
                    regex = re.compile(r"[^A-Za-z0-9_]", re.IGNORECASE)
                    X_test_processed.columns = [regex.sub("_", str(x)) for x in X_test_processed.columns]
                    logging.debug("Sanitized column names for XGBoost evaluation.")
                    # Use sanitized names for importance plotting if applicable
                    feature_names = list(X_test_processed.columns)
             except Exception as rename_error:
                 logging.warning(f"Could not rename columns for XGBoost evaluation: {rename_error}")

        # --- Predict ---
        y_pred = model.predict(X_test_processed)
        y_pred[y_pred < 0] = 0

        # --- Regression Metrics ---
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        acc_5min = np.mean(np.abs(y_test - y_pred) <= 5) * 100

        logging.info(f"📉 Regression Results for {model_name}:")
        logging.info(f"✅ RMSE: {rmse:.2f}")
        logging.info(f"✅ MAE: {mae:.2f}")
        logging.info(f"✅ R² Score: {r2:.4f}")
        logging.info(f"✅ Accuracy within ±5 minutes: {acc_5min:.2f}%")

        results.update({
            'model_name': model_name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'acc_5min': acc_5min,
            'test_set_size': len(y_test),
            'actual_mean': y_test.mean(), 'actual_std': y_test.std(),
            'predicted_mean': y_pred.mean(), 'predicted_std': y_pred.std()
        })

        # --- Non-Zero Metrics ---
        non_zero_indices = y_test > 0
        mae_nz, rmse_nz, r2_nz = np.nan, np.nan, np.nan
        count_nz = 0
        if non_zero_indices.any():
            y_test_nz = y_test[non_zero_indices]
            y_pred_nz = y_pred[non_zero_indices]
            count_nz = len(y_test_nz)
            if count_nz > 0:
                 mae_nz = mean_absolute_error(y_test_nz, y_pred_nz)
                 rmse_nz = np.sqrt(mean_squared_error(y_test_nz, y_pred_nz))
                 r2_nz = r2_score(y_test_nz, y_pred_nz)
                 logging.info("--- Metrics for Actual > 0 ---")
                 logging.info(f"MAE (Non-Zero): {mae_nz:.2f}")
                 logging.info(f"RMSE (Non-Zero): {rmse_nz:.2f}")
                 logging.info(f"R2 (Non-Zero): {r2_nz:.4f}")
            else:
                 logging.warning("Non-zero subset was empty during evaluation.")
        else:
            logging.warning("No non-zero actual delays found in test set.")

        results.update({'rmse_non_zero': rmse_nz, 'mae_non_zero': mae_nz, 'r2_non_zero': r2_nz, 'count_non_zero': count_nz})

        # --- Classification Proxy Metrics ---
        clf_results = evaluate_regression_as_classification(
            y_test, y_pred, config.CLASSIFICATION_THRESHOLD, model_name
        )
        if clf_results:
            results.update(clf_results)

        # --- Plotting ---
        # Construct unique filenames for plots for this run
        plot_base_name = f"{model_name}_evaluation"
        pred_actual_path = config.PLOTS_DIR / f"{plot_base_name}_pred_vs_actual.png"
        residuals_path = config.PLOTS_DIR / f"{plot_base_name}_residuals.png"
        error_dist_path = config.PLOTS_DIR / f"{plot_base_name}_error_dist.png"
        feat_imp_path = config.PLOTS_DIR / f"{plot_base_name}_feat_imp.png"

        plot_predicted_vs_actual(y_test, y_pred, pred_actual_path, title_suffix=f"({model_name})")
        plot_residuals(y_test, y_pred, residuals_path, title_suffix=f"({model_name})")
        plot_error_distribution(y_test, y_pred, error_dist_path, title_suffix=f"({model_name})")
        plot_feature_importance(model, feature_names, feat_imp_path, title_suffix=f"({model_name})")

        return results

    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {e}", exc_info=True)
        return None
