# src/2_weatherPrediction/evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             confusion_matrix, classification_report, accuracy_score,
                             precision_recall_curve, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path
import re

# Model specific types for feature importance check
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Enhanced Plotting Functions ---
def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = "", sample_size: int = 1000):
    """Plots predicted vs actual values (with optional sampling)."""
    logging.info(f"Generating Predicted vs Actual plot {title_suffix}...")
    try:
        plt.figure(figsize=(10, 8))
        indices = np.arange(len(y_test))
        if len(y_test) > sample_size:
            indices = np.random.choice(indices, sample_size, replace=False)

        y_test_sample = y_test.iloc[indices]
        y_pred_sample = y_pred[indices]

        # Create scatter plot with hexbin for better visualization with many points
        plt.hexbin(y_test_sample, y_pred_sample, 
                  gridsize=30, cmap='viridis', 
                  mincnt=1, alpha=0.8)
        
        plt.colorbar(label='Count')
        
        # Add reference line
        min_val = min(y_test_sample.min(), y_pred_sample.min())
        max_val = max(y_test_sample.max(), y_pred_sample.max())
        ideal_min = 0 if min_val >= 0 else min_val
        ideal_max = max_val
        plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], '--r', linewidth=2, label='Ideal Line (y=x)')
        
        # Add some sample points as scatter for better understanding
        small_sample = np.random.choice(indices, min(200, len(indices)), replace=False)
        plt.scatter(y_test.iloc[small_sample], y_pred[small_sample], 
                   alpha=0.5, edgecolor='k', s=40, label='Sample Points')
        
        plt.title(f'Predicted vs Actual Weather Delay {title_suffix}', fontsize=14)
        plt.xlabel('Actual Weather Delay (Minutes)', fontsize=12)
        plt.ylabel('Predicted Weather Delay (Minutes)', fontsize=12)
        
        # Zoom in to where most data points are
        q95 = np.percentile(y_test_sample[y_test_sample > 0], 95) if len(y_test_sample[y_test_sample > 0]) > 0 else max_val
        plt.xlim(0, min(q95*1.2, max_val))
        plt.ylim(0, min(q95*1.2, max_val))
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        # Create a second zoomed-out plot showing all data
        plt.figure(figsize=(10, 8))
        plt.hexbin(y_test_sample, y_pred_sample, 
                  gridsize=30, cmap='viridis', 
                  mincnt=1, alpha=0.8)
        plt.colorbar(label='Count')
        plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], '--r', linewidth=2)
        plt.title(f'Full Range: Predicted vs Actual Weather Delay {title_suffix}', fontsize=14)
        plt.xlabel('Actual Weather Delay (Minutes)', fontsize=12)
        plt.ylabel('Predicted Weather Delay (Minutes)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        zoomed_path = str(file_path).replace('.png', '_full_range.png')
        plt.savefig(zoomed_path, dpi=300)
        plt.close()
        
        logging.info(f"Prediction scatter plots saved to {file_path} and {zoomed_path}")
    except Exception as e:
        logging.error(f"Error generating Predicted vs Actual plot {title_suffix}: {e}", exc_info=True)

def plot_residuals(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = "", sample_size: int = 1000):
    """Plots residuals (actual - predicted) vs predicted values."""
    logging.info(f"Generating Residuals plot {title_suffix}...")
    try:
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 8))

        indices = np.arange(len(y_test))
        if len(y_test) > sample_size:
            indices = np.random.choice(indices, sample_size, replace=False)

        y_pred_sample = y_pred[indices]
        residuals_sample = residuals.iloc[indices]

        # Use hexbin for better visualization with many points
        plt.hexbin(y_pred_sample, residuals_sample, 
                  gridsize=30, cmap='coolwarm', 
                  mincnt=1, alpha=0.8)
        
        plt.colorbar(label='Count')
        
        # Add zero line
        min_pred = y_pred_sample.min()
        max_pred = y_pred_sample.max()
        plt.hlines(0, xmin=min_pred, xmax=max_pred, colors='black', linestyles='--', linewidth=2, label='Zero Residual Line')
        
        # Add LOESS smoother to show trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            z = lowess(residuals_sample, y_pred_sample, frac=0.3, it=1)
            plt.plot(z[:, 0], z[:, 1], 'r-', linewidth=3, label='LOWESS Trend')
        except:
            logging.warning("Could not add LOWESS trend line - statsmodels may not be installed")
            
        plt.title(f'Residuals vs Predicted Weather Delay {title_suffix}', fontsize=14)
        plt.xlabel('Predicted Weather Delay (Minutes)', fontsize=12)
        plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
        
        # Add standard residual boundaries (±1.96)
        std_resid = residuals_sample.std()
        plt.hlines([-1.96*std_resid, 1.96*std_resid], xmin=min_pred, xmax=max_pred, 
                  colors=['green', 'green'], linestyles=':', linewidth=1.5, 
                  label='±1.96σ (95% Confidence)')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        logging.info(f"Residuals plot saved to {file_path}")
    except Exception as e:
        logging.error(f"Error generating Residuals plot {title_suffix}: {e}", exc_info=True)

def plot_error_distribution(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    """Plots the distribution of prediction errors."""
    logging.info(f"Generating Error Distribution plot {title_suffix}...")
    try:
        errors = y_test - y_pred
        plt.figure(figsize=(12, 8))
        
        # Create a more detailed distribution plot
        ax = plt.subplot(1, 1, 1)
        sns.histplot(errors, bins=50, kde=True, stat="density", ax=ax)
        
        # Add a normal distribution curve for comparison
        from scipy import stats
        mu, sigma = errors.mean(), errors.std()
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal Dist. (μ={mu:.2f}, σ={sigma:.2f})')
        
        # Add vertical lines for mean and zero
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
        plt.axvline(x=mu, color='red', linestyle='--', linewidth=1.5, label=f'Mean Error ({mu:.2f})')
        
        # Mark +/- 1 and 2 standard deviations
        plt.axvline(x=mu+sigma, color='green', linestyle=':', linewidth=1, label=f'+1σ ({mu+sigma:.2f})')
        plt.axvline(x=mu-sigma, color='green', linestyle=':', linewidth=1, label=f'-1σ ({mu-sigma:.2f})')
        plt.axvline(x=mu+2*sigma, color='purple', linestyle=':', linewidth=1, label=f'+2σ ({mu+2*sigma:.2f})')
        plt.axvline(x=mu-2*sigma, color='purple', linestyle=':', linewidth=1, label=f'-2σ ({mu-2*sigma:.2f})')
        
        plt.title(f"Prediction Error Distribution {title_suffix}", fontsize=14)
        plt.xlabel("Error (Actual - Predicted Minutes)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistical metrics as text box
        kurt = stats.kurtosis(errors)
        skew = stats.skew(errors)
        
        stats_text = (f"Mean: {mu:.2f}\n"
                     f"Std Dev: {sigma:.2f}\n"
                     f"Median: {np.median(errors):.2f}\n"
                     f"Min: {np.min(errors):.2f}\n"
                     f"Max: {np.max(errors):.2f}\n"
                     f"Skewness: {skew:.2f}\n"
                     f"Kurtosis: {kurt:.2f}")
        
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                    va="top", ha="left", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        # Create a second plot with zoomed view
        plt.figure(figsize=(12, 8))
        # Percentile-based zoom
        lower_bound = np.percentile(errors, 5)
        upper_bound = np.percentile(errors, 95)
        
        sns.histplot(errors, bins=50, kde=True, stat="density")
        plt.xlim(lower_bound, upper_bound)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        plt.axvline(x=mu, color='red', linestyle='--', linewidth=1.5)
        
        plt.title(f"Prediction Error Distribution (90% Range) {title_suffix}", fontsize=14)
        plt.xlabel("Error (Actual - Predicted Minutes)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        zoomed_path = str(file_path).replace('.png', '_zoomed.png')
        plt.tight_layout()
        plt.savefig(zoomed_path, dpi=300)
        plt.close()
        
        logging.info(f"Error distribution plots saved to {file_path} and {zoomed_path}")
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
                 feature_imp_df = pd.DataFrame({'feature': fnames, 'importance': importances})
                 feature_imp_df = feature_imp_df.sort_values(by='importance', ascending=False).head(20)
                 plt.figure(figsize=(10, 8))
                 # Fixed: Use hue instead of palette
                 sns.barplot(x='importance', y='feature', hue='feature', data=feature_imp_df, legend=False)
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

        # Create a second version with normalized importance
        if isinstance(model, lgb.LGBMRegressor):
            # For LGBM, get importances directly from model
            importances = model.feature_importances_
            if len(feature_names) == len(importances):
                feature_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_imp_df = feature_imp_df.sort_values(by='importance', ascending=False).head(20)
                feature_imp_df['normalized_importance'] = feature_imp_df['importance'] / feature_imp_df['importance'].sum()
                
                plt.figure(figsize=(12, 10))
                # Fixed: Use hue instead of palette
                sns.barplot(x='normalized_importance', y='feature', hue='feature', data=feature_imp_df, legend=False)
                
                # Add percentage values
                for i, v in enumerate(feature_imp_df['normalized_importance']):
                    plt.text(v + max(feature_imp_df['normalized_importance'])*0.01, i, f"{v*100:.2f}%", va='center')
                    
                plt.title(f'LGBM Normalized Feature Importance {title_suffix}', fontsize=14)
                plt.xlabel('Normalized Importance (%)', fontsize=12)
                plt.ylabel('Feature', fontsize=12)
                plt.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                normalized_path = str(file_path).replace('.png', '_normalized.png')
                plt.savefig(normalized_path, dpi=300)
                plt.close()
                logging.info(f"Normalized feature importance plot saved to {normalized_path}")
                
        elif isinstance(model, xgb.XGBRegressor):
            # For XGBoost, get importances directly from model
            importances = model.feature_importances_
            if len(feature_names) == len(importances):
                feature_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_imp_df = feature_imp_df.sort_values(by='importance', ascending=False).head(20)
                feature_imp_df['normalized_importance'] = feature_imp_df['importance'] / feature_imp_df['importance'].sum()
                
                plt.figure(figsize=(12, 10))
                # Fixed: Use hue instead of palette
                sns.barplot(x='normalized_importance', y='feature', hue='feature', data=feature_imp_df, legend=False)
                
                # Add percentage values
                for i, v in enumerate(feature_imp_df['normalized_importance']):
                    plt.text(v + max(feature_imp_df['normalized_importance'])*0.01, i, f"{v*100:.2f}%", va='center')
                    
                plt.title(f'XGBoost Normalized Feature Importance {title_suffix}', fontsize=14)
                plt.xlabel('Normalized Importance (%)', fontsize=12)
                plt.ylabel('Feature', fontsize=12)
                plt.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                normalized_path = str(file_path).replace('.png', '_normalized.png')
                plt.savefig(normalized_path, dpi=300)
                plt.close()
                logging.info(f"Normalized feature importance plot saved to {normalized_path}")
                
        elif isinstance(model, cb.CatBoostRegressor):
            # Already calculated for CatBoost above
            if 'feature_imp_df' in locals():
                feature_imp_df['normalized_importance'] = feature_imp_df['importance'] / feature_imp_df['importance'].sum()
                plt.figure(figsize=(12, 10))
                # Fixed: Use hue instead of palette
                sns.barplot(x='normalized_importance', y='feature', hue='feature', data=feature_imp_df, legend=False)
                
                # Add percentage values
                for i, v in enumerate(feature_imp_df['normalized_importance']):
                    plt.text(v + max(feature_imp_df['normalized_importance'])*0.01, i, f"{v*100:.2f}%", va='center')
                    
                plt.title(f'CatBoost Normalized Feature Importance {title_suffix}', fontsize=14)
                plt.xlabel('Normalized Importance (%)', fontsize=12)
                plt.ylabel('Feature', fontsize=12)
                plt.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                normalized_path = str(file_path).replace('.png', '_normalized.png')
                plt.savefig(normalized_path, dpi=300)
                plt.close()
                logging.info(f"Normalized feature importance plot saved to {normalized_path}")

    except Exception as e:
        logging.error(f"Error generating Feature Importance plot {title_suffix}: {e}", exc_info=True)


# --- Confusion Matrix Visualization ---
def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, threshold: float = 0, title_suffix: str = ""):
    """Creates a visual confusion matrix heatmap."""
    logging.info(f"Generating Confusion Matrix plot (threshold > {threshold}) {title_suffix}...")
    try:
        # Convert to binary classification based on threshold
        y_test_binary = (y_test > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test_binary, y_pred_binary)
        
        # Extract values
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        # Calculate metrics
        accuracy = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['No Delay', 'Delay'],
                   yticklabels=['No Delay', 'Delay'])
        
        # Add annotations for clarity
        plt.text(0.5, -0.1, 'Predicted Label', ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
        plt.text(-0.1, 0.5, 'True Label', ha='center', va='center', fontsize=12, rotation=90, transform=plt.gca().transAxes)
        
        # Annotate each cell with percentage
        total = np.sum(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm[i, j] / total:.1%})', 
                       ha='center', va='center', fontsize=11, color='black' if cm[i, j] < total/2 else 'white')
                
        plt.title(f'Confusion Matrix (Threshold > {threshold}) {title_suffix}', fontsize=14)
        
        # Add metrics as a text box
        metrics_text = (f"Accuracy: {accuracy:.4f}\n"
                       f"Precision: {precision:.4f}\n"
                       f"Recall: {recall:.4f}\n"
                       f"F1 Score: {f1:.4f}\n"
                       f"Specificity: {specificity:.4f}")
        
        plt.annotate(metrics_text, xy=(1.05, 0.5), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                    va="center", ha="left", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        logging.info(f"Confusion matrix plot saved to {file_path}")
    except Exception as e:
        logging.error(f"Error generating Confusion Matrix plot {title_suffix}: {e}", exc_info=True)

# --- ROC Curve Visualization ---
def plot_roc_curve(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    """Creates a ROC curve visualization."""
    logging.info(f"Generating ROC curve plot {title_suffix}...")
    try:
        # Convert to binary classification for ROC calculation
        y_test_binary = (y_test > 0).astype(int)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        
        # Add thresholds markers
        # We'll plot a few key threshold points for reference
        threshold_markers = [0.1, 1, 5, 10, 20, 50, 100]
        for t in threshold_markers:
            try:
                # Find the closest threshold index
                idx = (np.abs(thresholds - t)).argmin()
                if idx < len(fpr):  # Make sure we don't exceed array bounds
                    plt.plot(fpr[idx], tpr[idx], 'o', markersize=8, 
                           label=f'Threshold ≈ {thresholds[idx]:.1f}')
            except:
                pass  # Skip if threshold not found
                
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title(f'Receiver Operating Characteristic (ROC) Curve {title_suffix}', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        logging.info(f"ROC curve plot saved to {file_path}")
    except Exception as e:
        logging.error(f"Error generating ROC curve plot {title_suffix}: {e}", exc_info=True)

# --- Precision-Recall Curve ---
def plot_precision_recall_curve(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    """Creates a precision-recall curve visualization."""
    logging.info(f"Generating Precision-Recall curve plot {title_suffix}...")
    try:
        # Convert to binary classification for precision-recall calculation
        y_test_binary = (y_test > 0).astype(int)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test_binary, y_pred)
        
        # Calculate average precision score (equivalent to area under PR curve)
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(y_test_binary, y_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, color='blue', lw=2, 
               label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
        
        # Random baseline
        baseline = sum(y_test_binary) / len(y_test_binary)
        plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', 
                   label=f'Random Baseline (Prevalence = {baseline:.4f})')
        
        # Add thresholds markers
        # We'll plot a few key threshold points for reference
        if len(thresholds) > 0:  # Make sure we have thresholds
            threshold_markers = [0.1, 1, 5, 10, 20, 50, 100]
            for t in threshold_markers:
                try:
                    if len(thresholds) > 1:  # Need at least 2 thresholds to find closest
                        # Find the closest threshold index
                        idx = (np.abs(thresholds - t)).argmin()
                        if idx < len(recall) - 1:  # precision-recall arrays are one longer than thresholds
                            plt.plot(recall[idx], precision[idx], 'o', markersize=8, 
                                    label=f'Threshold ≈ {thresholds[idx]:.1f}')
                except:
                    pass  # Skip if threshold not found
                
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (True Positive Rate)', fontsize=12)
        plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
        plt.title(f'Precision-Recall Curve {title_suffix}', fontsize=14)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        logging.info(f"Precision-Recall curve plot saved to {file_path}")
    except Exception as e:
        logging.error(f"Error generating Precision-Recall curve plot {title_suffix}: {e}", exc_info=True)

# --- Calibration Plot ---
def plot_prediction_calibration(y_test: pd.Series, y_pred: np.ndarray, file_path: Path, title_suffix: str = ""):
    """Creates a calibration plot to assess prediction reliability."""
    logging.info(f"Generating calibration plot {title_suffix}...")
    try:
        # We'll bin predictions and compare with actual values
        # Create bins for predicted values
        n_bins = 10
        bins = np.linspace(0, y_pred.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Initialize arrays to store results
        mean_predicted = np.zeros(n_bins)
        mean_actual = np.zeros(n_bins)
        count = np.zeros(n_bins)
        
        # Bin predictions and calculate means
        for i in range(n_bins):
            bin_mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
            if np.sum(bin_mask) > 0:
                mean_predicted[i] = np.mean(y_pred[bin_mask])
                mean_actual[i] = np.mean(y_test.iloc[bin_mask])
                count[i] = np.sum(bin_mask)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot calibration curve
        plt.scatter(mean_predicted, mean_actual, s=count/sum(count)*500, 
                   c=count, cmap='viridis', alpha=0.7, edgecolor='k')
        
        # Add colorbar to show counts
        plt.colorbar(label='Count in Bin')
        
        # Add perfect calibration reference line
        max_val = max(np.max(mean_predicted), np.max(mean_actual))
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Calibration')
        
        # Add annotations for each point
        for i in range(n_bins):
            if count[i] > 0:
                plt.annotate(f'{count[i]:.0f}', 
                            (mean_predicted[i], mean_actual[i]),
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center')
        
        plt.xlabel('Mean Predicted Value', fontsize=12)
        plt.ylabel('Mean Actual Value', fontsize=12)
        plt.title(f'Prediction Calibration Plot {title_suffix}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add a regression line to show trend
        from scipy import stats
        valid_idx = count > 0
        if np.sum(valid_idx) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                mean_predicted[valid_idx], mean_actual[valid_idx])
            
            plt.plot(mean_predicted, intercept + slope*mean_predicted, 'g-', 
                    label=f'Regression Line (r²={r_value**2:.4f})')
            plt.legend()
            
            # Add regression information
            regression_text = (f"Slope: {slope:.4f}\n"
                              f"Intercept: {intercept:.4f}\n"
                              f"R²: {r_value**2:.4f}\n"
                              f"P-value: {p_value:.4f}")
            
            plt.annotate(regression_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                        va="top", ha="left", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        logging.info(f"Calibration plot saved to {file_path}")
    except Exception as e:
        logging.error(f"Error generating calibration plot {title_suffix}: {e}", exc_info=True)

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
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        prevalence = (tp + fn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        
        # Process class report 
        precision_0 = class_report.get('0', {}).get('precision', 0)
        recall_0 = class_report.get('0', {}).get('recall', 0)
        f1_0 = class_report.get('0', {}).get('f1-score', 0)
        precision_1 = class_report.get('1', {}).get('precision', 0)
        recall_1 = class_report.get('1', {}).get('recall', 0)
        f1_1 = class_report.get('1', {}).get('f1-score', 0)

        results = {
            f'clf_accuracy_thresh{threshold}': accuracy,
            f'clf_tn_thresh{threshold}': tn,
            f'clf_fp_thresh{threshold}': fp,
            f'clf_fn_thresh{threshold}': fn,
            f'clf_tp_thresh{threshold}': tp,
            f'clf_specificity_thresh{threshold}': specificity,
            f'clf_prevalence_thresh{threshold}': prevalence,
            f'clf_precision_0_thresh{threshold}': precision_0,
            f'clf_recall_0_thresh{threshold}': recall_0,
            f'clf_f1_0_thresh{threshold}': f1_0,
            f'clf_precision_1_thresh{threshold}': precision_1,
            f'clf_recall_1_thresh{threshold}': recall_1,
            f'clf_f1_1_thresh{threshold}': f1_1,
        }
        return results

    except Exception as e:
        logging.error(f"Error during classification proxy evaluation for {model_name}: {e}", exc_info=True)
        return None


# --- Main Evaluation Function (Enhanced) ---
def evaluate_single_run(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, feature_names: list) -> dict | None:
    """Evaluates a single trained model run, returns metrics, and generates plots."""
    logging.info(f"--- Evaluating {model_name} ---")
    X_test_processed = X_test.copy()
    results = {}
    try:
        # --- Ensure correct numeric types for evaluation ---
        # Ensure y_test is float64 to avoid numeric overflow issues
        if not y_test.dtype == np.float64:
            logging.info(f"Converting y_test from {y_test.dtype} to float64 for stable calculations")
            y_test = y_test.astype(np.float64)
            
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
        y_pred = np.clip(y_pred, 0, None)  # Ensure no negative predictions

        # --- Regression Metrics ---
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        acc_5min = np.mean(np.abs(y_test - y_pred) <= 5) * 100
        
        # Additional metrics
        median_ae = np.median(np.abs(y_test - y_pred))  # Median absolute error
        
        # Calculate MAPE carefully to avoid overflow
        with np.errstate(all='ignore'):  # Temporarily ignore numeric warnings
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-10
            mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(epsilon, np.abs(y_test)))) * 100)

        logging.info(f"🔍 Regression Results for {model_name}:")
        logging.info(f"➤ RMSE: {rmse:.2f}")
        logging.info(f"➤ MAE: {mae:.2f}")
        logging.info(f"➤ Median AE: {median_ae:.2f}")
        logging.info(f"➤ R² Score: {r2:.4f}")
        logging.info(f"➤ Accuracy within ±5 minutes: {acc_5min:.2f}%")
        logging.info(f"➤ MAPE: {mape:.2f}%")

        results.update({
            'model_name': model_name, 
            'rmse': rmse, 
            'mae': mae, 
            'median_ae': median_ae,
            'r2': r2, 
            'acc_5min': acc_5min,
            'mape': mape,
            'test_set_size': len(y_test),
            'actual_mean': y_test.mean(), 
            'actual_std': y_test.std(),
            'predicted_mean': y_pred.mean(), 
            'predicted_std': y_pred.std()
        })

        # --- Non-Zero Metrics ---
        non_zero_indices = y_test > 0
        mae_nz, rmse_nz, r2_nz, mape_nz = np.nan, np.nan, np.nan, np.nan
        count_nz = 0
        if non_zero_indices.any():
            y_test_nz = y_test[non_zero_indices]
            y_pred_nz = y_pred[non_zero_indices]
            count_nz = len(y_test_nz)
            if count_nz > 0:
                 # Use float64 for all calculations to avoid overflow
                 mae_nz = float(mean_absolute_error(y_test_nz, y_pred_nz))
                 rmse_nz = float(np.sqrt(mean_squared_error(y_test_nz, y_pred_nz)))
                 r2_nz = float(r2_score(y_test_nz, y_pred_nz))
                 
                 # Calculate MAPE carefully to avoid overflow
                 with np.errstate(all='ignore'):  # Temporarily ignore numeric warnings
                    # Add a small epsilon to avoid division by zero
                    epsilon = 1e-10
                    mape_nz = float(np.mean(np.abs((y_test_nz - y_pred_nz) / np.maximum(epsilon, np.abs(y_test_nz)))) * 100)
                 
                 logging.info("--- Metrics for Actual > 0 ---")
                 logging.info(f"MAE (Non-Zero): {mae_nz:.2f}")
                 logging.info(f"RMSE (Non-Zero): {rmse_nz:.2f}")
                 logging.info(f"R2 (Non-Zero): {r2_nz:.4f}")
                 logging.info(f"MAPE (Non-Zero): {mape_nz:.2f}%")
            else:
                 logging.warning("Non-zero subset was empty during evaluation.")
        else:
            logging.warning("No non-zero actual delays found in test set.")

        results.update({
            'rmse_non_zero': rmse_nz, 
            'mae_non_zero': mae_nz, 
            'r2_non_zero': r2_nz, 
            'mape_non_zero': mape_nz,
            'count_non_zero': count_nz
        })

        # --- Classification Proxy Metrics ---
        # Evaluate at multiple thresholds
        thresholds = [0, 5, 10, 15, 30]
        for threshold in thresholds:
            clf_results = evaluate_regression_as_classification(
                y_test, y_pred, threshold, model_name
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
        
        # New plot paths
        confusion_matrix_path = config.PLOTS_DIR / f"{plot_base_name}_confusion_matrix.png"
        roc_curve_path = config.PLOTS_DIR / f"{plot_base_name}_roc_curve.png"
        pr_curve_path = config.PLOTS_DIR / f"{plot_base_name}_pr_curve.png"
        calibration_path = config.PLOTS_DIR / f"{plot_base_name}_calibration.png"

        # Generate standard plots
        plot_predicted_vs_actual(y_test, y_pred, pred_actual_path, title_suffix=f"({model_name})")
        plot_residuals(y_test, y_pred, residuals_path, title_suffix=f"({model_name})")
        plot_error_distribution(y_test, y_pred, error_dist_path, title_suffix=f"({model_name})")
        plot_feature_importance(model, feature_names, feat_imp_path, title_suffix=f"({model_name})")
        
        # Generate new plots
        plot_confusion_matrix(y_test, y_pred, confusion_matrix_path, threshold=config.CLASSIFICATION_THRESHOLD, title_suffix=f"({model_name})")
        plot_roc_curve(y_test, y_pred, roc_curve_path, title_suffix=f"({model_name})")
        plot_precision_recall_curve(y_test, y_pred, pr_curve_path, title_suffix=f"({model_name})")
        plot_prediction_calibration(y_test, y_pred, calibration_path, title_suffix=f"({model_name})")

        return results

    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {e}", exc_info=True)
        return None
