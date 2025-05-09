# eu_flight_predictor/eu_ev.py
# (or name it eu_evaluate_predictions.py)

import numpy as np
import pandas as pd  # Not strictly used in this version, but often useful
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pathlib  # For Path objects

# Import necessary configurations from eu_config.py
try:
    from eu_config import (
        PROCESSED_EU_CHAINS_DIR,
        EU_TEST_LABELS_FILE,
        EU_PREDICTION_MODEL_TYPE,
        DELAY_STATUS_MAP_EU,
        NUM_CLASSES_CLASSIFICATION,
        EU_FLIGHT_PREDICTOR_ROOT,  # If EVALUATION_RESULTS_DIR is relative to this
    )
except ImportError as e:
    print(f"Error importing from eu_config: {e}")
    print(
        "Please ensure eu_config.py is in the same directory or accessible in PYTHONPATH."
    )
    exit(1)

# Define output directory for evaluation results
# Place it within the eu_flight_predictor directory for organization
EVALUATION_RESULTS_DIR = EU_FLIGHT_PREDICTOR_ROOT / "eu_evaluation_results"
EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_eu_predictions():
    print(
        f"--- Evaluating EU Predictions for Model: {EU_PREDICTION_MODEL_TYPE.upper()} ---"
    )

    # Construct file paths for predictions based on model type
    pred_indices_filename = f"eu_predicted_indices_{EU_PREDICTION_MODEL_TYPE}.npy"
    pred_probs_filename = f"eu_predicted_probabilities_{EU_PREDICTION_MODEL_TYPE}.npy"  # Though not used in current metrics

    predicted_indices_path = PROCESSED_EU_CHAINS_DIR / pred_indices_filename
    predicted_probs_path = (
        PROCESSED_EU_CHAINS_DIR / pred_probs_filename
    )  # Path for probabilities

    # Load predicted indices
    if not predicted_indices_path.exists():
        print(f"ERROR: Predictions file {predicted_indices_path} not found.")
        print(
            f"       Please ensure 'eu_predict.py' ran successfully for model type '{EU_PREDICTION_MODEL_TYPE}' and saved its output."
        )
        return
    try:
        y_pred_indices = np.load(predicted_indices_path)
        print(f"Successfully loaded predicted indices from {predicted_indices_path}")
    except Exception as e:
        print(
            f"ERROR: Could not load predicted indices from {predicted_indices_path}: {e}"
        )
        return

    # Load predicted probabilities (optional, can be used for ROC AUC, etc. later)
    y_pred_probs = None
    if predicted_probs_path.exists():
        try:
            y_pred_probs = np.load(predicted_probs_path)
            print(
                f"Successfully loaded predicted probabilities from {predicted_probs_path}"
            )
        except Exception as e:
            print(
                f"Warning: Could not load predicted probabilities from {predicted_probs_path}: {e}. Proceeding without them."
            )
    else:
        print(
            f"Info: Predicted probabilities file {predicted_probs_path} not found. Proceeding without probabilities."
        )

    # Load true labels
    if not EU_TEST_LABELS_FILE.exists():
        print(
            f"ERROR: True labels file {EU_TEST_LABELS_FILE} not found. Cannot perform evaluation."
        )
        return
    try:
        y_true = np.load(EU_TEST_LABELS_FILE)
        print(f"Successfully loaded true labels from {EU_TEST_LABELS_FILE}")
    except Exception as e:
        print(f"ERROR: Could not load true labels from {EU_TEST_LABELS_FILE}: {e}")
        return

    # Basic check for data presence
    if len(y_pred_indices) == 0:
        print("ERROR: Predicted indices array is empty. Cannot evaluate.")
        return
    if len(y_true) == 0:
        print("ERROR: True labels array is empty. Cannot evaluate.")
        return

    # Handle potential length mismatch (e.g., if one file was truncated or from a different run)
    if len(y_pred_indices) != len(y_true):
        print(
            f"WARNING: Length mismatch between predictions ({len(y_pred_indices)}) and true labels ({len(y_true)})."
        )
        min_len = min(len(y_pred_indices), len(y_true))
        print(
            f"         Truncating both to the shorter length: {min_len} for evaluation."
        )
        y_pred_indices = y_pred_indices[:min_len]
        y_true = y_true[:min_len]
        if y_pred_probs is not None:
            y_pred_probs = y_pred_probs[:min_len]

    if len(y_true) == 0:  # Check again after potential truncation
        print("No data to evaluate after handling length mismatch.")
        return

    # --- Perform Evaluation ---
    accuracy = accuracy_score(y_true, y_pred_indices)

    # Define all possible class labels and their names for the report
    # This ensures the report includes all classes, even if some have 0 samples
    all_possible_labels = list(
        range(NUM_CLASSES_CLASSIFICATION)
    )  # e.g., [0, 1, 2, 3, 4]

    # Generate target names from DELAY_STATUS_MAP_EU, ensuring order matches all_possible_labels
    target_names = [
        DELAY_STATUS_MAP_EU.get(i, f"Class {i}") for i in all_possible_labels
    ]

    # Classification report (string for printing)
    report_str = classification_report(
        y_true,
        y_pred_indices,
        labels=all_possible_labels,  # Explicitly state all labels to consider
        target_names=target_names,
        digits=4,
        zero_division=0,  # How to handle division by zero (e.g., for classes with no true samples)
    )

    # Classification report (dictionary for saving to JSON)
    report_dict = classification_report(
        y_true,
        y_pred_indices,
        labels=all_possible_labels,  # Explicitly state all labels to consider
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_indices, labels=all_possible_labels)

    print("\n--- Evaluation Metrics ---")
    print(f"Model Type: {EU_PREDICTION_MODEL_TYPE.upper()}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report_str)
    print("\nConfusion Matrix (Rows: True, Cols: Predicted):")
    # Print confusion matrix with labels for clarity
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # Save metrics to a JSON file
    metrics_data = {
        "model_type": EU_PREDICTION_MODEL_TYPE,
        "accuracy": accuracy,
        "classification_report_dict": report_dict,  # Save the dictionary version
        "confusion_matrix_list": cm.tolist(),  # Convert numpy array to list for JSON
    }
    metrics_filename = f"eu_evaluation_metrics_{EU_PREDICTION_MODEL_TYPE}.json"
    metrics_output_path = EVALUATION_RESULTS_DIR / metrics_filename
    try:
        with open(metrics_output_path, "w") as f:
            json.dump(metrics_data, f, indent=4)
        print(f"\nEvaluation metrics saved to: {metrics_output_path}")
    except Exception as e:
        print(f"Error saving evaluation metrics to {metrics_output_path}: {e}")

    # Plot and save confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            annot_kws={"size": 8},
        )  # Smaller annotation font if numbers are large
        plt.title(
            f"Confusion Matrix - EU Data\nModel: {EU_PREDICTION_MODEL_TYPE.upper()}",
            fontsize=14,
        )
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        cm_figure_filename = f"eu_confusion_matrix_{EU_PREDICTION_MODEL_TYPE}.png"
        cm_figure_path = EVALUATION_RESULTS_DIR / cm_figure_filename
        plt.savefig(cm_figure_path)
        print(f"Confusion matrix plot saved to: {cm_figure_path}")
        # plt.show() # Uncomment to display plot interactively if running in a suitable environment
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"Error generating or saving confusion matrix plot: {e}")

    print("\n--- EU Evaluation Finished ---")


if __name__ == "__main__":
    evaluate_eu_predictions()
