# eu_flight_predictor/eu_evaluate_predictions.py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns  # For a nicer confusion matrix plot
import os
import json

# Import necessary configurations
from eu_config import (
    PROCESSED_EU_CHAINS_DIR,
    EU_TEST_LABELS_FILE,  # To load true labels
    EU_PREDICTION_MODEL_TYPE,  # To get the model type for filenames
    DELAY_STATUS_MAP_EU,  # For human-readable labels
    NUM_CLASSES_CLASSIFICATION,  # For class names/labels in plots
)

# Define output directory for evaluation results
EVALUATION_RESULTS_DIR = (
    PROCESSED_EU_CHAINS_DIR.parent / "eu_evaluation_results"
)  # e.g., eu_flight_predictor/eu_evaluation_results/
EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_eu_predictions():
    print(
        f"--- Evaluating EU Predictions for Model: {EU_PREDICTION_MODEL_TYPE.upper()} ---"
    )

    # Construct file paths for predictions based on model type
    pred_indices_filename = f"eu_predicted_indices_{EU_PREDICTION_MODEL_TYPE}.npy"
    pred_probs_filename = f"eu_predicted_probabilities_{EU_PREDICTION_MODEL_TYPE}.npy"

    predicted_indices_path = PROCESSED_EU_CHAINS_DIR / pred_indices_filename
    predicted_probs_path = PROCESSED_EU_CHAINS_DIR / pred_probs_filename

    # Load predicted indices
    if not predicted_indices_path.exists():
        print(
            f"Predictions file {predicted_indices_path} not found. Please ensure 'eu_predict.py' ran successfully for model type '{EU_PREDICTION_MODEL_TYPE}' and saved its output."
        )
        return
    try:
        y_pred_indices = np.load(predicted_indices_path)
        print(f"Successfully loaded predicted indices from {predicted_indices_path}")
    except Exception as e:
        print(f"Error loading predicted indices from {predicted_indices_path}: {e}")
        return

    # Load predicted probabilities (optional, but good for some metrics or analysis)
    y_pred_probs = None
    if predicted_probs_path.exists():
        try:
            y_pred_probs = np.load(predicted_probs_path)
            print(
                f"Successfully loaded predicted probabilities from {predicted_probs_path}"
            )
        except Exception as e:
            print(
                f"Warning: Error loading predicted probabilities from {predicted_probs_path}: {e}. Proceeding without probabilities."
            )
    else:
        print(
            f"Predicted probabilities file {predicted_probs_path} not found. Proceeding without probabilities."
        )

    # Load true labels
    if not EU_TEST_LABELS_FILE.exists():
        print(
            f"True labels file {EU_TEST_LABELS_FILE} not found. Cannot perform evaluation."
        )
        return
    try:
        y_true = np.load(EU_TEST_LABELS_FILE)
        print(f"Successfully loaded true labels from {EU_TEST_LABELS_FILE}")
    except Exception as e:
        print(f"Error loading true labels from {EU_TEST_LABELS_FILE}: {e}")
        return

    if len(y_pred_indices) != len(y_true):
        print(
            f"WARNING: Length mismatch! Predictions ({len(y_pred_indices)}) and True Labels ({len(y_true)}). Truncating to shorter length for evaluation."
        )
        min_len = min(len(y_pred_indices), len(y_true))
        y_pred_indices = y_pred_indices[:min_len]
        y_true = y_true[:min_len]
        if y_pred_probs is not None:
            y_pred_probs = y_pred_probs[:min_len]

    if len(y_true) == 0:
        print(
            "No data to evaluate (true labels array is empty after potential truncation)."
        )
        return

    # --- Perform Evaluation ---
    accuracy = accuracy_score(y_true, y_pred_indices)

    # Use NUM_CLASSES_CLASSIFICATION to generate target_names if DELAY_STATUS_MAP_EU covers all
    # Or define them explicitly
    target_names = [
        DELAY_STATUS_MAP_EU.get(i, f"Class {i}")
        for i in range(NUM_CLASSES_CLASSIFICATION)
    ]

    report = classification_report(
        y_true, y_pred_indices, target_names=target_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(
        y_true, y_pred_indices, labels=list(range(NUM_CLASSES_CLASSIFICATION))
    )

    print("\n--- Evaluation Metrics ---")
    print(f"Model Type: {EU_PREDICTION_MODEL_TYPE.upper()}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    # Save metrics to a JSON file
    metrics_data = {
        "model_type": EU_PREDICTION_MODEL_TYPE,
        "accuracy": accuracy,
        "classification_report": classification_report(
            y_true,
            y_pred_indices,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),  # Convert numpy array to list for JSON serialization
    }
    metrics_output_path = (
        EVALUATION_RESULTS_DIR
        / f"eu_evaluation_metrics_{EU_PREDICTION_MODEL_TYPE}.json"
    )
    with open(metrics_output_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\nEvaluation metrics saved to: {metrics_output_path}")

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix - EU Data - Model: {EU_PREDICTION_MODEL_TYPE.upper()}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_figure_path = (
        EVALUATION_RESULTS_DIR / f"eu_confusion_matrix_{EU_PREDICTION_MODEL_TYPE}.png"
    )
    plt.savefig(cm_figure_path)
    print(f"Confusion matrix plot saved to: {cm_figure_path}")
    # plt.show() # Optionally display plot

    print("\n--- EU Evaluation Finished ---")


if __name__ == "__main__":
    # This will automatically pick up EU_PREDICTION_MODEL_TYPE from eu_config when imported
    evaluate_eu_predictions()
