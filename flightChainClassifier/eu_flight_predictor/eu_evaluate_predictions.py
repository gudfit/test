# eu_flight_predictor/eu_evaluate_predictions.py
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import os

# Assuming eu_config is in the same directory
from eu_config import (
    PROCESSED_EU_CHAINS_DIR,
    NUM_CLASSES_CLASSIFICATION,
    EU_TEST_LABELS_FILE,  # <<< ADD THIS IMPORT
)

DELAY_STATUS_MAP_EU = {
    0: "On Time / Slight Delay (<= 15 min)",
    1: "Delayed (15-60 min)",
    2: "Significantly Delayed (60-120 min)",
    3: "Severely Delayed (120-240 min)",
    4: "Extremely Delayed (> 240 min)",
}


def evaluate_eu_predictions(predicted_indices_path, true_labels_path):
    if not predicted_indices_path.exists():
        print(f"ERROR: Predicted indices file not found: {predicted_indices_path}")
        return
    if not true_labels_path.exists():
        print(
            f"ERROR: True labels file not found: {true_labels_path}. Cannot evaluate."
        )
        return

    predictions_np = np.load(predicted_indices_path)
    true_labels_np = np.load(true_labels_path)

    if len(predictions_np) != len(true_labels_np):
        print(
            f"ERROR: Mismatch in length of predictions ({len(predictions_np)}) and true labels ({len(true_labels_np)})."
        )
        return

    if len(predictions_np) == 0:
        print("No predictions to evaluate.")
        return

    print(f"\n--- Evaluating {len(predictions_np)} EU Predictions ---")

    acc = accuracy_score(true_labels_np, predictions_np)

    # Ensure target_names covers all unique labels present, up to NUM_CLASSES
    present_labels = np.unique(np.concatenate((true_labels_np, predictions_np)))
    max_label_val = int(max(present_labels.max(), NUM_CLASSES_CLASSIFICATION - 1))
    target_names_list = [
        DELAY_STATUS_MAP_EU.get(i, f"Class {i}") for i in range(max_label_val + 1)
    ]

    # Filter target_names to only those present if sklearn report requires it for some versions
    # For classification_report, labels argument can be used to specify order and presence
    unique_sorted_labels = np.unique(
        np.concatenate((true_labels_np, predictions_np))
    ).astype(int)

    report = classification_report(
        true_labels_np,
        predictions_np,
        labels=unique_sorted_labels,  # Use only present labels
        target_names=[
            DELAY_STATUS_MAP_EU.get(i, f"Class_{i}") for i in unique_sorted_labels
        ],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(
        true_labels_np, predictions_np, labels=unique_sorted_labels
    )  # Use same labels for CM

    print(f"\nAccuracy on EU data: {acc:.4f}\n")
    print("Classification Report on EU data:")
    print(report)

    # Save metrics
    eval_results_dir = PROCESSED_EU_CHAINS_DIR.parent / "eu_evaluation_results"
    eval_results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = eval_results_dir / "eu_transfer_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "classification_report": report}, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_path}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjusted size
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            DELAY_STATUS_MAP_EU.get(i, f"Class_{i}") for i in unique_sorted_labels
        ],
    )
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d", xticks_rotation="vertical")
    plt.title("Confusion Matrix - EU Data Transfer Evaluation")
    fig.tight_layout()
    cm_plot_path = eval_results_dir / "eu_transfer_confusion_matrix.png"
    fig.savefig(cm_plot_path)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {cm_plot_path}")
    print("--- EU Evaluation Finished ---")


if __name__ == "__main__":
    # Path to the predictions file saved by eu_predict.py
    preds_path = PROCESSED_EU_CHAINS_DIR / "eu_predicted_indices.npy"

    # labels_path now correctly uses the imported EU_TEST_LABELS_FILE
    # This variable is defined in eu_config.py and points to where
    # eu_chain_constructor.py saves the true labels for the EU test chains.
    labels_path = EU_TEST_LABELS_FILE

    if not preds_path.exists():
        print(
            f"Predictions file {preds_path} not found. Please ensure 'eu_predict.py' ran successfully and saved its output."
        )
    elif not labels_path.exists():
        print(
            f"True labels file {labels_path} not found. Please ensure 'eu_chain_constructor.py' ran successfully and saved the labels."
        )
    else:
        evaluate_eu_predictions(preds_path, labels_path)
