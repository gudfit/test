# eu_flight_predictor/eu_predict.py
import torch
import json
import sys
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

from eu_config import (
    DEVICE,
    CHAIN_LENGTH,
    NUM_CLASSES_CLASSIFICATION,
    MODEL_PATH_FOR_EU_PREDICTION,
    EU_TEST_CHAINS_FILE,
    EU_TEST_LABELS_FILE,
    PROCESSED_EU_CHAINS_DIR,
    load_original_data_stats_for_scaling,
    load_model_hyperparameters,
    MAIN_PROJECT_ROOT,  # Used for constructing the sys.path entry
)

# Dynamically add the parent of 'src' (i.e., flightChainClassifier directory) to sys.path
# MAIN_PROJECT_ROOT from eu_config.py is /.../flightChainClassifier
flight_classifier_project_dir_for_sys_path = MAIN_PROJECT_ROOT
if str(flight_classifier_project_dir_for_sys_path) not in sys.path:
    sys.path.insert(0, str(flight_classifier_project_dir_for_sys_path))
    print(
        f"DEBUG: Added to sys.path for import: {flight_classifier_project_dir_for_sys_path}"
    )

try:
    # Now import as if 'src' is a package within flightChainClassifier
    from src.modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model

    print(
        f"DEBUG: Successfully imported QTSimAM_CNN_LSTM_Model using 'from src.modeling...'"
    )
except ImportError as e:
    print(
        f"ERROR: Could not import QTSimAM_CNN_LSTM_Model using 'from src.modeling...'. Error: {e}"
    )
    print(f"Current sys.path: {sys.path}")
    print(
        f"Attempted to add '{flight_classifier_project_dir_for_sys_path}' to sys.path."
    )
    print(
        "Ensure this path correctly points to your 'flightChainClassifier' project directory, which contains the 'src' sub-package."
    )
    print(
        "Also check for typos in the module path 'src.modeling.queue_augment_models' or class name."
    )
    sys.exit(1)
except ModuleNotFoundError as mnfe:
    print(
        f"ERROR: ModuleNotFoundError while trying to import QTSimAM_CNN_LSTM_Model. Error: {mnfe}"
    )
    print(
        f"This usually means a submodule (e.g., 'src' or 'src.modeling') is missing or not on the path correctly."
    )
    print(
        f"Attempted import after adding '{flight_classifier_project_dir_for_sys_path}' to sys.path. Please verify its contents and __init__.py files."
    )
    sys.exit(1)


DELAY_STATUS_MAP_EU = {
    0: "On Time / Slight Delay (<= 15 min)",
    1: "Delayed (15-60 min)",
    2: "Significantly Delayed (60-120 min)",
    3: "Severely Delayed (120-240 min)",
    4: "Extremely Delayed (> 240 min)",
}
UNKNOWN_STATUS_EU = "Unknown Delay Status"


class EUPredictor:
    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.original_data_stats = load_original_data_stats_for_scaling()
        self.model_hyperparams = load_model_hyperparameters()

        if not self.original_data_stats:
            raise ValueError(
                "Failed to load original data stats (from flightChainClassifier training). These are required for feature count."
            )

        self._load_model()

    def _load_model(self):
        if not MODEL_PATH_FOR_EU_PREDICTION.exists():
            raise FileNotFoundError(
                f"Pre-trained model (.pt file) not found at the configured path: {MODEL_PATH_FOR_EU_PREDICTION}"
            )

        num_features = self.original_data_stats.get("num_features")
        if num_features is None:
            raise ValueError(
                "'num_features' key missing from original data stats (data_stats.json). This is required for model initialization."
            )

        fallback_cnn_channels = [64, 128, 256]
        fallback_kernel_size = 3
        fallback_lstm_hidden = 128
        fallback_lstm_layers = 1
        fallback_lstm_bidir = False
        fallback_dropout_rate = 0.2
        fallback_attn_heads = 4

        cnn_channels = fallback_cnn_channels
        kernel_size = fallback_kernel_size
        lstm_hidden = fallback_lstm_hidden
        lstm_layers = fallback_lstm_layers
        lstm_bidir = fallback_lstm_bidir
        dropout_rate = fallback_dropout_rate
        attn_heads = fallback_attn_heads

        loaded_params_source = "fallback defaults"

        if self.model_hyperparams:
            loaded_params_source = "loaded file"
            print(
                f"DEBUG: Using model_hyperparams loaded by eu_config: {self.model_hyperparams}"
            )

            cnn_channels = self.model_hyperparams.get(
                "cnn_channels", fallback_cnn_channels
            )
            kernel_size = self.model_hyperparams.get(
                "kernel_size", fallback_kernel_size
            )
            lstm_hidden = self.model_hyperparams.get(
                "lstm_hidden_size", fallback_lstm_hidden
            )
            lstm_layers = self.model_hyperparams.get(
                "lstm_num_layers", fallback_lstm_layers
            )
            dropout_rate = self.model_hyperparams.get(
                "dropout_rate",
                self.model_hyperparams.get("dropout", fallback_dropout_rate),
            )
            lstm_bidir = self.model_hyperparams.get(
                "lstm_bidir",
                self.model_hyperparams.get("lstm_bidirectional", fallback_lstm_bidir),
            )
            attn_heads = self.model_hyperparams.get("attn_heads", fallback_attn_heads)
        else:
            print(
                f"Warning: Model hyperparameters file not found or failed to load. Using fallback defaults for QTSimAM model architecture."
            )

        print(
            f"Initializing QTSimAM_CNN_LSTM_Model with parameters (source: {loaded_params_source}):"
        )
        print(
            f"  num_features={num_features}, num_classes={NUM_CLASSES_CLASSIFICATION}"
        )
        print(f"  cnn_channels={cnn_channels}, kernel_size={kernel_size}")
        print(
            f"  lstm_hidden={lstm_hidden}, lstm_layers={lstm_layers}, lstm_bidir={lstm_bidir}"
        )
        print(f"  dropout_rate={dropout_rate}, attn_heads={attn_heads}")

        self.model = QTSimAM_CNN_LSTM_Model(
            num_features=num_features,
            num_classes=NUM_CLASSES_CLASSIFICATION,
            cnn_channels=cnn_channels,
            kernel_size=kernel_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            lstm_bidir=lstm_bidir,
            dropout_rate=dropout_rate,
            attn_heads=attn_heads,
        )
        try:
            print(
                f"Loading model state_dict into QTSimAM_CNN_LSTM_Model from {MODEL_PATH_FOR_EU_PREDICTION}..."
            )
            self.model.load_state_dict(
                torch.load(MODEL_PATH_FOR_EU_PREDICTION, map_location=self.device),
                strict=True,
            )
        except RuntimeError as e:
            print(
                f"!!! RuntimeError loading model state_dict for QTSimAM_CNN_LSTM_Model: {e}"
            )
            print(
                "This very often indicates a mismatch between the loaded model's saved architecture and the architecture defined by the current parameters used for initialization."
            )
            print(
                f"Please ensure the parameters printed above precisely match the parameters of the model saved at: {MODEL_PATH_FOR_EU_PREDICTION}"
            )
            meta_file_to_check = MODEL_PATH_FOR_EU_PREDICTION.with_suffix(".meta.json")
            print(
                f"RECOMMENDATION: Compare with the .meta.json file (path: {meta_file_to_check}) "
                f"or the specific hyperparameter file (e.g., best_hyperparameters_qtsimam.json if that was used for training this model)."
            )
            sys.exit(1)

        self.model.to(self.device)
        self.model.eval()
        print(f"QTSimAM_CNN_LSTM_Model model loaded successfully to {self.device}.")

    @torch.no_grad()
    def predict_on_eu_chains(self, eu_chains_file_path):
        if not eu_chains_file_path.exists():
            print(f"ERROR: Processed EU chains file not found: {eu_chains_file_path}")
            return None, None

        try:
            eu_chains_np = np.load(eu_chains_file_path)
        except Exception as e:
            print(
                f"ERROR: Could not load EU chains numpy file from {eu_chains_file_path}: {e}"
            )
            return None, None

        if eu_chains_np.ndim != 3 or eu_chains_np.shape[1] != CHAIN_LENGTH:
            print(
                f"ERROR: EU chains numpy array has incorrect shape: {eu_chains_np.shape}. Expected (N, {CHAIN_LENGTH}, Features)."
            )
            return None, None

        expected_num_features = self.original_data_stats.get("num_features")
        if eu_chains_np.shape[2] != expected_num_features:
            print(
                f"ERROR: Feature count mismatch. EU chains have {eu_chains_np.shape[2]} features, model expects {expected_num_features}."
            )
            print(
                "This likely means an issue in 'eu_chain_constructor.py' in aligning features with 'data_stats.json'."
            )
            return None, None

        if len(eu_chains_np) == 0:
            print("Warning: EU chains file is empty. No predictions to make.")
            return np.array([]), np.array([])

        print(
            f"Predicting on {len(eu_chains_np)} EU flight chains using QTSimAM_CNN_LSTM_Model..."
        )
        all_predictions_indices, all_probabilities_np = [], []
        batch_size = 64
        for i in tqdm(
            range(0, len(eu_chains_np), batch_size),
            desc="Batch Prediction",
            unit="batch",
        ):
            batch_chains_np = eu_chains_np[i : i + batch_size]
            batch_chains_tensor = torch.tensor(batch_chains_np, dtype=torch.float32).to(
                self.device
            )

            outputs = self.model(batch_chains_tensor)

            probabilities_batch = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_indices_batch = np.argmax(probabilities_batch, axis=1)

            all_predictions_indices.extend(predicted_indices_batch.tolist())
            all_probabilities_np.append(probabilities_batch)

        if not all_predictions_indices:
            return np.array([]), np.array([])

        return np.array(all_predictions_indices), np.concatenate(
            all_probabilities_np, axis=0
        )


def run_eu_prediction():
    print("--- Starting EU Flight Chain Prediction ---")
    if not EU_TEST_CHAINS_FILE.exists():
        print(
            f"Processed EU chains file {EU_TEST_CHAINS_FILE} not found. Run 'eu_chain_constructor.py' first to generate it."
        )
        sys.exit(1)

    try:
        predictor = EUPredictor()
    except Exception as e:
        print(f"FATAL: Failed to initialize EUPredictor: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None

    predicted_category_indices, predicted_probabilities = (
        predictor.predict_on_eu_chains(EU_TEST_CHAINS_FILE)
    )

    if predicted_category_indices is None:
        print("Prediction process failed.")
        return None, None, None

    if len(predicted_category_indices) == 0:
        print(
            "No predictions were made (e.g., input chains file was empty or processing issue)."
        )
        return np.array([]), np.array([]), None

    predicted_statuses = [
        DELAY_STATUS_MAP_EU.get(idx, UNKNOWN_STATUS_EU)
        for idx in predicted_category_indices
    ]

    print(f"\n--- Prediction Summary ---")
    print(f"Total chains predicted: {len(predicted_category_indices)}")
    for i in range(min(5, len(predicted_statuses))):
        print(
            f"  Chain {i}: Predicted CatIdx: {predicted_category_indices[i]}, Status: '{predicted_statuses[i]}', Probs: {np.round(predicted_probabilities[i],3)}"
        )

    true_labels_np = None
    if EU_TEST_LABELS_FILE.exists():
        try:
            true_labels_np = np.load(EU_TEST_LABELS_FILE)
            if len(true_labels_np) != len(predicted_category_indices):
                print(
                    f"Warning: Length mismatch between true labels ({len(true_labels_np)}) and predictions ({len(predicted_category_indices)}). Evaluation might be affected."
                )
        except Exception as e:
            print(
                f"Warning: Could not load true labels file {EU_TEST_LABELS_FILE}: {e}"
            )
            true_labels_np = None
    else:
        print(
            f"True labels file {EU_TEST_LABELS_FILE} not found. Full evaluation against true labels not possible here."
        )

    print("--- EU Prediction Finished ---")
    return predicted_category_indices, predicted_probabilities, true_labels_np


if __name__ == "__main__":
    predicted_indices, probabilities, true_labels = run_eu_prediction()

    if predicted_indices is not None and len(predicted_indices) > 0:
        PROCESSED_EU_CHAINS_DIR.mkdir(parents=True, exist_ok=True)

        output_pred_indices_path = PROCESSED_EU_CHAINS_DIR / "eu_predicted_indices.npy"
        np.save(output_pred_indices_path, predicted_indices)
        print(f"Predicted indices saved to {output_pred_indices_path}")

        if probabilities is not None and len(probabilities) > 0:
            output_pred_probs_path = (
                PROCESSED_EU_CHAINS_DIR / "eu_predicted_probabilities.npy"
            )
            np.save(output_pred_probs_path, probabilities)
            print(f"Predicted probabilities saved to {output_pred_probs_path}")
        print(
            "You can now run 'eu_evaluate_predictions.py' if true labels are available and were loaded."
        )
    elif predicted_indices is not None and len(predicted_indices) == 0:
        print("Prediction ran, but no chains resulted in predictions.")
    else:
        print("Prediction script did not produce valid output.")
