# eu_flight_predictor/eu_predict.py
import torch
import json
import sys
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import traceback  # For detailed error printing

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
    MAIN_PROJECT_ROOT,
    EU_PREDICTION_MODEL_TYPE,  # Import the selected model type
    FORCE_LSTM_BIDIR_FOR_EU,  # Import the override flag <<<<<<<<<<< IMPORTED
    DELAY_STATUS_MAP_EU,  # Now imported from eu_config
    UNKNOWN_STATUS_EU,  # Now imported from eu_config
)

# Dynamically add the parent of 'src' (i.e., flightChainClassifier directory) to sys.path
flight_classifier_project_dir_for_sys_path = MAIN_PROJECT_ROOT
if str(flight_classifier_project_dir_for_sys_path) not in sys.path:
    sys.path.insert(0, str(flight_classifier_project_dir_for_sys_path))
    print(
        f"DEBUG: Added to sys.path for import: {flight_classifier_project_dir_for_sys_path}"
    )

# Import project structure from main project to get defaults if needed
# This is needed if fallbacks rely on main project's config.py
try:
    # Attempt to import the main project's config
    from src import config as main_project_config

    print(f"DEBUG: Successfully imported main project config from src.config")
except ImportError:
    main_project_config = None
    print(
        "Warning: Could not import main project config (src.config). Fallbacks might use hardcoded values."
    )
except Exception as e:
    main_project_config = None
    print(f"Warning: An error occurred while importing main project config: {e}")


try:
    # Import all potential model classes
    from src.modeling.flight_chain_models import CBAM_CNN_Model, SimAM_CNN_LSTM_Model
    from src.modeling.queue_augment_models import (
        QTSimAM_CNN_LSTM_Model,
        QTSimAM_MaxPlus_Model,
    )

    print(f"DEBUG: Successfully imported models using 'from src.modeling...'")
except ImportError as e:
    print(
        f"ERROR: Could not import one or more models using 'from src.modeling...'. Error: {e}"
    )
    print(f"Current sys.path: {sys.path}")
    print(
        f"Attempted to add '{flight_classifier_project_dir_for_sys_path}' to sys.path."
    )
    print(
        "Ensure this path correctly points to your 'flightChainClassifier' project directory, which contains the 'src' sub-package."
    )
    sys.exit(1)
except ModuleNotFoundError as mnfe:
    print(f"ERROR: ModuleNotFoundError while trying to import models. Error: {mnfe}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during model imports: {e}")
    traceback.print_exc()
    sys.exit(1)


class EUPredictor:
    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.original_data_stats = load_original_data_stats_for_scaling()
        self.model_hyperparams = load_model_hyperparameters()
        self.model_type_to_load = EU_PREDICTION_MODEL_TYPE

        if not self.original_data_stats:
            raise ValueError(
                "Failed to load original data stats required for feature count."
            )

        self._load_model()

    def _load_model(self):
        if not MODEL_PATH_FOR_EU_PREDICTION.exists():
            raise FileNotFoundError(
                f"Pre-trained model not found: {MODEL_PATH_FOR_EU_PREDICTION}"
            )

        num_features_from_stats = self.original_data_stats.get("num_features")
        if num_features_from_stats is None:
            raise ValueError("'num_features' missing from original data stats.")

        # --- Determine Hyperparameters (with fallbacks and overrides) ---

        # Get defaults from main project config if available, else hardcode
        fallback_cnn_channels = getattr(
            main_project_config, "DEFAULT_CNN_CHANNELS", [64, 128, 256]
        )
        fallback_kernel_size = getattr(main_project_config, "DEFAULT_KERNEL_SIZE", 3)
        fallback_lstm_hidden = getattr(
            main_project_config, "DEFAULT_LSTM_HIDDEN_SIZE", 128
        )
        fallback_lstm_layers = getattr(
            main_project_config, "DEFAULT_LSTM_NUM_LAYERS", 1
        )
        # Default if nothing else specifies bidirectional setting
        fallback_lstm_bidir_default = getattr(
            main_project_config, "DEFAULT_LSTM_BIDIRECTIONAL", False
        )
        fallback_dropout_rate = getattr(
            main_project_config, "DEFAULT_DROPOUT_RATE", 0.2
        )
        fallback_attn_heads = 4  # Not typically in main config, use fallback

        # Load from hyperparams file if available
        loaded_params_source = "main project defaults or hardcoded fallbacks"
        cnn_channels = fallback_cnn_channels
        kernel_size = fallback_kernel_size
        lstm_hidden = fallback_lstm_hidden
        lstm_layers = fallback_lstm_layers
        # Start with default for bidir, check hyperparams later
        lstm_bidir_from_params = fallback_lstm_bidir_default
        dropout_rate = fallback_dropout_rate
        attn_heads = fallback_attn_heads

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
                "lstm_hidden_size",
                self.model_hyperparams.get("lstm_hidden", fallback_lstm_hidden),
            )
            lstm_layers = self.model_hyperparams.get(
                "lstm_num_layers",
                self.model_hyperparams.get("lstm_layers", fallback_lstm_layers),
            )
            # Check for bidirectional flag in loaded params (check both possible keys)
            lstm_bidir_from_params = self.model_hyperparams.get(
                "lstm_bidir",
                self.model_hyperparams.get(
                    "lstm_bidirectional", fallback_lstm_bidir_default
                ),
            )
            dropout_rate = self.model_hyperparams.get(
                "dropout_rate",
                self.model_hyperparams.get("dropout", fallback_dropout_rate),
            )
            attn_heads = self.model_hyperparams.get("attn_heads", fallback_attn_heads)
        else:
            print(
                f"Warning: Model hyperparameters file not found or failed to load. Using defaults for {self.model_type_to_load} model architecture."
            )

        # --- Override Bidirectional based on FORCE flag from eu_config.py --- <<<<<< LOGIC ADDED
        if FORCE_LSTM_BIDIR_FOR_EU is True:
            final_lstm_bidir = True
            print(
                f"INFO: Overriding LSTM bidirectional setting to True based on FORCE_LSTM_BIDIR_FOR_EU flag in eu_config."
            )
        elif FORCE_LSTM_BIDIR_FOR_EU is False:
            final_lstm_bidir = False
            print(
                f"INFO: Overriding LSTM bidirectional setting to False based on FORCE_LSTM_BIDIR_FOR_EU flag in eu_config."
            )
        else:  # FORCE_LSTM_BIDIR_FOR_EU is None
            final_lstm_bidir = lstm_bidir_from_params  # Use value derived from hyperparams file or default
            print(
                f"INFO: Using LSTM bidirectional setting from loaded params or default: lstm_bidir = {final_lstm_bidir} (FORCE_LSTM_BIDIR_FOR_EU is None)."
            )
        # --- End Override Logic ---

        print(
            f"Initializing {self.model_type_to_load.upper()} model with parameters (source: {loaded_params_source}):"
        )

        # --- Model-specific instantiation using FINAL values ---
        if self.model_type_to_load == "cbam":
            # CBAM doesn't use LSTM params
            print(
                f"  num_features={num_features_from_stats}, num_classes={NUM_CLASSES_CLASSIFICATION}"
            )
            print(f"  cnn_channels={cnn_channels}, kernel_size={kernel_size}")
            print(f"  dropout_rate={dropout_rate}, attn_heads={attn_heads}")
            self.model = CBAM_CNN_Model(
                num_features=num_features_from_stats,
                num_classes=NUM_CLASSES_CLASSIFICATION,
                cnn_channels=cnn_channels,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                attn_heads=attn_heads,
            )
        elif self.model_type_to_load == "simam":
            print(
                f"  num_features={num_features_from_stats}, num_classes={NUM_CLASSES_CLASSIFICATION}"
            )
            print(f"  cnn_channels={cnn_channels}, kernel_size={kernel_size}")
            print(
                f"  lstm_hidden={lstm_hidden}, lstm_layers={lstm_layers}, lstm_bidir={final_lstm_bidir}"
            )  # Use final_lstm_bidir
            print(f"  dropout_rate={dropout_rate}, attn_heads={attn_heads}")
            self.model = SimAM_CNN_LSTM_Model(
                num_features=num_features_from_stats,
                num_classes=NUM_CLASSES_CLASSIFICATION,
                cnn_channels=cnn_channels,
                kernel_size=kernel_size,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                lstm_bidir=final_lstm_bidir,  # Pass the FINAL determined value <<<<<<< MODIFIED
                dropout_rate=dropout_rate,
                attn_heads=attn_heads,
            )
        elif self.model_type_to_load == "qtsimam":
            print(
                f"  num_features={num_features_from_stats}, num_classes={NUM_CLASSES_CLASSIFICATION}"
            )
            print(f"  cnn_channels={cnn_channels}, kernel_size={kernel_size}")
            print(
                f"  lstm_hidden={lstm_hidden}, lstm_layers={lstm_layers}, lstm_bidir={final_lstm_bidir}"
            )  # Use final_lstm_bidir
            print(f"  dropout_rate={dropout_rate}, attn_heads={attn_heads}")
            self.model = QTSimAM_CNN_LSTM_Model(
                num_features=num_features_from_stats,
                num_classes=NUM_CLASSES_CLASSIFICATION,
                cnn_channels=cnn_channels,
                kernel_size=kernel_size,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                lstm_bidir=final_lstm_bidir,  # Pass the FINAL determined value <<<<<<< MODIFIED
                dropout_rate=dropout_rate,
                attn_heads=attn_heads,
            )
        elif self.model_type_to_load == "qtsimam_mp":
            num_features_for_constructor = num_features_from_stats + 1
            print(
                f"  num_features_from_stats={num_features_from_stats} -> num_features_for_constructor={num_features_for_constructor} (for QTSimAM_MaxPlus internal augmentation)"
            )
            print(f"  num_classes={NUM_CLASSES_CLASSIFICATION}")
            print(f"  cnn_channels={cnn_channels}, kernel_size={kernel_size}")
            print(
                f"  lstm_hidden={lstm_hidden}, lstm_layers={lstm_layers}, lstm_bidir={final_lstm_bidir}"
            )  # Use final_lstm_bidir
            print(f"  dropout_rate={dropout_rate}, attn_heads={attn_heads}")
            self.model = QTSimAM_MaxPlus_Model(
                num_features=num_features_for_constructor,
                num_classes=NUM_CLASSES_CLASSIFICATION,
                cnn_channels=cnn_channels,
                kernel_size=kernel_size,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                lstm_bidir=final_lstm_bidir,  # Pass the FINAL determined value <<<<<<< MODIFIED
                dropout_rate=dropout_rate,
                attn_heads=attn_heads,
                beta=(
                    self.model_hyperparams.get("beta", 10.0)
                    if self.model_hyperparams
                    else 10.0
                ),
            )
        else:
            raise ValueError(
                f"Unsupported EU_PREDICTION_MODEL_TYPE: '{self.model_type_to_load}'."
            )

        # --- Load State Dict ---
        try:
            print(
                f"Loading model state_dict into {self.model.__class__.__name__} from {MODEL_PATH_FOR_EU_PREDICTION}..."
            )
            self.model.load_state_dict(
                torch.load(MODEL_PATH_FOR_EU_PREDICTION, map_location=self.device),
                strict=True,
            )
        except RuntimeError as e:
            print(
                f"!!! RuntimeError loading model state_dict for {self.model.__class__.__name__}: {e}"
            )
            print(
                "This very often indicates a mismatch between the loaded model's saved architecture and the architecture defined by the current parameters used for initialization."
            )
            print("Current Initialization Parameters:")
            print(f"  Model Type Instantiated: {self.model_type_to_load.upper()}")
            # Explicitly print the final_lstm_bidir value that was used
            print(f"  LSTM Bidirectional Used for Instantiation: {final_lstm_bidir}")
            print(f"  Other Params: (See printout above)")
            print(
                f"Please ensure these parameters (especially lstm_bidir={final_lstm_bidir}) precisely match the parameters of the model saved at: {MODEL_PATH_FOR_EU_PREDICTION}"
            )
            meta_file_to_check = MODEL_PATH_FOR_EU_PREDICTION.with_suffix(".meta.json")
            # Construct potential specific hyperparam filename based on type used for instantiation
            hyperparam_file_to_check = (
                MAIN_PROJECT_ROOT
                / "results"
                / f"best_hyperparameters_{self.model_type_to_load}.json"
            )
            print(
                f"RECOMMENDATION: Check the contents of {meta_file_to_check} or {hyperparam_file_to_check} (if it exists) to confirm the architecture of the saved model."
            )
            print(
                f"Current FORCE_LSTM_BIDIR_FOR_EU setting in eu_config.py is: {FORCE_LSTM_BIDIR_FOR_EU}"
            )
            sys.exit(1)
        except FileNotFoundError:  # Should have been caught earlier, but good practice
            print(f"ERROR: Model file not found at {MODEL_PATH_FOR_EU_PREDICTION}")
            sys.exit(1)
        except Exception as e:  # Catch other potential errors during loading
            print(f"An unexpected error occurred during model state_dict loading: {e}")
            traceback.print_exc()
            sys.exit(1)

        self.model.to(self.device)
        self.model.eval()
        print(
            f"{self.model.__class__.__name__} model loaded successfully to {self.device}."
        )

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
                f"ERROR: Feature count mismatch in input data. EU chains have {eu_chains_np.shape[2]} features, model was expecting inputs processed to {expected_num_features} features (as per data_stats.json from original training)."
            )
            if self.model_type_to_load == "qtsimam_mp":
                print(
                    f"  Note: {self.model.__class__.__name__} internally augments this to {expected_num_features + 1} features, but still expects input data with {expected_num_features} features."
                )
            print(
                "This likely means an issue in 'eu_chain_constructor.py' in aligning features with 'data_stats.json'."
            )
            return None, None

        if len(eu_chains_np) == 0:
            print("Warning: EU chains file is empty. No predictions to make.")
            return np.array([]), np.array([])

        print(
            f"Predicting on {len(eu_chains_np)} EU flight chains using {self.model.__class__.__name__}..."
        )
        all_predictions_indices, all_probabilities_np = [], []
        batch_size = 64  # Consider making this configurable
        for i in tqdm(
            range(0, len(eu_chains_np), batch_size),
            desc="Batch Prediction",
            unit="batch",
            ncols=100,  # Nicer progress bar width
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

        if (
            not all_predictions_indices
        ):  # Should be redundant if len(eu_chains_np) > 0 check passed
            return np.array([]), np.array([])

        return np.array(all_predictions_indices), np.concatenate(
            all_probabilities_np, axis=0
        )


def run_eu_prediction():
    print("--- Starting EU Flight Chain Prediction ---")
    print(f"Using model type: {EU_PREDICTION_MODEL_TYPE.upper()}")
    # Also print the status of the force flag for clarity
    print(f"Force BiLSTM override active in config: {FORCE_LSTM_BIDIR_FOR_EU}")

    if not EU_TEST_CHAINS_FILE.exists():
        print(
            f"Processed EU chains file {EU_TEST_CHAINS_FILE} not found. Run 'eu_chain_constructor.py' first."
        )
        sys.exit(1)

    try:
        predictor = EUPredictor()
    except Exception as e:
        print(f"FATAL: Failed to initialize EUPredictor: {e}")
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
    for i in range(min(5, len(predicted_statuses))):  # Print first 5 samples
        print(
            f"  Chain {i}: Predicted CatIdx: {predicted_category_indices[i]}, Status: '{predicted_statuses[i]}', Probs: {np.round(predicted_probabilities[i],3)}"
        )

    # Load true labels if available for evaluation context
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
        # Construct filenames using the model type
        output_pred_indices_path = (
            PROCESSED_EU_CHAINS_DIR
            / f"eu_predicted_indices_{EU_PREDICTION_MODEL_TYPE}.npy"
        )
        np.save(output_pred_indices_path, predicted_indices)
        print(f"Predicted indices saved to {output_pred_indices_path}")

        if probabilities is not None and len(probabilities) > 0:
            output_pred_probs_path = (
                PROCESSED_EU_CHAINS_DIR
                / f"eu_predicted_probabilities_{EU_PREDICTION_MODEL_TYPE}.npy"
            )
            np.save(output_pred_probs_path, probabilities)
            print(f"Predicted probabilities saved to {output_pred_probs_path}")
        else:
            print("Predicted probabilities were not generated or returned.")

        print(
            "You can now run 'eu_ev.py' (or 'eu_evaluate_predictions.py') if true labels are available and were loaded."
        )
    elif predicted_indices is not None and len(predicted_indices) == 0:
        print("Prediction ran, but no chains resulted in predictions.")
    else:
        print("Prediction script did not produce valid output.")
