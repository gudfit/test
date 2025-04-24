# flightChainClassifier/src/evaluation/evaluate.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
import os
import sys
import re
from tqdm import tqdm

# -------------------------------------------------------------------------
#  Conditional project‑relative imports
# -------------------------------------------------------------------------
try:
    # Absolute imports when package is on PYTHONPATH
    from .. import config
    from ..training.dataset import FlightChainDataset
    from ..modeling.flight_chain_models import (
        CBAM_CNN_Model,
        SimAM_CNN_LSTM_Model,
    )
    from ..modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model  # NEW
except ImportError:
    # Fallback when running the script directly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)        # evaluation/ -> src/
    project_dir = os.path.dirname(src_dir)       # src/ -> flightChainClassifier
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from src import config
    from src.training.dataset import FlightChainDataset
    from src.modeling.flight_chain_models import (
        CBAM_CNN_Model,
        SimAM_CNN_LSTM_Model,
    )
    from src.modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model  # NEW


@torch.no_grad()  # Disable gradient calculations during evaluation
def run_evaluation(model_type: str = "simam") -> None:
    """
    Evaluate a trained model on the test split.

    Args:
        model_type:  One of {"cbam", "simam", "qtsimam"}.
    """
    print(f"--- Starting Evaluation for {model_type.upper()} model ---")
    device = config.DEVICE

    # --------------------------------------------------------------
    # 1. Load meta‑data
    # --------------------------------------------------------------
    data_stats = config.load_data_stats()
    if data_stats is None or "num_features" not in data_stats:
        print("Error: Could not load data stats or 'num_features' missing.")
        sys.exit(1)
    num_features = data_stats["num_features"]

    # --------------------------------------------------------------
    # 2. Load test set
    # --------------------------------------------------------------
    try:
        test_dataset = FlightChainDataset(
            config.TEST_CHAINS_FILE, config.TEST_LABELS_FILE
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
    except FileNotFoundError:
        print(
            "Error: test .npy files not found. Run chain_constructor first."
        )
        sys.exit(1)

    # --------------------------------------------------------------
    # 3. Build model skeleton
    # --------------------------------------------------------------
    if not config.MODEL_SAVE_PATH.exists():
        print(
            f"Error: model file not found at {config.MODEL_SAVE_PATH}. "
            "Run training first."
        )
        sys.exit(1)

    print(f"Loading best model state from {config.MODEL_SAVE_PATH}...")

    best_params = config.load_best_hyperparameters()
    if best_params is not None:
        print("Best hyperparameters loaded successfully.")
    else:
        print("No best hyperparameters found; using defaults.")

    # —―― determine how many LSTM layers were saved (for simam/qtsimam) ―――
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    lstm_layers_in_state = 1
    for k in state_dict:
        m = re.match(r"lstm\.lstm\.weight_ih_l(\d+)", k)
        if m:
            lstm_layers_in_state = max(lstm_layers_in_state, int(m.group(1)) + 1)
    print(f"Using {lstm_layers_in_state} LSTM layer(s).")

    # —―― instantiate requested architecture ―――
    if model_type == "cbam":
        model = CBAM_CNN_Model(
            num_features=num_features, num_classes=config.NUM_CLASSES
        )
    elif model_type == "simam":
        model = SimAM_CNN_LSTM_Model(
            num_features=num_features,
            num_classes=config.NUM_CLASSES,
            lstm_hidden=best_params.get(
                "lstm_hidden_size", config.DEFAULT_LSTM_HIDDEN_SIZE
            )
            if best_params
            else config.DEFAULT_LSTM_HIDDEN_SIZE,
            lstm_layers=lstm_layers_in_state,
            dropout_rate=best_params.get(
                "dropout_rate", config.DEFAULT_DROPOUT_RATE
            )
            if best_params
            else config.DEFAULT_DROPOUT_RATE,
        )
    elif model_type == "qtsimam":  # ← NEW CASE
        model = QTSimAM_CNN_LSTM_Model(
            num_features=num_features,
            num_classes=config.NUM_CLASSES,
            lstm_hidden=best_params.get(
                "lstm_hidden_size", config.DEFAULT_LSTM_HIDDEN_SIZE
            )
            if best_params
            else config.DEFAULT_LSTM_HIDDEN_SIZE,
            lstm_layers=lstm_layers_in_state,
            dropout_rate=best_params.get(
                "dropout_rate", config.DEFAULT_DROPOUT_RATE
            )
            if best_params
            else config.DEFAULT_DROPOUT_RATE,
        )
    else:
        print(f"Error: Unknown model type '{model_type}'.")
        sys.exit(1)

    # --------------------------------------------------------------
    # 4. Load weights & switch to eval
    # --------------------------------------------------------------
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --------------------------------------------------------------
    # 5. Inference loop
    # --------------------------------------------------------------
    y_pred, y_true = [], []
    for chains, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        chains = chains.to(device)
        outputs = model(chains)
        y_pred.extend(torch.argmax(outputs, 1).cpu().numpy())
        y_true.extend(labels.numpy())

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # --------------------------------------------------------------
    # 6. Metrics & artefacts
    # --------------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Save JSON metrics
    metrics_path = config.EVAL_DIR / f"{model_type}_metrics.json"
    metrics = {"model_type": model_type, "accuracy": acc, "report": report}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Save confusion‑matrix figure
    fig_path = config.PLOTS_DIR / f"{model_type}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=[f"C{i}" for i in range(config.NUM_CLASSES)]).plot(
        ax=ax, cmap=plt.cm.Blues, values_format="d"
    )
    plt.title(f"Confusion Matrix – {model_type.upper()}")
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {fig_path}")

    print(f"--- Evaluation Finished for {model_type.upper()} model ---")


if __name__ == "__main__":
    # Allow quick CLI testing, e.g.  `python -m src.evaluation.evaluate qtsimam`
    mt = sys.argv[1] if len(sys.argv) > 1 else "simam"
    run_evaluation(model_type=mt)

