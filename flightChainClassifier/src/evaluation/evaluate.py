# flightChainClassifier/src/evaluation/evaluate.py
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# -------- project imports with fallback -----------------------------------
try:
    from .. import config
    from ..training.dataset import FlightChainDataset
    from ..modeling.flight_chain_models import CBAM_CNN_Model, SimAM_CNN_LSTM_Model
    from ..modeling.queue_augment_models import (
        QTSimAM_CNN_LSTM_Model,
        QTSimAM_MaxPlus_Model,
    )
except ImportError:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from src import config
    from src.training.dataset import FlightChainDataset
    from src.modeling.flight_chain_models import CBAM_CNN_Model, SimAM_CNN_LSTM_Model


# -------- helper: infer hidden/layers from state-dict ----------------------
def _infer_from_state_dict(sd: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Return (hidden_size, num_layers) by inspecting LSTM/QMogrifier weights."""
    lstm_keys = [k for k in sd if "lstm_stack.layers" in k and ".weight_ih" in k]
    num_layers = len(lstm_keys)
    hidden_size = sd[lstm_keys[0]].shape[0] // 4  # weight_ih: (4*H, input)
    return hidden_size, num_layers


@torch.no_grad()
def run_evaluation(
    *,
    model_type: str = "simam",
    lstm_layers: int | None = None,
    lstm_hidden_size: int | None = None,
) -> None:
    print(f"--- Starting Evaluation for {model_type.upper()} ---")
    device = config.DEVICE

    # ---------- data ------------------------------------------------------
    stats = config.load_data_stats()
    if stats is None or "num_features" not in stats:
        print("❌ data_stats.json missing – run chain_constructor first.")
        sys.exit(1)
    num_features = stats["num_features"]

    test_ds = FlightChainDataset(config.TEST_CHAINS_FILE, config.TEST_LABELS_FILE)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # ---------- checkpoint ------------------------------------------------
    ckpt_path = config.MODEL_SAVE_PATH
    if not ckpt_path.exists():
        print(f"❌ checkpoint not found at {ckpt_path}")
        sys.exit(1)

    # ---------- gather arch hints ----------------------------------------
    arch_meta: dict[str, Any] = {}
    arch_best: dict[str, Any] = {}

    # 1) best-params (lower priority)
    best_file = (
        config.BEST_PARAMS_FILE.with_name("best_hyperparameters_qtsimam.json")
        if model_type == "qtsimam"
        else config.BEST_PARAMS_FILE
    )
    if best_file.exists():
        arch_best = json.loads(best_file.read_text())
        print(f"⤷ Loaded arch params from {best_file.name}")

    # 2) side-car meta (higher priority)
    meta_path = ckpt_path.with_suffix(".meta.json")
    if meta_path.exists():
        arch_meta = json.loads(meta_path.read_text())
        print(f"⤷ Loaded arch params from {meta_path.name}")

    def resolve(key, cli_val, default):
        if cli_val is not None:  # CLI wins
            return cli_val
        if key in arch_meta:  # then meta
            return arch_meta[key]
        if key in arch_best:  # then best-params
            return arch_best[key]
        return default  # else default/None

    hidden = resolve("lstm_hidden_size", lstm_hidden_size, None)
    layers = resolve("lstm_num_layers", lstm_layers, None)

    # 3) tensor inference if still missing
    if hidden is None or layers is None:
        print("⤷ No meta/CLI value – inferring from checkpoint tensors")
        sd_cpu = torch.load(ckpt_path, map_location="cpu")
        inferred_hidden, inferred_layers = _infer_from_state_dict(sd_cpu)
        hidden = hidden or inferred_hidden
        layers = layers or inferred_layers

    # fallback to config defaults (should rarely happen)
    if hidden is None:
        hidden = config.DEFAULT_LSTM_HIDDEN_SIZE
    if layers is None:
        layers = config.DEFAULT_LSTM_NUM_LAYERS

    print(f"Using hidden={hidden}, layers={layers}")

    # ---------- build model ----------------------------------------------
    if model_type == "cbam":
        model = CBAM_CNN_Model(num_features, config.NUM_CLASSES)
    elif model_type == "simam":
        model = SimAM_CNN_LSTM_Model(
            num_features, config.NUM_CLASSES, lstm_hidden=hidden, lstm_layers=layers
        )
    elif model_type == "qtsimam":
        model = QTSimAM_CNN_LSTM_Model(
            num_features, config.NUM_CLASSES, lstm_hidden=hidden, lstm_layers=layers
        )
    elif model_type == "qtsimam_mp":
        model = QTSimAM_CNN_LSTM_Model(
            num_features + 1, config.NUM_CLASSES, lstm_hidden=hidden, lstm_layers=layers
        )
    else:
        print(f"Unknown model_type '{model_type}'.")
        sys.exit(1)

    # strict=True detects mismatch immediately
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.to(device).eval()

    # ---------- inference -------------------------------------------------
    preds, labels = [], []
    for x, y in test_loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    # ---------- metrics ---------------------------------------------------
    acc = accuracy_score(labels, preds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        report = classification_report(labels, preds, digits=4, zero_division=0)
    cm = confusion_matrix(labels, preds)

    print(f"\nAccuracy: {acc:.4f}\n")
    print(report)

    metrics_path = config.EVAL_DIR / f"{model_type}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"accuracy": acc, "report": report}, indent=4))

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    plt.title(f"{model_type.upper()} – Confusion matrix")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / f"{model_type}_confusion_matrix.png")
    plt.close(fig)
    print("Evaluation finished – metrics & plot saved.")


# ---------------- CLI wrapper ---------------------------------------------
if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser()
    cli.add_argument("--model", default="simam", choices=["cbam", "simam", "qtsimam"])
    cli.add_argument(
        "--lstm-layers",
        type=int,
        default=None,
        help="Override number of LSTM/QMogrifier layers",
    )
    cli.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=None,
        help="Override hidden size of LSTM/QMogrifier layers",
    )
    args = cli.parse_args()
    run_evaluation(
        model_type=args.model,
        lstm_layers=args.lstm_layers,
        lstm_hidden_size=args.lstm_hidden_size,
    )
