# flightChainClassifier/src/training/trainer.py

from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .. import config
    from .dataset import FlightChainDataset, FlightChainDatasetSim
    from ..modeling.flight_chain_models import CBAM_CNN_Model, SimAM_CNN_LSTM_Model
except ImportError:
    # fallback when run as top-level module
    proj = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(proj))
    from src import config
    from src.training.dataset import FlightChainDataset, FlightChainDatasetSim
    from src.modeling.flight_chain_models import CBAM_CNN_Model, SimAM_CNN_LSTM_Model


def train_epoch(model, loader, criterion, optim_, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Training", leave=False, ncols=100, unit="batch"):
        x, y = x.to(device), y.to(device)
        optim_.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if not torch.isfinite(loss):  # skip weird batches
            continue
        loss.backward()
        optim_.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Validation", leave=False, ncols=100, unit="batch"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        if not torch.isfinite(loss):
            continue
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def run_training(
    *,
    model_type: str = "simam",
    epochs: int | None = None,
    batch_size: int | None = None,
    lstm_layers: int | None = None,
    lstm_hidden_size: int | None = None,
    hyperparams: dict | None = None,
):
    """Train the requested architecture."""
    device = config.DEVICE
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE

    stats = config.load_data_stats()
    if not stats:
        print("❌  data stats missing – run chain constructor first")
        sys.exit(1)
    num_feat = stats["num_features"]

    if config.USE_SIM_AUG:
        Dataset = FlightChainDatasetSim
        tr_kwargs = {"sim_factor": config.SIM_FACTOR, "data_stats": stats}
        va_kwargs = {"sim_factor": 1, "data_stats": stats}
        print(f"Using simulated dataset (SIM_FACTOR={config.SIM_FACTOR})")
    else:
        Dataset = FlightChainDataset
        tr_kwargs = va_kwargs = {}
        print("Using plain dataset (no augmentation)")

    train_ds = Dataset(config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE, **tr_kwargs)
    val_ds = Dataset(config.VAL_CHAINS_FILE, config.VAL_LABELS_FILE, **va_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    lstm_layers = lstm_layers or (hyperparams or {}).get("lstm_num_layers", 1)
    lstm_hidden_size = lstm_hidden_size or (hyperparams or {}).get(
        "lstm_hidden_size", 128
    )

    lr = (hyperparams or {}).get("lr", config.DEFAULT_LEARNING_RATE)
    weight_decay = (hyperparams or {}).get("weight_decay", config.DEFAULT_WEIGHT_DECAY)
    dropout = (hyperparams or {}).get("dropout_rate", config.DEFAULT_DROPOUT_RATE)

    base_labels = getattr(train_ds, "base_labels", train_ds.labels)
    cls_weights = compute_class_weight(
        "balanced", classes=np.arange(config.NUM_CLASSES), y=base_labels
    )
    weight_tensor = torch.tensor(cls_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    if model_type.lower() == "cbam":
        model = CBAM_CNN_Model(num_feat, config.NUM_CLASSES, dropout_rate=dropout)
    elif model_type.lower() == "simam":
        model = SimAM_CNN_LSTM_Model(
            num_feat,
            config.NUM_CLASSES,
            lstm_hidden=lstm_hidden_size,
            lstm_layers=lstm_layers,
            dropout_rate=dropout,
        )
    elif model_type.lower() == "qtsimam":
        from ..modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model

        model = QTSimAM_CNN_LSTM_Model(
            num_feat,
            config.NUM_CLASSES,
            lstm_hidden=lstm_hidden_size,
            lstm_layers=lstm_layers,
            dropout_rate=dropout,
        )
    else:
        print(f"Unknown model type {model_type}")
        sys.exit(1)

    model = model.to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        optim_, "min", factor=0.2, patience=3, min_lr=1e-7
    )

    best_val = float("inf")
    meta = {
        "lstm_hidden_size": lstm_hidden_size,
        "lstm_num_layers": lstm_layers,
        "dropout": dropout,
        "sim_factor": config.SIM_FACTOR if config.USE_SIM_AUG else 1,
    }

    print(
        f"🏋️  hidden={lstm_hidden_size}  layers={lstm_layers}  "
        f"bs={batch_size}  epochs={epochs}"
    )
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optim_, device)
        va_loss, va_acc = validate_epoch(model, val_loader, criterion, device)
        sched.step(va_loss)
        dur = time.time() - t0
        print(
            f"Epoch {ep:02d}/{epochs} | {dur:5.1f}s "
            f"| Train {tr_loss:.4f}/{tr_acc:.3f}"
            f" | Val {va_loss:.4f}/{va_acc:.3f}",
            flush=True,
        )
        # save best
        if va_loss < best_val:
            best_val = va_loss
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            with open(config.MODEL_SAVE_PATH.with_suffix(".meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print("  ▲ best model + meta saved")

    print(f"Finished.  Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    run_training(model_type="qtsimam", epochs=1, batch_size=64)
