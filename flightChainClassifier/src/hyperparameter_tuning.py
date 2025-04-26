from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader


import plotly.io as pio

try:
    from src import config
    from src.modeling.flight_chain_models import SimAM_CNN_LSTM_Model
    from src.training.dataset import FlightChainDataset
    from src.training.trainer import validate_epoch
except ImportError:
    logging.exception("Failed to import project modules — check PYTHONPATH.")
    raise

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Optuna hyper‑parameter search")
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Oversample to balance class distribution in the training set.",
    )
    return parser.parse_args()


def _assert_required_files() -> None:
    """Exit if any of the pre‑processed artefacts is missing."""
    required: List[Path] = [
        config.TRAIN_CHAINS_FILE,
        config.TRAIN_LABELS_FILE,
        config.VAL_CHAINS_FILE,
        config.VAL_LABELS_FILE,
        config.DATA_STATS_FILE,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.error("Missing required data files:\n%s", "\n".join(missing))
        sys.exit(1)
    logger.info("All required data files located.")


def _load_datasets(
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor], int]:
    """Return train/val data‑loaders, class‑weights tensor and num features."""
    data_stats: Optional[Dict[str, Any]] = config.load_data_stats()
    if not data_stats or "num_features" not in data_stats:
        raise RuntimeError(
            "`num_features` missing in data_stats — run preprocessing first."
        )

    num_features: int = int(data_stats["num_features"])

    train_ds = FlightChainDataset(config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE)
    val_ds = FlightChainDataset(config.VAL_CHAINS_FILE, config.VAL_LABELS_FILE)

    class_weights_tensor: Optional[torch.Tensor] = None
    if np.unique(train_ds.labels).size > 1:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(config.NUM_CLASSES),
            y=train_ds.labels,
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
            config.DEVICE
        )

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
    return train_loader, val_loader, class_weights_tensor, num_features


def _objective(trial: optuna.Trial) -> float:
    """Optuna objective that trains the model and returns validation loss."""
    logger.info("Starting trial %d", trial.number)

    # Suggested hyper‑parameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 2)

    batch_size = config.BATCH_SIZE
    device = config.DEVICE

    try:
        train_loader, val_loader, class_weights, num_features = _load_datasets(
            batch_size
        )
    except Exception as exc:
        logger.exception("Data loading failed: %s", exc)
        return float("inf")

    model = SimAM_CNN_LSTM_Model(
        num_features=num_features,
        num_classes=config.NUM_CLASSES,
        lstm_hidden=lstm_hidden_size,
        lstm_layers=lstm_num_layers,
        dropout_rate=dropout_rate,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        samples = 0

        for chains, labels in train_loader:
            chains, labels = chains.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(chains)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                continue  # skip NaNs silently

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            samples += labels.size(0)

        avg_train_loss = running_loss / samples if samples else 0.0
        val_loss, _ = validate_epoch(model, val_loader, criterion, device)
        best_val_loss = min(best_val_loss, val_loss)

        logger.info(
            "Epoch %02d | train %.4f | val %.4f | best %.4f",
            epoch + 1,
            avg_train_loss,
            val_loss,
            best_val_loss,
        )

        trial.report(val_loss, epoch)
        if trial.should_prune() or not np.isfinite(val_loss) or val_loss > 100.0:
            logger.warning(
                "Trial %d pruned/aborted at epoch %d", trial.number, epoch + 1
            )
            raise optuna.TrialPruned()

    return best_val_loss


def _summarise_study(study: optuna.Study) -> None:
    """Log a brief summary of trial outcomes."""
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    failed = sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)

    logger.info(
        "Trials — total: %d | complete: %d | pruned: %d | failed: %d",
        len(study.trials),
        complete,
        pruned,
        failed,
    )

    if complete:
        best = study.best_trial
        logger.info("Best trial #%d — loss: %.6f", best.number, best.value)
        for k, v in best.params.items():
            logger.info("  %s: %s", k, v)


def _export_results(study: optuna.Study, study_name: str) -> None:
    """Save best parameters and (optionally) visualisations to disk."""
    if not study.trials:
        return

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.BEST_PARAMS_FILE, "w", encoding="utf-8") as fp:
        json.dump(study.best_trial.params, fp, indent=4)
    logger.info("Best hyper‑parameters saved to %s", config.BEST_PARAMS_FILE)

    try:
        if optuna.visualization.is_available():
            config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

            pio.write_image(
                optuna.visualization.plot_optimization_history(study),
                config.PLOTS_DIR / f"{study_name}_history.png",
            )
            if len(study.trials) >= 1:
                pio.write_image(
                    optuna.visualization.plot_param_importances(study),
                    config.PLOTS_DIR / f"{study_name}_param_importance.png",
                )
            logger.info("Optuna figures saved to %s", config.PLOTS_DIR)
        else:
            logger.info(
                "Install plotly for Optuna visualisations: `pip install plotly`"
            )
    except Exception:
        logger.exception("Failed to write Optuna figures.")


def main() -> None:
    args = _parse_args()
    config.BALANCED = args.balanced
    logger.info("Using balanced oversampling? %s", config.BALANCED)

    _assert_required_files()

    study_name = "simam-cnn-lstm-tuning-v1"
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3, n_min_trials=5),
    )

    try:
        study.optimize(
            _objective,
            n_trials=20,
            timeout=2 * 3600,  # 2 h
            n_jobs=1,
        )
    except KeyboardInterrupt:
        logger.warning("Tuning interrupted by user.")
    finally:
        _summarise_study(study)
        _export_results(study, study_name)


if __name__ == "__main__":
    main()
