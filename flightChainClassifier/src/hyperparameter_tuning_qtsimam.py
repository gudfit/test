# flightChainClassifier/src/hyperparameter_tuning_qtsimam.py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import time
import json
from sklearn.utils.class_weight import compute_class_weight
import traceback
import argparse  # NEW: For parsing command-line arguments

# --- Path Setup & Imports ---
try:
    from src import config
    from src.training.dataset import FlightChainDataset
    from src.modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model
    from src.training.trainer import validate_epoch
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    try:
        from src import config
        from src.training.dataset import FlightChainDataset
        from src.modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model
        from src.training.trainer import validate_epoch
    except ImportError as e:
        print(
            f"CRITICAL: Error importing modules in hyperparameter_tuning_qtsimam.py: {e}"
        )
        sys.exit(1)

# NEW: Command-line argument parsing for balanced flag.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning for QTSimAM CNN-LSTM Model"
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced training data by oversampling",
    )
    args = parser.parse_args()
    config.BALANCED = args.balanced
    print(
        f"Hyperparameter Tuning (QTSimAM): Using balanced training data? {config.BALANCED}"
    )


def objective(trial: optuna.Trial) -> float:
    print(f"\n--- Starting Optuna Trial {trial.number} for QTSimAM model ---")
    device = config.DEVICE
    try:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
        lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128])
        lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 2)
    except Exception as e:
        print("Error during hyperparameter suggestion:", e)
        return float("inf")
    print(
        f"Trial {trial.number} Parameters: lr={lr:.1E}, weight_decay={weight_decay:.1E}, dropout_rate={dropout_rate:.2f}, lstm_hidden_size={lstm_hidden_size}, lstm_num_layers={lstm_num_layers}"
    )
    try:
        data_stats = config.load_data_stats()
        if data_stats is None or "num_features" not in data_stats:
            print("Error: Data stats missing!")
            return float("inf")
        num_features = data_stats["num_features"]
        train_dataset = FlightChainDataset(
            config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE
        )
        val_dataset = FlightChainDataset(config.VAL_CHAINS_FILE, config.VAL_LABELS_FILE)
        class_weights_tensor = None
        unique_classes_train, counts_train = np.unique(
            train_dataset.labels, return_counts=True
        )
        if len(unique_classes_train) > 1:
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(config.NUM_CLASSES),
                y=train_dataset.labels,
            )
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
                device
            )
        else:
            print("Warning: Only one class in training set.")
        batch_size = config.BATCH_SIZE
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
    except Exception as e:
        print("Error loading datasets:", e)
        traceback.print_exc()
        return float("inf")
    try:
        model = QTSimAM_CNN_LSTM_Model(
            num_features=num_features,
            num_classes=config.NUM_CLASSES,
            lstm_hidden=lstm_hidden_size,
            lstm_layers=lstm_num_layers,
            dropout_rate=dropout_rate,
        ).to(device)
    except Exception as e:
        print("Error initializing QTSimAM model:", e)
        traceback.print_exc()
        return float("inf")
    if class_weights_tensor is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_trial_val_loss = float("inf")
    epochs_for_tuning = config.EPOCHS
    for epoch in range(epochs_for_tuning):
        model.train()
        total_train_loss = 0.0
        train_samples = 0
        for chains, labels in train_loader:
            chains, labels = chains.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(chains)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * chains.size(0)
            train_samples += labels.size(0)
        avg_train_loss = (
            total_train_loss / train_samples if train_samples > 0 else float("inf")
        )
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}/{epochs_for_tuning}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        best_trial_val_loss = min(best_trial_val_loss, val_loss)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch+1}")
            return (
                best_trial_val_loss
                if np.isfinite(best_trial_val_loss)
                else float("inf")
            )
        if not np.isfinite(val_loss) or val_loss > 100.0:
            print(f"Trial {trial.number} aborted due to unstable val loss: {val_loss}")
            return float("inf")
    print(
        f"Trial {trial.number} finished with Best Val Loss: {best_trial_val_loss:.4f}"
    )
    return best_trial_val_loss


if __name__ == "__main__":
    study_name = "qtsimam_tuning_v1"
    n_trials = 20
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3, n_min_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, timeout=3600, n_jobs=1)
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    fail_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(
        f"\nStudy Summary: Total Trials: {len(study.trials)}, Complete: {len(complete_trials)}, Pruned: {len(pruned_trials)}, Failed: {len(fail_trials)}"
    )
    if complete_trials:
        best_trial = study.best_trial
        print(
            f"\nBest Trial: Number: {best_trial.number}, Value (Loss): {best_trial.value:.6f}"
        )
        print("Parameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        try:
            config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            best_params_file = config.BEST_PARAMS_FILE.with_name(
                "best_hyperparameters_qtsimam.json"
            )
            with open(best_params_file, "w") as f:
                json.dump(best_trial.params, f, indent=4)
            print(f"Best hyperparameters saved to {best_params_file}")
        except Exception as e:
            print(f"Error saving best hyperparameters: {e}")
    else:
        print("No trials completed successfully.")
