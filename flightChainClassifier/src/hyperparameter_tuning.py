# flightChainClassifier/src/hyperparameter_tuning.py
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
    # Use absolute imports assuming project root is in path
    from src import config
    from src.training.dataset import FlightChainDataset  # Absolute
    from src.modeling.flight_chain_models import SimAM_CNN_LSTM_Model  # Absolute
    from src.training.trainer import validate_epoch  # Absolute
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = script_dir  # Assumes this script is in src/
    project_dir = os.path.dirname(src_dir)
    if project_dir not in sys.path: sys.path.insert(0, project_dir)
    try:
        from src import config
        from src.training.dataset import FlightChainDataset
        from src.modeling.flight_chain_models import SimAM_CNN_LSTM_Model
        from src.training.trainer import validate_epoch
    except ImportError as e:
         print(f"CRITICAL: Error importing modules in hyperparameter_tuning.py: {e}")
         traceback.print_exc()
         sys.exit(1)

# NEW: Command-line argument parsing for balanced flag.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for SimAM CNN-LSTM Model")
    parser.add_argument("--balanced", action="store_true", help="Use balanced training data by oversampling")
    args = parser.parse_args()
    config.BALANCED = args.balanced
    print(f"Hyperparameter Tuning: Using balanced training data? {config.BALANCED}")

# --- Objective Function (objective) ---
def objective(trial: optuna.Trial):
    print(f"\n--- Starting Optuna Trial {trial.number} ---")
    device = config.DEVICE
    # --- Suggest Hyperparameters ---
    try:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
        lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 2)
        batch_size = config.BATCH_SIZE  # Keep fixed
        optimizer_name = "Adam"
    except Exception as e:
        return float('inf')
    print(f"Trial {trial.number} Parameters: LR={lr:.1E}, WD={weight_decay:.1E}, Dropout={dropout_rate:.2f}, LSTMHidden={lstm_hidden_size}, LSTMLayers={lstm_num_layers}")
    # --- Load Data & Weights ---
    try:
        data_stats = config.load_data_stats()
        num_features = data_stats['num_features']
        train_dataset = FlightChainDataset(config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE)
        val_dataset = FlightChainDataset(config.VAL_CHAINS_FILE, config.VAL_LABELS_FILE)
        class_weights_tensor = None
        unique_classes_train = np.unique(train_dataset.labels)
        if len(unique_classes_train) > 1:
             try:
                 class_weights = compute_class_weight('balanced', classes=np.arange(config.NUM_CLASSES), y=train_dataset.labels)
                 class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
             except ValueError:
                 print("  Warn: Could not compute weights.");
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    except Exception as e:
        print(f"Error loading data: {e}")
        return float('inf')
    # --- Initialize Model ---
    try:
        model = SimAM_CNN_LSTM_Model(num_features, config.NUM_CLASSES, lstm_hidden=lstm_hidden_size, lstm_layers=lstm_num_layers, dropout_rate=dropout_rate).to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return float('inf')
    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_trial_val_loss = float('inf')
    epochs_for_tuning = config.EPOCHS
    for epoch in range(epochs_for_tuning):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0
        train_samples = 0
        for chains, labels in train_loader:
            chains, labels = chains.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(chains)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * chains.size(0)
            train_samples += labels.size(0)
        avg_train_loss = total_train_loss / train_samples if train_samples > 0 else 0
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        epoch_duration = time.time() - epoch_start_time
        best_trial_val_loss = min(best_trial_val_loss, val_loss)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned at epoch {epoch+1}.")
            return best_trial_val_loss if np.isfinite(best_trial_val_loss) else float('inf')
        if not np.isfinite(val_loss) or val_loss > 100.0:
            print(f"  Trial {trial.number} aborted: unstable val loss ({val_loss}).")
            return float('inf')
    print(f"--- Trial {trial.number} Finished. Best Val Loss: {best_trial_val_loss:.4f} ---")
    return best_trial_val_loss

# --- Main Tuning Execution Block ---
if __name__ == "__main__":
    print("Checking for processed data files...")
    required_files = [config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE,
                      config.VAL_CHAINS_FILE, config.VAL_LABELS_FILE, config.DATA_STATS_FILE]
    if not all(f.exists() for f in required_files):
        print("Error: Processed data files (.npy, .json) not found.")
        sys.exit(1)
    else:
        print("Processed data files found.")
    print("\n--- Starting Hyperparameter Tuning with Optuna ---")
    study_name = "simam-cnn-lstm-tuning-v1"
    n_trials = 20
    study_timeout_hours = 2
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=3, n_min_trials=5))
    study_start_time = time.time()
    try:
        study.optimize(objective, n_trials=n_trials, timeout=study_timeout_hours * 3600, n_jobs=1)
    except KeyboardInterrupt:
        print("\n--- Tuning stopped manually ---")
    except Exception as e:
        print(f"\n--- Error during tuning: {e} ---")
        traceback.print_exc()
    finally:
        study_duration = time.time() - study_start_time
        print(f"\n--- Optuna Study Finished (Duration: {study_duration:.2f}s) ---")
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    fail_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"\nStudy Summary: Total={len(study.trials)}, Complete={len(complete_trials)}, Pruned={len(pruned_trials)}, Failed={len(fail_trials)}")
    if complete_trials:
        best_trial = study.best_trial
        print(f"\nBest trial completed: Number: {best_trial.number}, Value (Loss): {best_trial.value:.6f}")
        print("Parameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print(f"\nSaving best hyperparameters to {config.BEST_PARAMS_FILE}")
        try:
            config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.BEST_PARAMS_FILE, 'w') as f:
                json.dump(best_trial.params, f, indent=4)
            print("Best parameters saved successfully.")
        except Exception as e:
            print(f"Error saving best parameters: {e}")
    else:
        print("\nNo trials completed successfully.")
    if complete_trials or pruned_trials:
        try:
            if optuna.visualization.is_available():
                print("\nGenerating Optuna visualization plots...")
                config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
                fig1 = optuna.visualization.plot_optimization_history(study)
                fig1.write_image(config.PLOTS_DIR / f"{study_name}_optuna_history.png")
                if len(complete_trials) >= 1:
                    fig2 = optuna.visualization.plot_param_importances(study)
                    fig2.write_image(config.PLOTS_DIR / f"{study_name}_optuna_param_importance.png")
                print(f"Optuna plots saved to {config.PLOTS_DIR}")
            else:
                print("\nInstall plotly for Optuna plots: `pip install plotly`")
        except ImportError:
            print("\nPlotly not found. Skipping plots.")
        except Exception as e:
            print(f"\nError generating Optuna plots: {e}")
    else:
        print("\nSkipping plot generation (no completed/pruned trials).")
    print("\n--- Hyperparameter Tuning Script Finished ---")
