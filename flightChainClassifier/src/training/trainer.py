# flightChainClassifier/src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import sys
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import traceback

# --- Path Setup & Imports ---
try:
    from src import config
    from src.training.dataset import FlightChainDataset
    from src.modeling.flight_chain_models import CBAM_CNN_Model, SimAM_CNN_LSTM_Model
except ImportError:
    # Fallback if run directly or path is not set correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(src_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    try:
        from src import config
        from src.training.dataset import FlightChainDataset
        from src.modeling.flight_chain_models import CBAM_CNN_Model, SimAM_CNN_LSTM_Model
    except ImportError as e:
         print(f"CRITICAL: Error importing modules in trainer.py: {e}")
         traceback.print_exc()
         sys.exit(1)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False, ncols=100, unit="batch")
    for i, (chains, labels) in enumerate(progress_bar):
        chains, labels = chains.to(device), labels.to(device)
        try:
            outputs = model(chains)
            loss = criterion(outputs, labels)
        except Exception as e:
            print(f"\nError during forward/loss calculation at batch {i}: {e}")
            continue

        if not torch.isfinite(loss):
            print(f"\nWarning: Non-finite loss ({loss.item()}) detected at batch {i}. Skipping batch.")
            continue

        optimizer.zero_grad()
        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"\nError during backward/optimizer step at batch {i}: {e}")
            continue

        total_loss += loss.item() * chains.size(0)
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False, ncols=100, unit="batch")
    with torch.no_grad():
        for chains, labels in progress_bar:
            chains, labels = chains.to(device), labels.to(device)
            try:
                outputs = model(chains)
                loss = criterion(outputs, labels)
            except Exception as e:
                print(f"\nError during validation forward/loss calculation: {e}")
                continue

            if not torch.isfinite(loss):
                print(f"\nWarning: Non-finite validation loss ({loss.item()}) detected. Skipping batch result.")
                continue

            total_loss += loss.item() * chains.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def run_training(model_type='simam', hyperparams=None):
    print(f"--- Starting Training for {model_type.upper()} model ---")
    if hyperparams:
        print("Using Provided Hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
    else:
        print("Using Default Hyperparameters from config.")

    device = config.DEVICE

    # --- Load Data Statistics ---
    print("Loading data statistics...")
    data_stats = config.load_data_stats()
    if data_stats is None or 'num_features' not in data_stats:
        print("Error: Could not load data stats or 'num_features' missing.")
        sys.exit(1)
    num_features = data_stats['num_features']
    print(f"Number of features per flight step: {num_features}")

    # --- Create Datasets and DataLoaders ---
    try:
        print("Creating datasets...")
        train_dataset = FlightChainDataset(config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE)
        val_dataset = FlightChainDataset(config.VAL_CHAINS_FILE, config.VAL_LABELS_FILE)
        print("Calculating class weights...")
        class_weights_tensor = None
        unique_classes_train, counts_train = np.unique(train_dataset.labels, return_counts=True)
        print(f"Training label distribution: {dict(zip(unique_classes_train, counts_train))}")

        if len(unique_classes_train) > 1:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.arange(config.NUM_CLASSES),
                y=train_dataset.labels
            )
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            print(f"Using Class Weights: {class_weights_tensor.cpu().numpy()}")
        else:
            print("Warning: Only one class found in training data. Cannot compute balanced weights.")

        print("Creating dataloaders...")
        batch_size = hyperparams.get("batch_size", config.BATCH_SIZE) if hyperparams else config.BATCH_SIZE
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
        )
    except FileNotFoundError:
        print("Error: Processed data files (.npy) not found.")
        print("Please run data processing (chain_constructor.py) first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating datasets/dataloaders: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Extract Hyperparameters ---
    lr = hyperparams.get("lr", config.DEFAULT_LEARNING_RATE) if hyperparams else config.DEFAULT_LEARNING_RATE
    weight_decay = hyperparams.get("weight_decay", config.DEFAULT_WEIGHT_DECAY) if hyperparams else config.DEFAULT_WEIGHT_DECAY
    dropout_rate = hyperparams.get("dropout_rate", config.DEFAULT_DROPOUT_RATE) if hyperparams else config.DEFAULT_DROPOUT_RATE
    lstm_hidden = hyperparams.get("lstm_hidden_size", config.DEFAULT_LSTM_HIDDEN_SIZE) if hyperparams else config.DEFAULT_LSTM_HIDDEN_SIZE
    lstm_layers = hyperparams.get("lstm_num_layers", config.DEFAULT_LSTM_NUM_LAYERS) if hyperparams else config.DEFAULT_LSTM_NUM_LAYERS

    # --- Initialize Model ---
    try:
        if model_type == 'cbam':
            from src.modeling.flight_chain_models import CBAM_CNN_Model
            model = CBAM_CNN_Model(
                num_features=num_features, num_classes=config.NUM_CLASSES,
                dropout_rate=dropout_rate
            ).to(device)
        elif model_type == 'simam':
            from src.modeling.flight_chain_models import SimAM_CNN_LSTM_Model
            model = SimAM_CNN_LSTM_Model(
                num_features=num_features, num_classes=config.NUM_CLASSES,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout_rate=dropout_rate
            ).to(device)
        elif model_type == 'qtsimam':
            from src.modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model
            model = QTSimAM_CNN_LSTM_Model(
                num_features=num_features, num_classes=config.NUM_CLASSES,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout_rate=dropout_rate
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        print(f"Error initializing model '{model_type}': {e}")
        traceback.print_exc()
        sys.exit(1)

    if class_weights_tensor is not None:
        print("Using Weighted CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        print("Using standard CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-7)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    overfitting_counter = 0
    patience_no_improve = 10  # epochs to wait before early stopping
    patience_overfitting = 3  # epochs to wait if overfitting is detected
    overfitting_tolerance = 0.01  # tolerance for overfitting

    print("\nStarting training loop...")
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{config.EPOCHS} | Dur: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        if val_loss > (train_loss + overfitting_tolerance):
            overfitting_counter += 1
            print(f"    INFO: Potential Overfitting Detected (Val Loss > Train Loss + Tol): Count={overfitting_counter}")
        else:
            overfitting_counter = 0

        if val_loss < best_val_loss:
            print(f"    INFO: Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            try:
                config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            except Exception as e:
                print(f"    ERROR saving model: {e}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_no_improve:
            print(f"\nEarly stopping triggered: Validation loss hasn't improved for {patience_no_improve} epochs.")
            break
        if overfitting_counter >= patience_overfitting:
            print(f"\nEarly stopping triggered: Potential overfitting detected for {patience_overfitting} consecutive epochs.")
            break

    print(f"\n--- Training Finished for {model_type.upper()} model ---")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    if config.MODEL_SAVE_PATH.exists() and best_val_loss != float('inf'):
        print(f"Best model state saved to: {config.MODEL_SAVE_PATH}")
    elif best_val_loss == float('inf'):
        print("Warning: No valid epochs completed. Model not saved.")
    else:
        print("Warning: Best model was not saved (possibly due to errors or no improvement beyond initial).")


if __name__ == "__main__":
    print("Running trainer.py directly with default settings...")
    if config.TRAIN_CHAINS_FILE.exists():
        run_training(model_type='simam', hyperparams=None)
    else:
        print("Processed training data not found. Cannot run training directly.")
        print("Run data processing first: python src/data_processing/chain_constructor.py")

