# flightChainClassifier/src/evaluation/evaluate.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os
import sys
from tqdm import tqdm

try:
    # Use absolute imports assuming project root is in path
    from src import config
    from src.training.dataset import FlightChainDataset
    from src.modeling.flight_chain_models import CBAM_CNN_Model, SimAM_CNN_LSTM_Model
except ImportError:
    # Fallback if run directly or path is not set correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)  # evaluation -> src
    project_dir = os.path.dirname(src_dir)  # src -> flightChainClassifier
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    try:
        from src import config
        from src.training.dataset import FlightChainDataset
        from src.modeling.flight_chain_models import CBAM_CNN_Model, SimAM_CNN_LSTM_Model
    except ImportError as e:
         print(f"CRITICAL: Error importing modules in evaluate.py: {e}")
         sys.exit(1)

@torch.no_grad()  # Disable gradient calculations during evaluation
def run_evaluation(model_type='simam'):
    """Loads the best model and evaluates it on the test set."""
    print(f"--- Starting Evaluation for {model_type.upper()} model ---")
    device = config.DEVICE

    # --- Load Data Stats (for num_features) ---
    data_stats = config.load_data_stats()
    if data_stats is None or 'num_features' not in data_stats:
        print("Error: Could not load data stats or 'num_features' missing.")
        sys.exit(1)
    num_features = data_stats['num_features']

    # --- Load Test Data ---
    try:
        test_dataset = FlightChainDataset(config.TEST_CHAINS_FILE, config.TEST_LABELS_FILE)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                   num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    except FileNotFoundError:
        print("Error: Test data files (.npy) not found. Run chain_constructor first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating test dataset/dataloader: {e}")
        sys.exit(1)

    # --- Load Best Model ---
    if not config.MODEL_SAVE_PATH.exists():
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}. Run training first.")
        sys.exit(1)

    print(f"Loading best model state from {config.MODEL_SAVE_PATH}...")

    # --- Load Best Hyperparameters if Available ---
    best_params = config.load_best_hyperparameters()
    if best_params is not None:
        print("Best hyperparameters loaded successfully.")
    else:
        print("No best hyperparameters found; using defaults.")

    # --- Instantiate the Model based on model_type ---
    if model_type == 'cbam':
        model = CBAM_CNN_Model(num_features=num_features, num_classes=config.NUM_CLASSES)
    elif model_type == 'simam':
        if best_params is not None:
            lstm_hidden = best_params.get("lstm_hidden_size", config.DEFAULT_LSTM_HIDDEN_SIZE)
            lstm_layers = best_params.get("lstm_num_layers", config.DEFAULT_LSTM_NUM_LAYERS)
            dropout_rate = best_params.get("dropout_rate", config.DEFAULT_DROPOUT_RATE)
        else:
            lstm_hidden = config.DEFAULT_LSTM_HIDDEN_SIZE
            lstm_layers = config.DEFAULT_LSTM_NUM_LAYERS
            dropout_rate = config.DEFAULT_DROPOUT_RATE
        model = SimAM_CNN_LSTM_Model(
            num_features=num_features,
            num_classes=config.NUM_CLASSES,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate
        )
    elif model_type == 'qtsimam':
        # Use the best hyperparameters for qtsimam as well.
        if best_params is not None:
            lstm_hidden = best_params.get("lstm_hidden_size", config.DEFAULT_LSTM_HIDDEN_SIZE)
            lstm_layers = best_params.get("lstm_num_layers", config.DEFAULT_LSTM_NUM_LAYERS)
            dropout_rate = best_params.get("dropout_rate", config.DEFAULT_DROPOUT_RATE)
        else:
            lstm_hidden = config.DEFAULT_LSTM_HIDDEN_SIZE
            lstm_layers = config.DEFAULT_LSTM_NUM_LAYERS
            dropout_rate = config.DEFAULT_DROPOUT_RATE
        try:
            from src.modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model
        except ImportError as e:
            print(f"Error importing QTSimAM_CNN_LSTM_Model: {e}")
            sys.exit(1)
        model = QTSimAM_CNN_LSTM_Model(
            num_features=num_features,
            num_classes=config.NUM_CLASSES,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate
        )
    else:
        print(f"Error: Unknown model type '{model_type}'.")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        model.to(device)
        model.eval()  # Set the model to evaluation mode.
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)

    # --- Run Inference ---
    print("Running inference on test set...")
    all_preds = []
    all_labels = []

    for chains, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        chains = chains.to(device)
        outputs = model(chains)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Calculate Metrics ---
    print("Calculating metrics...")
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # --- Save Metrics ---
    metrics_dict = {
        'model_type': model_type,
        'accuracy': accuracy,
        'classification_report': report
    }
    eval_file_path = config.EVAL_DIR / f"{model_type}_metrics.json"
    print(f"Saving evaluation metrics to {eval_file_path}...")
    try:
        with open(eval_file_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
    except Exception as e:
        print(f"Error saving metrics JSON: {e}")

    # --- Plot Confusion Matrix ---
    plot_file_path = config.PLOTS_DIR / f"{model_type}_confusion_matrix.png"
    print(f"Saving confusion matrix plot to {plot_file_path}...")
    try:
        class_names = [f"Class {i}" for i in range(config.NUM_CLASSES)]
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 8))
        display.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix - {model_type.upper()} Model')
        plt.tight_layout()
        plt.savefig(plot_file_path)
        plt.close(fig)
        print("Plot saved.")
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")

    print(f"--- Evaluation Finished for {model_type.upper()} model ---")


if __name__ == "__main__":
    # Example: Running evaluation for qtsimam model (or change model_type as needed)
    run_evaluation(model_type='qtsimam')

