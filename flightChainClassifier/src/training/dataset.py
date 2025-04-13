# flightChainClassifier/src/training/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import sys
try:
    # Assumes the script using this module adds project root to path
    from .. import config 
except ImportError:
    # Fallback if run directly or path is not set correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir) # training -> src
    project_dir = os.path.dirname(src_dir) # src -> flightChainClassifier
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    try:
        from src import config
    except ImportError as e:
         print(f"CRITICAL: Error importing config in dataset.py: {e}")
         sys.exit(1)

class FlightChainDataset(Dataset):
    """PyTorch Dataset for loading flight chain sequences and labels."""

    def __init__(self, chains_file, labels_file):
        """
        Args:
            chains_file (pathlib.Path): Path to the .npy file containing chains (N, SeqLen, Features).
            labels_file (pathlib.Path): Path to the .npy file containing labels (N,).
        """
        if not chains_file.exists() or not labels_file.exists():
             raise FileNotFoundError(f"Data files not found: {chains_file.name} or {labels_file.name} in {chains_file.parent}")
        print(f"Loading chains from: {chains_file.name}")
        self.chains = np.load(chains_file)
        print(f"Loading labels from: {labels_file.name}")
        self.labels = np.load(labels_file)
        if len(self.chains) != len(self.labels):
            raise ValueError(f"Number of chains ({len(self.chains)}) and labels ({len(self.labels)}) must match.")
        print(f"Loaded dataset with {len(self.chains)} samples.")

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):
        chain = self.chains[idx]
        label = self.labels[idx]

        # Convert to tensors
        chain_tensor = torch.tensor(chain, dtype=torch.float32)
        # Ensure label is LongTensor for CrossEntropyLoss
        label_tensor = torch.tensor(label, dtype=torch.long)

        return chain_tensor, label_tensor

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Assumes data has been processed and saved
    if config.TRAIN_CHAINS_FILE.exists() and config.TRAIN_LABELS_FILE.exists():
        print("Testing Train Dataset loading...")
        train_dataset = FlightChainDataset(config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE)
        print(f"Number of training samples: {len(train_dataset)}")
        sample_chain, sample_label = train_dataset[0]
        print(f"Sample chain shape: {sample_chain.shape}")
        print(f"Sample label: {sample_label}")
    else:
        print("Processed training files not found, skipping dataset test.")
