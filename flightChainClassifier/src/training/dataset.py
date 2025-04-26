# flightChainClassifier/src/training/dataset.py
import torch
from torch.utils.data import Dataset

import numpy as np
import os
import sys
import json


try:
    from .. import config
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(src_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from src import config

from .augmentation import jitter_chain


class FlightChainDataset(Dataset):
    """PyTorch Dataset for loading flight chain sequences and labels."""

    def __init__(self, chains_file, labels_file):
        """
        Args:
            chains_file (pathlib.Path): Path to the .npy file containing chains (N, SeqLen, Features).
            labels_file (pathlib.Path): Path to the .npy file containing labels (N,).
        """
        if not chains_file.exists() or not labels_file.exists():
            raise FileNotFoundError(
                f"Data files not found: {chains_file.name} or {labels_file.name} in {chains_file.parent}"
            )
        print(f"Loading chains from: {chains_file.name}")
        self.chains = np.load(chains_file)
        print(f"Loading labels from: {labels_file.name}")
        self.labels = np.load(labels_file)
        if len(self.chains) != len(self.labels):
            raise ValueError(
                f"Number of chains ({len(self.chains)}) and labels ({len(self.labels)}) must match."
            )
        print(f"Loaded dataset with {len(self.chains)} samples.")

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):
        chain = self.chains[idx]
        label = self.labels[idx]
        return torch.tensor(chain, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


class FlightChainDatasetSim(Dataset):
    """
    Simulation-augmented Dataset: returns SIM_FACTOR variants per base chain,
    with on-the-fly jitter applied to all but the first replica.
    """

    def __init__(self, chains_file, labels_file, data_stats, sim_factor=None):
        self.base_chains = np.load(chains_file)
        self.base_labels = np.load(labels_file)
        # Alias `labels` for compatibility
        self.labels = self.base_labels
        self.stats = data_stats
        self.sim_factor = sim_factor if sim_factor is not None else config.SIM_FACTOR
        print(
            f"Loaded simulated dataset with base size={len(self.base_chains)} and SIM_FACTOR={self.sim_factor}"
        )

    def __len__(self):
        return len(self.base_chains) * self.sim_factor

    def __getitem__(self, idx):
        base_idx = idx // self.sim_factor
        replica_id = idx % self.sim_factor
        chain = self.base_chains[base_idx]
        label = self.base_labels[base_idx]
        if replica_id:  # jitter all but the first sample
            chain = jitter_chain(chain, self.stats)
        return torch.tensor(chain, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


if __name__ == "__main__":
    print("Running dataset.py self-test…")
    required = [
        config.TRAIN_CHAINS_FILE,
        config.TRAIN_LABELS_FILE,
        config.DATA_STATS_FILE,
    ]
    if not all(p.exists() for p in required):
        print("Processed data not found – run the chain constructor first.")
        sys.exit(0)
    with open(config.DATA_STATS_FILE) as f:
        stats = json.load(f)

    ds_plain = FlightChainDataset(config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE)
    ds_sim = FlightChainDatasetSim(
        config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE, stats
    )

    print("Plain sample   :", ds_plain[0][0].shape, ds_plain[0][1])
    print("Simulated sample:", ds_sim[1][0].shape, ds_sim[1][1])
    print("✔ dataset.py self-test finished")
