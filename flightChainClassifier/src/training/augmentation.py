# flightChainClassifier/src/training/augmentation.py
import numpy as np
from .. import config


def jitter_chain(chain_np, stats, jitter_prob=0.6, max_sigma=0.25):
    """
    Create a lightly-perturbed copy of a scaled (0-1) chain.

    Parameters
    ----------
        chain_np : np.ndarray, shape (SEQ, F)
        stats    : dict taken from DATA_STATS (numeric_stats + feature_names)
        jitter_prob : probability to perturb an individual feature
        max_sigma   : maximum std-multiple to add / subtract
    """
    chain = chain_np.copy()
    for f_idx, fname in enumerate(stats["feature_names"]):
        if np.random.rand() > jitter_prob:
            continue  # leave feature untouched
        sd = stats["numeric_stats"][fname]["std"]
        noise = np.random.normal(0.0, max_sigma * sd, size=chain[:, f_idx].shape)
        chain[:, f_idx] = np.clip(chain[:, f_idx] + noise, 0.0, 1.0)
    return chain
