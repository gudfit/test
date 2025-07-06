# Re-export Wanda’s pruning functions under the `wanda.prune` namespace
from lib.prune import (
    prune_wanda,
    prune_magnitude,
    prune_sparsegpt,
    prune_ablate,
    check_sparsity,
    find_layers,
)
