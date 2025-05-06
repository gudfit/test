import torch, torch.nn as nn
from torch import logsumexp


class SoftMaxPlus(nn.Module):
    r"""
    Differentiable approximation of a max-plus recurrence
    δ₀ = 0,
    δᵢ = max{0, cᵢ + δᵢ₋₁}
        ≈ (1/β)·log( e^{β·0} + e^{β·(cᵢ+δᵢ₋₁)} ).

    Args
    ----
    beta : initial sharpness; trainable so the network can learn
           how close to hard max it should operate.
    """

    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        c : (B, S)  pre-activation “costs”  cᵢ = αᵢ + τᵢ₋₁ − τᵢ
                    (you will build this outside the layer).

        Returns
        -------
        δ : (B, S)  propagated delay for each leg.
        """
        B, S = c.shape
        delta_prev = c.new_zeros(B)  # δ₀ = 0
        outs = []
        for i in range(S):
            z = torch.stack([delta_prev, c[:, i] + delta_prev], dim=0)  # (2,B)
            delta_prev = (1.0 / self.beta) * logsumexp(self.beta * z, dim=0)
            outs.append(delta_prev.unsqueeze(1))
        return torch.cat(outs, dim=1)  # (B,S)
