from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models import ConvBlock
from .attention_modules import MultiHeadSelfAttention1D
from .maxplus import SoftMaxPlus
from .. import config


# =====================================================================
# 1.   Small helper layers
# =====================================================================
class ResidualDelayLayer(nn.Module):
    """
    Compute crude distance- & airtime-based proxies for queue delay (d)
    and waiting-time-in-queue (lq), normalised to [0,1].
    """

    def __init__(self, idx_distance: int, idx_airtime: int):
        super().__init__()
        self.k_s = nn.Parameter(torch.tensor(1.0))
        self.k_a = nn.Parameter(torch.tensor(1.0))
        self.idx_distance = idx_distance
        self.idx_airtime = idx_airtime
        self.eps = 1e-6

    def forward(self, x):  # x (B,S,F)
        dist = x[:, :, self.idx_distance]
        air = x[:, :, self.idx_airtime]
        E_s = self.k_s * dist + self.eps  # service time proxy
        lam = self.k_a / (air + self.eps)  # arrival rate proxy
        rho = torch.clamp(lam * E_s, max=0.99)  # traffic intensity
        W_q = rho / (1 - rho + self.eps) * E_s  # queueing delay (Pollaczek)
        Wn = (W_q - W_q.amin()) / (W_q.amax() - W_q.amin() + self.eps)
        L_q = lam * W_q
        Ln = (L_q - L_q.amin()) / (L_q.amax() - L_q.amin() + self.eps)
        return Wn.unsqueeze(-1), Ln.unsqueeze(-1)  # (B,S,1)


class QTSimAM(nn.Module):
    """SimAM variant that also injects queue-derived scalars (d, lq)."""

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, d, lq):  # x (B,C,L), d/lq (B,1,1)
        mu = x.mean(dim=2, keepdim=True)
        e = ((x - mu) ** 2).mean(dim=2, keepdim=True) + d + 0.5 * lq + self.e_lambda
        return x * self.sigmoid(e)


# =====================================================================
# 2.   (Bi-)QMogrifier recurrent blocks
# =====================================================================
class QMogrifierCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.mod = nn.Linear(hidden_size + 2, input_size)

    def forward(self, seq, d_seq, lq_seq):  # seq (B,S,C)
        B, S, _ = seq.shape
        h = seq.new_zeros(B, self.hidden_size)
        c = seq.new_zeros(B, self.hidden_size)
        outs = []
        for t in range(S):
            gate_in = torch.cat([h, d_seq[:, t], lq_seq[:, t]], dim=1)
            m = torch.sigmoid(self.mod(gate_in))
            h, c = self.cell(m * seq[:, t], (h, c))
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, 1), h  # (B,S,H), (B,H)


class QMogrifierStack(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                QMogrifierCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )

    def forward(self, seq, d_seq, lq_seq):
        last_h = None
        for cell in self.layers:
            seq, last_h = cell(seq, d_seq, lq_seq)
        return seq, last_h


class BiQMogrifierStack(nn.Module):
    """Simple forward+backward wrapper around the stack."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.fwd = QMogrifierStack(input_size, hidden_size, num_layers)
        self.bwd = QMogrifierStack(input_size, hidden_size, num_layers)

    def forward(self, seq, d_seq, lq_seq):  # (B,S,C)
        seq_r = torch.flip(seq, [1])
        d_r = torch.flip(d_seq, [1])
        lq_r = torch.flip(lq_seq, [1])

        _, h_f = self.fwd(seq, d_seq, lq_seq)
        _, h_b = self.bwd(seq_r, d_r, lq_r)
        return None, torch.cat([h_f, h_b], dim=1)  # (B,2H)


# =====================================================================
# 3.   Main model
# =====================================================================
class QTSimAM_CNN_LSTM_Model(nn.Module):
    """
    CNN + QTSimAM  →  **global self-attention**  →  (Bi-)QMogrifier stack.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        cnn_channels: list[int] | None = None,
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        lstm_bidir: bool | None = None,
        dropout_rate: float = 0.2,
        attn_heads: int = 4,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [64, 128, 256]
        if lstm_bidir is None:
            lstm_bidir = getattr(config, "LSTM_BIDIRECTIONAL", False)

        # ---------------- queue delay proxies ----------------------
        idx_distance, idx_airtime = -3, -5
        self.queue_layer = ResidualDelayLayer(idx_distance, idx_airtime)

        # ---------------- CNN + QTSimAM blocks ---------------------
        self.cnn_blocks = nn.ModuleList()
        self.att_blocks = nn.ModuleList()

        ch_in = num_features
        for ch_out in cnn_channels:
            self.cnn_blocks.append(
                ConvBlock(
                    ch_in,
                    ch_out,
                    kernel_size,
                    padding="same",
                    dropout_rate=dropout_rate,
                )
            )
            self.att_blocks.append(QTSimAM())
            ch_in = ch_out

        # ---------------- global MH self-attention ----------------
        self.pre_qm_att = MultiHeadSelfAttention1D(
            ch_in, num_heads=attn_heads, dropout=0.1
        )

        # ---------------- (Bi-)QMogrifier stack -------------------
        if lstm_bidir:
            self.lstm_stack = BiQMogrifierStack(ch_in, lstm_hidden, lstm_layers)
            last_dim = lstm_hidden * 2
        else:
            self.lstm_stack = QMogrifierStack(ch_in, lstm_hidden, lstm_layers)
            last_dim = lstm_hidden

        # ---------------- heads -----------------------------------
        self.classifier = nn.Linear(last_dim, num_classes)
        self.delta_head = nn.Linear(lstm_hidden, 1)  # aux regression
        self.aux_w = 0.1

    # --------------------------------------------------------------
    def forward(self, x, *, return_aux: bool = False):
        # x (B,S,F)
        d_seq, lq_seq = self.queue_layer(x)  # (B,S,1)

        # CNN + local queue-aware attention
        z = x.permute(0, 2, 1)  # (B,F,S)
        for conv, att in zip(self.cnn_blocks, self.att_blocks):
            z = conv(z)
            d = d_seq.mean(1, keepdim=True).permute(0, 2, 1)
            l = lq_seq.mean(1, keepdim=True).permute(0, 2, 1)
            z = att(z, d, l)  # (B,C,S)

        # global self-attention (residual)
        z = self.pre_qm_att(z) + z  # (B,C,S)

        # QMogrifier expects (B,S,C)
        z = z.permute(0, 2, 1)
        _, h_last = self.lstm_stack(z, d_seq, lq_seq)  # (B,last_dim)

        logits = self.classifier(h_last)
        if return_aux:
            delta = self.delta_head(h_last).squeeze(-1)
            return logits, delta, d_seq[:, -1, 0]
        return logits

    # --------------------------------------------------------------
    def loss_fn(self, logits, y, delta_pred, delta_true):
        ce = F.cross_entropy(logits, y)
        mse = F.mse_loss(delta_pred, delta_true)
        return ce + self.aux_w * mse


# =====================================================================
# 4.   Soft-Max-Plus wrapper
# =====================================================================
class QTSimAM_MaxPlus_Model(QTSimAM_CNN_LSTM_Model):
    """
    Same as QTSimAM_CNN_LSTM_Model, but augments each feature vector
    with a differentiable soft max-plus delay estimate.
    """

    def __init__(self, *args, beta: float = 10.0, **kw):
        super().__init__(*args, **kw)
        self.maxplus = SoftMaxPlus(beta)

    def forward(self, x):
        # max-plus delay chain (δ_soft)
        d_seq, lq_seq = self.queue_layer(x)  # (B,S,1)
        alpha = d_seq.squeeze(-1) * lq_seq.squeeze(-1)
        tau = torch.zeros_like(alpha) + 0.25  # constant τ = 15 min
        c = alpha + torch.cat([tau[:, :-1], tau[:, :1]], 1) - tau
        delta_soft = self.maxplus(c)  # (B,S)

        # append as an extra scalar feature
        x_aug = torch.cat([x, delta_soft.unsqueeze(-1)], dim=2)
        return super().forward(x_aug)
