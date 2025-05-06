from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models import ConvBlock
from .maxplus import SoftMaxPlus
from .. import config  # ← changed (for default flag)


class ResidualDelayLayer(nn.Module):
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
        E_S = self.k_s * dist + self.eps
        lam = self.k_a / (air + self.eps)
        rho = torch.clamp(lam * E_S, max=0.99)
        W_q = rho / (1 - rho + self.eps) * E_S
        Wn = (W_q - W_q.amin()) / (W_q.amax() - W_q.amin() + self.eps)
        L_q = lam * W_q
        Ln = (L_q - L_q.amin()) / (L_q.amax() - L_q.amin() + self.eps)
        return Wn.unsqueeze(-1), Ln.unsqueeze(-1)  # (B,S,1)


class QTSimAM(nn.Module):
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, d, lq):  # x (B,C,L), d/lq (B,1,1)
        mu = x.mean(dim=2, keepdim=True)
        e = ((x - mu) ** 2).mean(dim=2, keepdim=True) + d + 0.5 * lq + self.e_lambda
        return x * self.sigmoid(e)


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
        return torch.cat(outs, dim=1), h  # (B,S,H), (B,H)


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


# ---------------------------------------------------------------------
# 3. *New* bidirectional wrapper
# ---------------------------------------------------------------------
class BiQMogrifierStack(nn.Module):
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
        h_concat = torch.cat([h_f, h_b], dim=1)
        return None, h_concat


class QTSimAM_CNN_LSTM_Model(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        cnn_channels: list[int] | None = None,
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_bidir: bool | None = None,  # ← changed
        lstm_layers: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [64, 128, 256]
        # fallback to config default when flag not provided
        if lstm_bidir is None:
            lstm_bidir = getattr(config, "LSTM_BIDIRECTIONAL", False)

        self.cnn_blocks = nn.ModuleList()
        self.att_blocks = nn.ModuleList()

        idx_distance = -3  # Distance column index
        idx_airtime = -5  # AirTime column index
        self.queue_layer = ResidualDelayLayer(idx_distance, idx_airtime)

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

        # --------------- QMogrifier stack -----------------------------
        if lstm_bidir:
            self.lstm_stack = BiQMogrifierStack(ch_in, lstm_hidden, lstm_layers)
            self.classifier = nn.Linear(lstm_hidden * 2, num_classes)
        else:
            self.lstm_stack = QMogrifierStack(ch_in, lstm_hidden, lstm_layers)
            self.classifier = nn.Linear(lstm_hidden, num_classes)

        self.delta_head = nn.Linear(lstm_hidden, 1)  # aux regression
        self.aux_w = 0.1

    # -------------- forward ------------------------------------------
    def forward(self, x, *, return_aux=False):
        B, S, F = x.shape
        d_seq, lq_seq = self.queue_layer(x)  # (B,S,1)

        # ----- CNN + queue-aware attention (B,C,S) ---------------
        z = x.permute(0, 2, 1)
        for conv, att in zip(self.cnn_blocks, self.att_blocks):
            z = conv(z)
            d = d_seq.mean(dim=1, keepdim=True).permute(0, 2, 1)
            l = lq_seq.mean(dim=1, keepdim=True).permute(0, 2, 1)
            z = att(z, d, l)
        z = z.permute(0, 2, 1)  # (B,S,C)

        # ----- temporal stack ------------------------------------
        _, h_last = self.lstm_stack(z, d_seq, lq_seq)

        logits = self.classifier(h_last)
        if return_aux:
            return logits, self.delta_head(h_last).squeeze(-1), d_seq[:, -1, 0]
        return logits

    def loss_fn(self, logits, y, delta_pred, delta_true):
        return F.cross_entropy(logits, y) + self.aux_w * F.mse_loss(
            delta_pred, delta_true
        )


# ---------------------------------------------------------------------
# 5.  Soft-Max-Plus wrapper
# ---------------------------------------------------------------------
class QTSimAM_MaxPlus_Model(QTSimAM_CNN_LSTM_Model):
    """Adds a differentiable max-plus delay chain as an extra input feature."""

    def __init__(self, *args, beta: float = 10.0, **kw):
        super().__init__(*args, **kw)
        self.maxplus = SoftMaxPlus(beta)

    def forward(self, x):
        d_seq, lq_seq = self.queue_layer(x)  # (B,S,1)

        alpha = d_seq.squeeze(-1) * lq_seq.squeeze(-1)
        tau = torch.zeros_like(alpha) + 0.25  # constant 15-min τ
        c = alpha + torch.cat([tau[:, :-1], tau[:, :1]], 1) - tau

        delta_soft = self.maxplus(c)  # (B,S)
        x_aug = torch.cat([x, delta_soft.unsqueeze(-1)], dim=2)

        return super().forward(x_aug)
