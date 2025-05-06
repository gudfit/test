# flightChainClassifier/src/modeling/flight_chain_models.py
# ---------------------------------------------------------------------
#  This file defines all “end-to-end” models for the project:
#
#   1.  CBAM_CNN_Model
#   2.  SimAM_CNN_LSTM_Model
#   3.  QTSimAM_CNN_LSTM_Model
#   4.  QTSimAM_MaxPlus_Model
#
#  New in this version:
#     • Multi-head self-attention block (1-D) available as
#       `MultiHeadSelfAttention1D` and integrated in ➊, ➋, ➌.
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ #
#  Project-internal imports
# ------------------------------------------------------------------ #
try:
    # absolute form (preferred when the package is installed)
    from .base_models import ConvBlock, StandardLSTM
    from .attention_modules import CBAM, SimAM, MultiHeadSelfAttention1D
    from .. import config
except ImportError:
    # fallback for “python file.py” without installing the package
    _here = os.path.dirname(os.path.abspath(__file__))
    _src = os.path.dirname(_here)
    _root = os.path.dirname(_src)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.modeling.base_models import ConvBlock, StandardLSTM
    from src.modeling.attention_modules import CBAM, SimAM, MultiHeadSelfAttention1D
    from src import config


# ======================================================================
# 1.  CBAM-CNN
# ======================================================================
class CBAM_CNN_Model(nn.Module):
    """
    Pure-CNN backbone with CBAM after every ConvBlock,
    followed by **one global MH-Self-Attention** and a linear classifier.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        cnn_channels: list[int] | None = None,
        kernel_size: int | None = None,
        dropout_rate: float | None = None,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()

        channels = cnn_channels or config.DEFAULT_CNN_CHANNELS
        ksize = kernel_size or config.DEFAULT_KERNEL_SIZE
        drop = dropout_rate if dropout_rate is not None else config.DEFAULT_DROPOUT_RATE

        # ------------- feature extractor (Conv + CBAM) ----------------
        chs = [num_features] + channels
        blocks: list[nn.Module] = []
        for c_in, c_out in zip(chs[:-1], chs[1:]):
            blocks.append(
                ConvBlock(c_in, c_out, ksize, padding="same", dropout_rate=drop)
            )
            blocks.append(CBAM(c_out))
        self.feature_extractor = nn.Sequential(*blocks)

        # ------------- NEW: global 1-D self attention -----------------
        self.self_att = MultiHeadSelfAttention1D(
            embed_dim=chs[-1], num_heads=attn_heads, dropout=0.1
        )

        # ------------- classifier head -------------------------------
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, C_last, 1)
        self.classifier = nn.Linear(chs[-1], num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S=3, F)  →  (B, F, S)
        x = x.permute(0, 2, 1)

        # Conv + CBAM → (B, C_last, S)
        x = self.feature_extractor(x)

        # MH-Self-Attention (residual)
        x = self.self_att(x) + x

        # Global pooling & FC
        x = self.pool(x).squeeze(-1)  # (B, C_last)
        return self.classifier(x)


# ======================================================================
# 2.  SimAM-CNN-LSTM
# ======================================================================
class SimAM_CNN_LSTM_Model(nn.Module):
    """
    CNN + SimAM at each layer, *then* a self-attention, *then* an LSTM.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        cnn_channels: list[int] | None = None,
        kernel_size: int | None = None,
        lstm_hidden: int | None = None,
        lstm_layers: int | None = None,
        lstm_bidir: bool | None = None,
        dropout_rate: float | None = None,
        attn_heads: int = 8,
    ) -> None:
        super().__init__()

        # Defaults from config
        channels = cnn_channels or config.DEFAULT_CNN_CHANNELS
        ksize = kernel_size or config.DEFAULT_KERNEL_SIZE
        lstm_hidden = lstm_hidden or config.DEFAULT_LSTM_HIDDEN_SIZE
        lstm_layers = lstm_layers or config.DEFAULT_LSTM_NUM_LAYERS
        lstm_bidir = (
            lstm_bidir if lstm_bidir is not None else config.DEFAULT_LSTM_BIDIRECTIONAL
        )
        drop = dropout_rate if dropout_rate is not None else config.DEFAULT_DROPOUT_RATE

        # ---------------- CNN + SimAM stack --------------------------
        chs = [num_features] + channels
        layers: list[nn.Module] = []
        for c_in, c_out in zip(chs[:-1], chs[1:]):
            layers.append(
                ConvBlock(c_in, c_out, ksize, padding="same", dropout_rate=drop)
            )
            layers.append(SimAM())
        self.feature_extractor = nn.Sequential(*layers)

        # ---------------- Self-Attention before LSTM -----------------
        self.self_att = MultiHeadSelfAttention1D(
            chs[-1], num_heads=attn_heads, dropout=0.1
        )

        # ---------------- LSTM ---------------------------------------
        self.lstm = StandardLSTM(
            input_size=chs[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=lstm_bidir,
            dropout_rate=drop if lstm_layers > 1 else 0.0,
        )

        # classifier
        lstm_out_feats = lstm_hidden * (2 if lstm_bidir else 1)
        self.classifier = nn.Linear(lstm_out_feats, num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, F) → (B, F, S)
        x = x.permute(0, 2, 1)

        x = self.feature_extractor(x)  # (B, C_last, S)
        x = self.self_att(x) + x  # residual

        # (B, C_last, S) → (B, S, C_last)
        x = x.permute(0, 2, 1)

        _seq, h_last = self.lstm(x)
        return self.classifier(h_last)
