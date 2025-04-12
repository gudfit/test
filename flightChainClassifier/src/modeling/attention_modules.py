# flightChainClassifier/src/modeling/attention_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- CBAM Implementation ---
# Based on various public implementations, e.g.,
# https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
# https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x shape: (Batch, Channels, SeqLen) for Conv1d
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x shape: (Batch, Channels, SeqLen)
        # Permute for avg/max pool across channels: (Batch, SeqLen, Channels)
        x_perm = x.permute(0, 2, 1)
        avg_out = torch.mean(x_perm, dim=2, keepdim=True) # (Batch, SeqLen, 1)
        max_out, _ = torch.max(x_perm, dim=2, keepdim=True) # (Batch, SeqLen, 1)
        # Concatenate along channel dim (now dim=2)
        y = torch.cat([avg_out, max_out], dim=2) # (Batch, SeqLen, 2)
        # Permute back for Conv1d: (Batch, 2, SeqLen)
        y = y.permute(0, 2, 1)
        y = self.conv1(y) # (Batch, 1, SeqLen)
        return self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelAtt = ChannelAttention(gate_channels, reduction_ratio)
        self.SpatialAtt = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # Input x shape: (Batch, Channels, SeqLen)
        # Apply Channel Attention
        channel_weights = self.ChannelAtt(x) # (Batch, Channels, 1)
        x_out = x * channel_weights # Broadcast multiplication: (B, C, S) * (B, C, 1) -> (B, C, S)

        # Apply Spatial Attention
        spatial_weights = self.SpatialAtt(x_out) # (Batch, 1, SeqLen)
        x_out = x_out * spatial_weights # Broadcast multiplication: (B, C, S) * (B, 1, S) -> (B, C, S)
        return x_out

# --- SimAM Implementation ---
# Based on the official implementation: https://github.com/ZjjConan/SimAM/blob/master/networks/simam.py
# Adapted for 1D convolutions (applied across sequence length)

class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4): # channels arg is not used in original SimAM, but good practice
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        # Note: Original SimAM calculates statistics per spatial location (H*W).
        # For 1D Conv (SeqLen), we calculate statistics across the sequence length.

    def forward(self, x):
        # Input x shape: (Batch, Channels, SeqLen)
        b, c, s = x.shape # Batch, Channels, Sequence Length

        # Calculate statistics across the sequence dimension (dim=2)
        mu = torch.mean(x, dim=2, keepdim=True) # (B, C, 1)
        var = torch.var(x, dim=2, keepdim=True) # (B, C, 1)

        # Calculate energy, avoiding division by zero
        numerator = (x - mu)**2
        denominator = 4 * (var + self.e_lambda)
        e_inv = numerator / denominator + 0.5 # This is 1 / e_t in the paper's notation

        # Apply sigmoid to get attention weights
        return x * self.activaton(e_inv)
