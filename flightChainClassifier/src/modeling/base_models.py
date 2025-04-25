# flightChainClassifier/src/modeling/base_models.py
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic 1D Convolutional Block: Conv1d -> BatchNorm -> Activation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        activation=nn.ReLU,
        dropout_rate=0.0,
    ):
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            activation(),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # Input x: (Batch, Channels, SeqLen)
        return self.block(x)


# Using standard LSTM as placeholder for Mogrifier LSTM
class StandardLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, bidirectional, dropout_rate
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Expect input as (Batch, SeqLen, Features)
            bidirectional=bidirectional,
            dropout=(
                dropout_rate if num_layers > 1 else 0
            ),  # Dropout between LSTM layers
        )

    def forward(self, x):
        # Input x: (Batch, SeqLen, Features)
        # Output: output (Batch, SeqLen, Hidden*Directions), (h_n, c_n)
        # h_n shape: (NumLayers*Directions, Batch, Hidden)
        output, (h_n, c_n) = self.lstm(x)
        # Usually need the final hidden state for classification
        # If bidirectional, h_n combines forward and backward last states
        if self.lstm.bidirectional:
            # Concatenate the last hidden state of forward and backward layers
            # h_n is (NumLayers*2, Batch, Hidden). Get last layer's forward and backward.
            # Forward: h_n[-2, :, :], Backward: h_n[-1, :, :]
            final_hidden = torch.cat(
                (h_n[-2, :, :], h_n[-1, :, :]), dim=1
            )  # (Batch, Hidden*2)
        else:
            # Get the last hidden state of the last layer
            final_hidden = h_n[-1, :, :]  # (Batch, Hidden)

        return output, final_hidden
