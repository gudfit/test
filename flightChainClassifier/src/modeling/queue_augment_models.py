# flightChainClassifier/src/modeling/queue_augment_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.base_models import ConvBlock

###############################################################################
# 1. QUEUE-AUGMENTED SIMAM MODULE
###############################################################################

class QTSimAM(nn.Module):
    """
    Queue-Augmented SimAM Attention Module.
    
    For an input feature tensor x of shape (B, C, L) and a scalar queue value delta
    (aggregated residual delay), the energy used for attention is augmented as:
    
      e = mean[(x - μ)^2] + λ_queue * delta
      
    The final attention weights are computed as: 
      att = sigmoid(e)
      
    and the output is given by elementwise multiplication: x * att.
    """
    def __init__(self, e_lambda=1e-4, lambda_queue=1.0):
        super(QTSimAM, self).__init__()
        self.e_lambda = e_lambda
        # lambda_queue is a learnable scalar parameter.
        self.lambda_queue = nn.Parameter(torch.tensor(lambda_queue, dtype=torch.float32))

    def forward(self, x, delta):
        # x: (B, C, L)
        mu = torch.mean(x, dim=2, keepdim=True)
        energy = torch.mean((x - mu) ** 2, dim=2, keepdim=True)
        # Augment the energy with the queue penalty.
        e = energy + self.lambda_queue * delta  # delta is broadcasted to (B, C, 1)
        att = torch.sigmoid(e)
        return x * att

###############################################################################
# 2. QUEUE-AUGMENTED (MOGRIFIER) LSTM MODULE
###############################################################################

class QMogrifierLSTM(nn.Module):
    """
    A simple LSTM cell with input modulation based on queuing information.
    
    For each time step t, the input x_t is modulated as:
       x_t ← sigmoid(W_m([h_{t-1}; Δ_{t-1}])) ⊙ x_t,
    where h_{t-1} is the previous hidden state and Δ_{t-1} is the residual delay 
    from the previous flight.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.0):
        super(QMogrifierLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        # The modulation linear layer takes in the concatenation of h_{t-1} and delta (size hidden_size + 1)
        # and outputs a modulation vector that matches the input_size.
        self.modulation = nn.Linear(hidden_size + 1, input_size)

    def forward(self, x, delta):
        """
        Args:
            x: Input tensor of shape (B, S, input_size), where S is the sequence length.
            delta: Queue parameters for each time step, tensor of shape (B, S).
        Returns:
            outputs: Tensor of shape (B, S, hidden_size) containing LSTM outputs.
            final_hidden: Tensor of shape (B, hidden_size) for the final hidden state.
        """
        batch_size, seq_len, input_size = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, input_size)
            # For t == 0, if no delta is available, use zeros.
            delta_t = delta[:, t] if t < delta.size(1) else torch.zeros(batch_size, device=x.device)
            # Concatenate previous hidden state and delta_t.
            mod_input = torch.cat([h_t, delta_t.unsqueeze(1)], dim=1)  # (B, hidden_size+1)
            m = torch.sigmoid(self.modulation(mod_input))  # (B, input_size)
            # Modulate the input.
            x_t = m * x_t
            h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))
            outputs.append(h_t.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (B, S, hidden_size)
        return outputs, h_t

###############################################################################
# 3. QUEUE-AUGMENTED SIMAM-CNN-LSTM MODEL
###############################################################################

class QTSimAM_CNN_LSTM_Model(nn.Module):
    """
    A queue-augmented CNN-LSTM model that incorporates queuing information into both:
      1) The attention mechanism (via QTSimAM) applied after convolutional blocks.
      2) The temporal processing (via QMogrifierLSTM) that modulates each time-step's input.
      
    This model expects an input tensor of shape (B, S, num_features) representing a flight chain
    (e.g., S = 3 flights). It internally estimates a residual delay (queue parameter) for each flight
    using a simple linear estimator that uses one feature as a proxy.
    """
    def __init__(self, num_features, num_classes,
                 cnn_channels=None, kernel_size=None,
                 lstm_hidden=None, lstm_layers=None,
                 lstm_bidir=None, dropout_rate=None):
        super(QTSimAM_CNN_LSTM_Model, self).__init__()

        # Use default parameters if not provided.
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]
        if kernel_size is None:
            kernel_size = 3
        if lstm_hidden is None:
            lstm_hidden = 64
        if lstm_layers is None:
            lstm_layers = 2
        if lstm_bidir is None:
            lstm_bidir = False
        if dropout_rate is None:
            dropout_rate = 0.2

        self.num_features = num_features
        # Build the list of channels for CNN layers.
        self.cnn_channels_list = [num_features] + cnn_channels

        cnn_layers = []
        self.qtsimam_modules = nn.ModuleList()
        # Build CNN blocks followed by queue-augmented attention modules.
        for i in range(len(self.cnn_channels_list) - 1):
            in_c = self.cnn_channels_list[i]
            out_c = self.cnn_channels_list[i+1]
            cnn_layers.append(ConvBlock(in_c, out_c, kernel_size=kernel_size, dropout_rate=dropout_rate, padding='same'))
            self.qtsimam_modules.append(QTSimAM(e_lambda=1e-4, lambda_queue=1.0))
        self.feature_extractor = nn.Sequential(*cnn_layers)

        # Delta estimator: now expects an input of shape (..., 1) instead of num_features.
        self.delta_estimator = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()  # scales output to [0, 1]
        )

        final_cnn_channels = self.cnn_channels_list[-1]
        # Use the QMogrifierLSTM to process the CNN features.
        self.qmogrifier_lstm = QMogrifierLSTM(input_size=final_cnn_channels, hidden_size=lstm_hidden, num_layers=1, dropout_rate=dropout_rate)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, S, num_features) where S is the chain length.
        Returns:
            logits: Tensor of shape (B, num_classes).
        """
        batch_size, seq_len, _ = x.size()
        # Permute input to (B, num_features, S) for 1D convolutions.
        x_cnn = x.permute(0, 2, 1)
        for i, layer in enumerate(self.feature_extractor):
            x_cnn = layer(x_cnn)
            # Compute delta using the first feature.
            # Ensure shape is (B, S, 1) using unsqueeze(-1).
            delta = self.delta_estimator(x[:, :, 0].unsqueeze(-1))  # (B, S, 1)
            # Average over the sequence dimension to get a scalar per batch.
            delta_avg = delta.mean(dim=1, keepdim=True)  # (B, 1, 1)
            # Apply the queue-augmented attention module.
            x_cnn = self.qtsimam_modules[i](x_cnn, delta_avg)
        # Permute back to (B, S, final_cnn_channels).
        features = x_cnn.permute(0, 2, 1)

        # Compute the sequence of delta values for LSTM input.
        # Again, ensure the input shape is (B, S, 1) and then remove the last dimension.
        delta_seq = self.delta_estimator(x[:, :, 0].unsqueeze(-1)).squeeze(-1)  # (B, S)

        # Process the features with the QMogrifierLSTM.
        lstm_out, final_hidden = self.qmogrifier_lstm(features, delta_seq)
        # Final classification using the last hidden state.
        logits = self.classifier(final_hidden)
        return logits
