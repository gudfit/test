# flightChainClassifier/src/modeling/flight_chain_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# --- Path Setup & Imports ---
# Ensure src is in path to import other modules
try:
    # Use absolute imports assuming project root is in path
    from src.modeling.base_models import ConvBlock, StandardLSTM
    from src.modeling.attention_modules import CBAM, SimAM
    from src import config # Absolute import for config
except ImportError:
    # Fallback if run directly or path is not set correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir) # modeling -> src
    project_dir = os.path.dirname(src_dir) # src -> flightChainClassifier
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    try:
        from src.modeling.base_models import ConvBlock, StandardLSTM
        from src.modeling.attention_modules import CBAM, SimAM
        from src import config
    except ImportError as e:
         print(f"CRITICAL: Error importing modules in flight_chain_models.py: {e}")
         sys.exit(1)


class CBAM_CNN_Model(nn.Module):
    """
    Model inspired by CondenseNet/DenseNet with CBAM attention.
    Uses CNN layers applied independently to each flight feature vector in the sequence,
    then pools features across the sequence dimension before classification.

    Accepts hyperparameters or uses defaults from config.py.
    """
    def __init__(self, num_features, num_classes,
                 cnn_channels=None, # List of output channels for CNN blocks
                 kernel_size=None,  # Kernel size for Conv1d
                 dropout_rate=None): # Dropout rate for ConvBlocks
        super().__init__()

        # --- Set Parameters (Use provided or default from config) ---
        cnn_channels_param = cnn_channels if cnn_channels is not None else config.DEFAULT_CNN_CHANNELS
        kernel_size_param = kernel_size if kernel_size is not None else config.DEFAULT_KERNEL_SIZE
        dropout_rate_param = dropout_rate if dropout_rate is not None else config.DEFAULT_DROPOUT_RATE

        self.num_features = num_features
        # Define channel progression including input features
        self.cnn_channels_list = [num_features] + cnn_channels_param

        # --- Build Feature Extractor Layers ---
        conv_layers = []
        # print(f"Building CBAM_CNN_Model with channels: {self.cnn_channels_list}, kernel: {kernel_size_param}, dropout: {dropout_rate_param}")
        for i in range(len(self.cnn_channels_list) - 1):
            in_c = self.cnn_channels_list[i]
            out_c = self.cnn_channels_list[i+1]
            conv_layers.append(ConvBlock(
                in_c, out_c,
                kernel_size=kernel_size_param,
                dropout_rate=dropout_rate_param,
                padding='same'
            ))
            conv_layers.append(CBAM(out_c))
            # print(f"  Added ConvBlock({in_c}, {out_c}) + CBAM({out_c})")

        self.feature_extractor = nn.Sequential(*conv_layers)

        # --- Classifier Head ---
        # Pool features across the sequence length dimension
        self.pool = nn.AdaptiveAvgPool1d(1) # Global Average Pooling -> (Batch, LastChannel, 1)
        self.flatten = nn.Flatten() # -> (Batch, LastChannel)

        # Final linear layer for classification
        final_cnn_channels = self.cnn_channels_list[-1]
        self.classifier = nn.Linear(final_cnn_channels, num_classes)
        # print(f"  Added Classifier Head: Linear({final_cnn_channels}, {num_classes})")

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, SeqLen=3, Features).
        Returns:
            torch.Tensor: Output logits of shape (Batch, NumClasses).
        """
        # Conv1d expects input shape (Batch, Features, SeqLen)
        x = x.permute(0, 2, 1) # Reshape to (Batch, Features, SeqLen=3)

        # Pass through CNN blocks + CBAM
        # Input: (B, F, S=3), Output: (B, C_last, S=3)
        features = self.feature_extractor(x)

        # Pool across sequence length dimension (dim=2)
        # Input: (B, C_last, S=3), Output: (B, C_last, 1)
        pooled_features = self.pool(features)

        # Flatten for the classifier
        # Input: (B, C_last, 1), Output: (B, C_last)
        flat_features = self.flatten(pooled_features)

        # Classify
        # Input: (B, C_last), Output: (B, NumClasses)
        logits = self.classifier(flat_features)
        return logits


class SimAM_CNN_LSTM_Model(nn.Module):
    """
    Model using CNN blocks with SimAM attention for feature extraction per step,
    followed by LSTM (StandardLSTM used here) to capture temporal dependencies
    across the flight chain sequence.

    Accepts hyperparameters or uses defaults from config.py.
    """
    def __init__(self, num_features, num_classes,
                 cnn_channels=None, kernel_size=None,
                 lstm_hidden=None, lstm_layers=None,
                 lstm_bidir=None, dropout_rate=None):
        super().__init__()

        # --- Set Parameters (Use provided or default from config) ---
        cnn_channels_param = cnn_channels if cnn_channels is not None else config.DEFAULT_CNN_CHANNELS
        kernel_size_param = kernel_size if kernel_size is not None else config.DEFAULT_KERNEL_SIZE
        lstm_hidden_param = lstm_hidden if lstm_hidden is not None else config.DEFAULT_LSTM_HIDDEN_SIZE
        lstm_layers_param = lstm_layers if lstm_layers is not None else config.DEFAULT_LSTM_NUM_LAYERS
        lstm_bidir_param = lstm_bidir if lstm_bidir is not None else config.DEFAULT_LSTM_BIDIRECTIONAL
        dropout_rate_param = dropout_rate if dropout_rate is not None else config.DEFAULT_DROPOUT_RATE

        self.num_features = num_features
        # Define CNN channel progression
        self.cnn_channels_list = [num_features] + cnn_channels_param

        # --- Build CNN Feature Extractor Layers ---
        # These layers process each flight's features somewhat independently before LSTM
        cnn_layers = []
        print(f"Building SimAM_CNN_LSTM_Model with CNN channels: {self.cnn_channels_list}, LSTM hidden: {lstm_hidden_param}, layers: {lstm_layers_param}, dropout: {dropout_rate_param}")
        for i in range(len(self.cnn_channels_list) - 1):
            in_c = self.cnn_channels_list[i]
            out_c = self.cnn_channels_list[i+1]
            # Basic Conv Block
            cnn_layers.append(ConvBlock(
                in_c, out_c,
                kernel_size=kernel_size_param,
                dropout_rate=dropout_rate_param,
                padding='same'
            ))
            # Add SimAM attention after each ConvBlock
            cnn_layers.append(SimAM())
            print(f"  Added ConvBlock({in_c}, {out_c}) + SimAM()")

        self.feature_extractor = nn.Sequential(*cnn_layers)

        # --- LSTM Layer ---
        # Input size to LSTM is the number of channels output by the last CNN layer
        final_cnn_channels = self.cnn_channels_list[-1]
        self.lstm = StandardLSTM(
            input_size=final_cnn_channels,
            hidden_size=lstm_hidden_param,
            num_layers=lstm_layers_param,
            bidirectional=lstm_bidir_param,
            # Pass overall dropout rate for dropout between LSTM layers (if num_layers > 1)
            dropout_rate=dropout_rate_param if lstm_layers_param > 1 else 0.0
        )
        print(f"  Added LSTM(input={final_cnn_channels}, hidden={lstm_hidden_param}, layers={lstm_layers_param}, bidir={lstm_bidir_param})")


        # --- Classifier Head ---
        # Input size is determined by LSTM hidden size and directionality
        lstm_output_features = lstm_hidden_param * 2 if lstm_bidir_param else lstm_hidden_param
        self.classifier = nn.Linear(lstm_output_features, num_classes)
        print(f"  Added Classifier Head: Linear({lstm_output_features}, {num_classes})")


    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, SeqLen=3, Features).
        Returns:
            torch.Tensor: Output logits of shape (Batch, NumClasses).
        """
        # Input x: (B, S=3, F)
        batch_size, seq_len, _ = x.shape

        # Apply CNN feature extractor to each time step independently.
        # Reshape for Conv1d: (Batch * SeqLen, Features, 1) - Treat each flight as separate sample with SeqLen=1
        # OR apply Conv1d across sequence: (Batch, Features, SeqLen)
        # Let's apply across sequence, as SimAM/CBAM expect spatial/sequential dimension.
        x_permuted = x.permute(0, 2, 1) # (B, F, S=3)

        # Pass through CNN blocks + SimAM
        # Input: (B, F, S=3), Output: (B, C_last, S=3)
        cnn_features = self.feature_extractor(x_permuted)

        # Prepare features for LSTM: (Batch, SeqLen, Features=C_last)
        lstm_input = cnn_features.permute(0, 2, 1) # (B, S=3, C_last)

        # Pass sequence through LSTM
        # lstm_output contains hidden states for each step: (B, S=3, H*dirs)
        # final_hidden is the final hidden state used for classification: (B, H*dirs)
        lstm_output, final_hidden = self.lstm(lstm_input)

        # Use the final hidden state from the LSTM for classification
        # Input: (B, H*dirs), Output: (B, NumClasses)
        logits = self.classifier(final_hidden)
        return logits
