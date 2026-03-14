"""Model architectures used by the EEG age-prediction pipeline."""

import torch.nn as nn


class EEGCNN(nn.Module):
    """1D CNN regressor that predicts age from single-channel EEG windows.

    This implementation exposes two tunable parameters used by the tuner:
    - `embedding_dim`: number of output channels from the final conv block (encoder dim)
    - `dropout`: CNN dropout applied after each activation
    """

    def __init__(self, input_length, embedding_dim: int = 128, dropout: float = 0.0):
        """Initialize convolutional feature extractor and regression head.

        Args:
            input_length (int): Input window length in samples (kept for API clarity).
            embedding_dim (int): Final feature/channel dimension output by the encoder.
            dropout (float): Dropout probability applied after activations.

        Returns:
            None
        """
        super().__init__()

        dp = float(dropout)
        drop_layer = nn.Dropout(dp) if dp > 0.0 else nn.Identity()

        # Keep intermediate channel progression modest; final out channels = embedding_dim
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            drop_layer,
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            drop_layer,
            nn.MaxPool1d(2),
            nn.Conv1d(64, int(embedding_dim), kernel_size=7, padding=3),
            nn.ReLU(),
            drop_layer,
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )

        reg_drop = nn.Dropout(dp) if dp > 0.0 else nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(int(embedding_dim), 64),
            nn.ReLU(),
            reg_drop,
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """Run forward pass and return one scalar age prediction per sample.

        Args:
            x (torch.Tensor): Input tensor shaped ``(batch, 1, window_len)``.

        Returns:
            torch.Tensor: Predicted ages with shape ``(batch,)``.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x).squeeze()
