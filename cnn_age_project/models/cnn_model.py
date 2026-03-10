"""Model architectures used by the EEG age-prediction pipeline."""

import torch.nn as nn


class EEGCNN(nn.Module):
    """1D CNN regressor that predicts age from single-channel EEG windows."""

    def __init__(self, input_length):
        """Initialize convolutional feature extractor and regression head.

        Args:
            input_length (int): Input window length in samples (kept for API clarity).

        Returns:
            None
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
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
