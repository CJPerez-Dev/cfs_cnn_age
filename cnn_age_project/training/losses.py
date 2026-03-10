"""Loss-function factory for model optimization."""

import torch.nn as nn


def get_loss_function(use_huber_loss=True, huber_beta=1.0):
    """Return SmoothL1 (Huber) or MSE loss based on runtime settings.

    Args:
        use_huber_loss (bool): If True, returns ``SmoothL1Loss``.
        huber_beta (float): Beta parameter used by ``SmoothL1Loss``.

    Returns:
        nn.Module: Instantiated loss module.
    """
    if use_huber_loss:
        return nn.SmoothL1Loss(beta=float(huber_beta))
    return nn.MSELoss()
