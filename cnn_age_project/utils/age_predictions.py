"""Post-processing for age predictions in year space (after denormalization)."""

from __future__ import annotations

import numpy as np
import torch

from cnn_age_project.config import CLIP_PREDICTED_AGE_AFTER_DENORM, MIN_PREDICTED_AGE_YEARS


def clip_predicted_ages_in_years(
    pred_years,
    *,
    enabled: bool | None = None,
    min_years: float | None = None,
):
    """Clamp predicted ages so they do not fall below ``min_years`` (default 0).

    Applied after converting model outputs to years (denormalization when used).
    When ``normalize_target`` is False, inputs are already in years and are
    clipped the same way.

    Args:
        pred_years: ``torch.Tensor`` or array-like of predicted ages in years.
        enabled: If False, return ``pred_years`` unchanged. If None, use
            ``CLIP_PREDICTED_AGE_AFTER_DENORM`` from config.
        min_years: Lower bound (inclusive). If None, use
            ``MIN_PREDICTED_AGE_YEARS`` from config.

    Returns:
        Same type as input for torch tensors; ``numpy.ndarray`` for array inputs.
    """
    clip_on = CLIP_PREDICTED_AGE_AFTER_DENORM if enabled is None else bool(enabled)
    if not clip_on:
        return pred_years
    lo = float(MIN_PREDICTED_AGE_YEARS if min_years is None else min_years)
    if isinstance(pred_years, torch.Tensor):
        return torch.clamp(pred_years, min=lo)
    arr = np.asarray(pred_years, dtype=np.float32)
    return np.maximum(arr, lo)
