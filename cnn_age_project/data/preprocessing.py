"""Normalization-statistics utilities for targets and input windows."""

import numpy as np


def compute_target_norm_stats(y_mem, train_indices, norm_eps=1e-6, chunk_size=1_000_000):
    """Compute mean/std for target ages using chunked access over train indices.

    Args:
        y_mem (np.memmap | np.ndarray): Target age array.
        train_indices (np.ndarray): Indices used to compute train statistics.
        norm_eps (float): Minimum variance guard for numerical stability.
        chunk_size (int): Number of rows processed per chunk.

    Returns:
        tuple[float, float]: ``(mean, std)`` of training targets.
    """
    total = 0.0
    total_sq = 0.0
    total_n = 0
    for start in range(0, train_indices.shape[0], chunk_size):
        end = min(start + chunk_size, train_indices.shape[0])
        chunk_indices = train_indices[start:end]
        vals = np.asarray(y_mem[chunk_indices], dtype=np.float64)
        total += float(vals.sum())
        total_sq += float(np.square(vals).sum())
        total_n += int(vals.shape[0])

    mean = total / max(1, total_n)
    var = (total_sq / max(1, total_n)) - (mean * mean)
    std = float(np.sqrt(max(var, norm_eps)))
    return float(mean), float(std)


def estimate_input_norm_stats(x_mem, train_indices, rng, norm_eps=1e-6, sample_windows=100_000, chunk_size=4096):
    """Estimate global input mean/std from a random subset of train windows.

    Args:
        x_mem (np.memmap | np.ndarray): Input windows array.
        train_indices (np.ndarray): Train indices eligible for sampling.
        rng (np.random.Generator): Random generator for subset selection.
        norm_eps (float): Minimum variance guard for numerical stability.
        sample_windows (int): Max windows sampled for estimation.
        chunk_size (int): Number of sampled windows processed per chunk.

    Returns:
        tuple[float, float]: ``(mean, std)`` of sampled inputs.
    """
    sample_n = min(sample_windows, train_indices.shape[0])
    if sample_n <= 0:
        return 0.0, 1.0

    sampled_idx = rng.choice(train_indices, size=sample_n, replace=False)

    total = 0.0
    total_sq = 0.0
    total_n = 0
    for start in range(0, sample_n, chunk_size):
        end = min(start + chunk_size, sample_n)
        idx_chunk = sampled_idx[start:end]
        x_chunk = np.asarray(x_mem[idx_chunk], dtype=np.float32)
        total += float(x_chunk.sum())
        total_sq += float(np.square(x_chunk).sum())
        total_n += int(x_chunk.size)

    mean = total / max(1, total_n)
    var = (total_sq / max(1, total_n)) - (mean * mean)
    std = float(np.sqrt(max(var, norm_eps)))
    return float(mean), float(std)
