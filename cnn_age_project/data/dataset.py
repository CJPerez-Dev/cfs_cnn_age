"""Indexing, sampling, and mini-batch iteration helpers for memmap arrays."""

import numpy as np
import torch


def build_subject_group_index(train_indices, subject_codes, n_subjects):
    """Group train indices by subject and return sorted indices plus offsets/counts.

    Args:
        train_indices (np.ndarray): Global row indices belonging to train split.
        subject_codes (np.ndarray | np.memmap): Subject code per global row.
        n_subjects (int): Total number of subject codes in codebook.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Sorted train indices by subject,
        cumulative offsets, and per-subject counts.
    """
    train_subj_codes = np.asarray(subject_codes[train_indices], dtype=np.int32)
    valid_mask = train_subj_codes >= 0
    valid_indices = train_indices[valid_mask]
    valid_codes = train_subj_codes[valid_mask]

    order = np.argsort(valid_codes, kind="mergesort")
    sorted_indices = valid_indices[order]
    sorted_codes = valid_codes[order]

    counts = np.bincount(sorted_codes, minlength=n_subjects)
    offsets = np.zeros(n_subjects + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return sorted_indices, offsets, counts


def sample_balanced_train_indices(sorted_indices, offsets, counts, max_per_subject, rng):
    """Sample up to max_per_subject windows per subject and shuffle the result.

    Args:
        sorted_indices (np.ndarray): Train indices sorted by subject code.
        offsets (np.ndarray): Start/end offsets into ``sorted_indices`` per subject.
        counts (np.ndarray): Number of windows available per subject.
        max_per_subject (int): Maximum windows sampled per subject.
        rng (np.random.Generator): Random generator for reproducible sampling.

    Returns:
        np.ndarray: Shuffled sampled global indices.
    """
    pieces = []
    for subject_code, n_items in enumerate(counts):
        if n_items <= 0:
            continue
        start = int(offsets[subject_code])
        end = int(offsets[subject_code + 1])
        if n_items <= max_per_subject:
            pieces.append(sorted_indices[start:end])
        else:
            rel = rng.choice(n_items, size=max_per_subject, replace=False)
            pieces.append(sorted_indices[start + rel])

    if not pieces:
        return np.array([], dtype=np.int64)
    sampled = np.concatenate(pieces)
    rng.shuffle(sampled)
    return sampled


def iter_memmap_batches(x_mem, y_mem, indices, batch_size, x_mean=0.0, x_std=1.0):
    """Yield normalized tensor batches from memmap arrays for a set of indices.

    Args:
        x_mem (np.memmap | np.ndarray): Input windows array.
        y_mem (np.memmap | np.ndarray): Target values array.
        indices (np.ndarray): Row indices to iterate.
        batch_size (int): Batch size for yielded chunks.
        x_mean (float): Mean used for input normalization.
        x_std (float): Std used for input normalization.

    Yields:
        tuple[torch.Tensor, torch.Tensor]: ``(x_batch, y_batch)`` tensors.
    """
    n_samples = indices.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        x_batch_np = np.asarray(x_mem[batch_indices], dtype=np.float32)
        x_batch_np = (x_batch_np - x_mean) / x_std
        y_batch_np = np.asarray(y_mem[batch_indices], dtype=np.float32)
        x_batch = torch.from_numpy(x_batch_np).unsqueeze(1)
        y_batch = torch.from_numpy(y_batch_np)
        yield x_batch, y_batch
