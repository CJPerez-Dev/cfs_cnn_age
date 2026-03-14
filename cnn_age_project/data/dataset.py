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


def sample_subject_pseudo_bag_indices(
    sorted_indices,
    offsets,
    counts,
    subject_code,
    rng,
    min_windows=256,
    max_windows=500,
    allow_replacement_when_small=True,
    sampling_strategy="random",
):
    """Sample one stochastic pseudo-bag for a specific subject.

    Args:
        sorted_indices (np.ndarray): Train indices sorted by subject code.
        offsets (np.ndarray): Start/end offsets into ``sorted_indices`` per subject.
        counts (np.ndarray): Number of windows available per subject.
        subject_code (int): Subject code whose bag should be sampled.
        rng (np.random.Generator): Random generator used for sampling.
        min_windows (int): Minimum sampled bag size.
        max_windows (int): Maximum sampled bag size.
        allow_replacement_when_small (bool): If True and subject has fewer than
            sampled bag size windows, sample with replacement.

    Returns:
        np.ndarray: Global indices for sampled pseudo-bag.
    """
    if min_windows <= 0 or max_windows <= 0:
        raise ValueError("min_windows and max_windows must be positive.")
    if min_windows > max_windows:
        raise ValueError("min_windows must be <= max_windows.")
    strategy = str(sampling_strategy).strip().lower()
    if strategy not in {"random", "sequential"}:
        raise ValueError("sampling_strategy must be either 'random' or 'sequential'.")

    n_items = int(counts[subject_code])
    if n_items <= 0:
        return np.array([], dtype=np.int64)

    start = int(offsets[subject_code])
    end = int(offsets[subject_code + 1])
    subject_indices = sorted_indices[start:end]

    target_bag_size = int(rng.integers(min_windows, max_windows + 1))

    if n_items >= target_bag_size:
        if strategy == "sequential":
            start_idx = int(rng.integers(0, n_items - target_bag_size + 1))
            return np.asarray(subject_indices[start_idx : start_idx + target_bag_size], dtype=np.int64)
        rel = rng.choice(n_items, size=target_bag_size, replace=False)
        return np.asarray(subject_indices[rel], dtype=np.int64)

    if allow_replacement_when_small:
        if strategy == "sequential":
            start_idx = int(rng.integers(0, n_items))
            rolled = np.roll(subject_indices, -start_idx)
            return np.resize(rolled, target_bag_size).astype(np.int64, copy=False)
        rel = rng.choice(n_items, size=target_bag_size, replace=True)
        return np.asarray(subject_indices[rel], dtype=np.int64)

    return np.asarray(subject_indices, dtype=np.int64)


def sample_epoch_subject_pseudo_bags(
    sorted_indices,
    offsets,
    counts,
    rng,
    min_windows=256,
    max_windows=500,
    allow_replacement_when_small=True,
    sampling_strategy="random",
):
    """Sample one stochastic pseudo-bag per subject for an epoch.

    Args:
        sorted_indices (np.ndarray): Train indices sorted by subject code.
        offsets (np.ndarray): Start/end offsets into ``sorted_indices`` per subject.
        counts (np.ndarray): Number of windows available per subject.
        rng (np.random.Generator): Random generator used for sampling.
        min_windows (int): Minimum sampled bag size.
        max_windows (int): Maximum sampled bag size.
        allow_replacement_when_small (bool): If True, small-subject bags are
            upsampled with replacement to the sampled bag size.

    Returns:
        list[tuple[int, np.ndarray]]: ``[(subject_code, bag_indices), ...]``.
    """
    bags = []
    for subject_code, n_items in enumerate(counts):
        if int(n_items) <= 0:
            continue
        bag_indices = sample_subject_pseudo_bag_indices(
            sorted_indices=sorted_indices,
            offsets=offsets,
            counts=counts,
            subject_code=subject_code,
            rng=rng,
            min_windows=min_windows,
            max_windows=max_windows,
            allow_replacement_when_small=allow_replacement_when_small,
            sampling_strategy=sampling_strategy,
        )
        if bag_indices.size > 0:
            bags.append((subject_code, bag_indices))

    rng.shuffle(bags)
    return bags


def iter_subject_pseudo_bag_batches(
    x_mem,
    y_mem,
    pseudo_bags,
    batch_size,
    x_mean=0.0,
    x_std=1.0,
):
    """Yield MIL bag mini-batches from sampled subject pseudo-bags.

    Args:
        x_mem (np.memmap | np.ndarray): Input windows array.
        y_mem (np.memmap | np.ndarray): Window-level target values.
        pseudo_bags (list[tuple[int, np.ndarray]]): Subject pseudo-bag entries
            as ``(subject_code, bag_indices)``.
        batch_size (int): Number of bags per mini-batch.
        x_mean (float): Mean used for input normalization.
        x_std (float): Std used for input normalization.

    Yields:
        tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
            ``(x_bags, y_bags, subject_codes, bag_mask)`` where ``x_bags`` is
            shaped ``(batch, max_instances, 1, window_len)``, ``y_bags`` is
            shaped ``(batch,)``, and ``bag_mask`` marks valid windows with shape
            ``(batch, max_instances)``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(pseudo_bags) == 0:
        return

    for start in range(0, len(pseudo_bags), batch_size):
        end = min(start + batch_size, len(pseudo_bags))
        batch_entries = pseudo_bags[start:end]

        bag_arrays = []
        bag_targets = []
        bag_subject_codes = []

        for subject_code, bag_indices in batch_entries:
            bag_indices = np.asarray(bag_indices, dtype=np.int64)
            if bag_indices.size == 0:
                continue

            bag_x = np.asarray(x_mem[bag_indices], dtype=np.float32)
            bag_x = (bag_x - x_mean) / x_std
            bag_arrays.append(bag_x)

            # Subject-level age is constant in expectation, but mean is robust
            # against accidental label inconsistencies in source metadata.
            bag_y = np.asarray(y_mem[bag_indices], dtype=np.float32)
            bag_targets.append(float(np.mean(bag_y)))
            bag_subject_codes.append(int(subject_code))

        if len(bag_arrays) == 0:
            continue

        bag_sizes = np.asarray([bag.shape[0] for bag in bag_arrays], dtype=np.int32)
        max_instances = int(bag_sizes.max())
        window_len = int(bag_arrays[0].shape[1])

        x_bags_np = np.zeros((len(bag_arrays), max_instances, window_len), dtype=np.float32)
        bag_mask_np = np.zeros((len(bag_arrays), max_instances), dtype=bool)
        for bag_idx, bag in enumerate(bag_arrays):
            n_instances = bag.shape[0]
            x_bags_np[bag_idx, :n_instances, :] = bag
            bag_mask_np[bag_idx, :n_instances] = True

        x_bags = torch.from_numpy(x_bags_np).unsqueeze(2)
        y_bags = torch.from_numpy(np.asarray(bag_targets, dtype=np.float32))
        subject_codes_np = np.asarray(bag_subject_codes, dtype=np.int32)
        bag_mask = torch.from_numpy(bag_mask_np)
        yield x_bags, y_bags, subject_codes_np, bag_mask
