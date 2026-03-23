#!/usr/bin/env python3
"""Plot age-stratum sampling diagnostics for CNN and MIL.

Creates one figure that compares:
1) Raw train-window distribution by age stratum
2) CNN weighted epoch sampling distribution by age stratum
3) MIL subject-weighted pseudo-bag sampling distribution by age stratum

Output: output/age_sampling_diagnostics_stratified.png
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

# Repository root (parent of ``scripts/``)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cnn_age_project.config import (  # noqa: E402
    CNN_SAMPLES_PER_EPOCH,
    MIL_ALLOW_REPLACEMENT_WHEN_SMALL,
    MIL_PSEUDO_BAG_MAX_WINDOWS,
    MIL_PSEUDO_BAG_MIN_WINDOWS,
    MIL_SUBJECT_DRAWS_PER_EPOCH,
    RANDOM_SEED,
    STRATIFY_AGE_BIN_YEARS,
    STRATIFY_TAIL_HIGH_MIN_AGE,
    STRATIFY_TAIL_LOW_MAX_AGE,
)
from cnn_age_project.data.age_strata import (  # noqa: E402
    build_mil_train_subject_inverse_frequency_probs,
    build_subject_code_stratum_lookup,
)
from cnn_age_project.data.dataset import (  # noqa: E402
    build_subject_group_index,
    compute_train_window_stratum_sample_weights,
    resolve_cnn_age_weighted_epoch_num_samples,
    sample_subject_pseudo_bag_indices,
    sample_weighted_train_epoch_indices,
)
from cnn_age_project.workflow.stages import load_data_context, setup_runtime_context  # noqa: E402


def _counts_by_stratum(indices: np.ndarray, subject_codes: np.ndarray, code_stratum_lookup: np.ndarray) -> np.ndarray:
    if indices is None or len(indices) == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64)
    codes = np.asarray(subject_codes[idx], dtype=np.int64)
    valid = (codes >= 0) & (codes < len(code_stratum_lookup))
    if not np.any(valid):
        return np.zeros(0, dtype=np.int64)
    strata = code_stratum_lookup[codes[valid]]
    max_st = int(np.max(strata))
    return np.bincount(strata, minlength=max_st + 1).astype(np.int64)


def _pad_to(a: np.ndarray, n: int) -> np.ndarray:
    if a.size >= n:
        return a
    out = np.zeros(n, dtype=a.dtype)
    out[: a.size] = a
    return out


def main() -> None:
    cli = SimpleNamespace(auto_split=True, age_stratified_split=True, validation_key=None)
    runtime_ctx = setup_runtime_context(cli)
    data_ctx = load_data_context(runtime_ctx, cli)
    rng = np.random.default_rng(RANDOM_SEED)

    # Subject-code -> age-stratum lookup aligned to current stratified split config.
    code_lookup = build_subject_code_stratum_lookup(
        data_ctx.subject_codebook,
        data_ctx.train_age_map,
        data_ctx.subject_stratum_merged,
        STRATIFY_TAIL_LOW_MAX_AGE,
        STRATIFY_TAIL_HIGH_MIN_AGE,
        STRATIFY_AGE_BIN_YEARS,
    )

    # 1) Raw train-window distribution.
    raw_counts = _counts_by_stratum(data_ctx.train_indices, data_ctx.subject_codes, code_lookup)

    # 2) CNN weighted sampling distribution (one synthetic epoch draw).
    train_idx = np.asarray(data_ctx.train_indices, dtype=np.int64)
    w = compute_train_window_stratum_sample_weights(train_idx, data_ctx.subject_codes, code_lookup)
    n_draws = resolve_cnn_age_weighted_epoch_num_samples(int(train_idx.size), CNN_SAMPLES_PER_EPOCH)
    cnn_epoch_idx = sample_weighted_train_epoch_indices(train_idx, w, n_draws, rng, replace=True)
    cnn_counts = _counts_by_stratum(cnn_epoch_idx, data_ctx.subject_codes, code_lookup)

    # 3) MIL subject-weighted pseudo-bag sampling distribution (one synthetic epoch).
    sorted_idx, offsets, counts = build_subject_group_index(
        train_indices=train_idx,
        subject_codes=data_ctx.subject_codes,
        n_subjects=len(data_ctx.subject_codebook),
    )
    eligible_codes, probs = build_mil_train_subject_inverse_frequency_probs(
        balanced_counts=counts,
        subject_codebook=data_ctx.subject_codebook,
        train_age_map=data_ctx.train_age_map,
        subject_stratum_merged=data_ctx.subject_stratum_merged,
        tail_low_max_age=STRATIFY_TAIL_LOW_MAX_AGE,
        tail_high_min_age=STRATIFY_TAIL_HIGH_MIN_AGE,
        bin_years=STRATIFY_AGE_BIN_YEARS,
    )
    n_subject_draws = int(MIL_SUBJECT_DRAWS_PER_EPOCH) if MIL_SUBJECT_DRAWS_PER_EPOCH is not None else int(eligible_codes.size)
    sampled_subject_codes = rng.choice(eligible_codes, size=n_subject_draws, replace=True, p=probs)

    mil_window_indices = []
    for sc in sampled_subject_codes:
        bag_idx = sample_subject_pseudo_bag_indices(
            sorted_indices=sorted_idx,
            offsets=offsets,
            counts=counts,
            subject_code=int(sc),
            rng=rng,
            min_windows=MIL_PSEUDO_BAG_MIN_WINDOWS,
            max_windows=MIL_PSEUDO_BAG_MAX_WINDOWS,
            allow_replacement_when_small=MIL_ALLOW_REPLACEMENT_WHEN_SMALL,
            sampling_strategy="random",
        )
        if bag_idx.size:
            mil_window_indices.append(np.asarray(bag_idx, dtype=np.int64))
    if mil_window_indices:
        mil_window_indices = np.concatenate(mil_window_indices)
    else:
        mil_window_indices = np.array([], dtype=np.int64)
    mil_counts = _counts_by_stratum(mil_window_indices, data_ctx.subject_codes, code_lookup)

    n_strata = max(raw_counts.size, cnn_counts.size, mil_counts.size)
    raw_counts = _pad_to(raw_counts, n_strata)
    cnn_counts = _pad_to(cnn_counts, n_strata)
    mil_counts = _pad_to(mil_counts, n_strata)

    def _to_pct(x: np.ndarray) -> np.ndarray:
        s = float(np.sum(x))
        return (x / s * 100.0) if s > 0 else np.zeros_like(x, dtype=np.float64)

    raw_pct = _to_pct(raw_counts)
    cnn_pct = _to_pct(cnn_counts)
    mil_pct = _to_pct(mil_counts)

    x = np.arange(n_strata)
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, raw_pct, width, label=f"Raw train windows (n={raw_counts.sum():,})", color="#1f77b4", alpha=0.85)
    ax.bar(x, cnn_pct, width, label=f"CNN sampled epoch (n={cnn_counts.sum():,})", color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, mil_pct, width, label=f"MIL sampled epoch windows (n={mil_counts.sum():,})", color="#2ca02c", alpha=0.85)

    ax.set_title("Age-Stratum Sampling Diagnostics (CNN + MIL)")
    ax.set_xlabel("Age stratum id (merged)")
    ax.set_ylabel("Share of windows in sampled set (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "age_sampling_diagnostics_stratified.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(
        "Counts | raw_train=%d cnn_sampled=%d mil_sampled_windows=%d strata=%d"
        % (int(raw_counts.sum()), int(cnn_counts.sum()), int(mil_counts.sum()), n_strata)
    )


if __name__ == "__main__":
    main()

