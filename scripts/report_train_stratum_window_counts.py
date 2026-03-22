#!/usr/bin/env python3
"""Report training-window counts per age stratum and suggest ``samples_per_epoch``.

Uses the same split / stratification logic as the main pipeline (``load_data_context``),
then counts how many **train** windows fall in each merged stratum. The suggested epoch
size follows the heuristic:

    samples_per_epoch ~= n_min_windows * B

where ``n_min_windows`` is the smallest stratum's train-window count and ``B`` is the
number of strata that contain at least one train window. Under inverse-frequency
stratum weights with roughly equal mass per stratum, this targets
E[draws from smallest stratum] ~= n_min_windows per epoch.

Run from the repository root (with ``input/`` populated and venv activated)::

    python scripts/report_train_stratum_window_counts.py
    python scripts/report_train_stratum_window_counts.py --no-auto-split --validation-key AgeValidation_Key.csv
    python scripts/report_train_stratum_window_counts.py --quiet

This script loads memmaps read-only and does not train a model.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np

# Repository root (parent of ``scripts/``)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cnn_age_project.config import (  # noqa: E402
    STRATIFY_AGE_BIN_YEARS,
    STRATIFY_MIN_SUBJECTS_PER_STRATUM,
    STRATIFY_TAIL_HIGH_MIN_AGE,
    STRATIFY_TAIL_LOW_MAX_AGE,
)
from cnn_age_project.data.age_strata import (  # noqa: E402
    build_subject_code_stratum_lookup,
    n_strata,
    subject_stratum_from_merged_map,
)
from cnn_age_project.workflow.stages import load_data_context, setup_runtime_context  # noqa: E402


def _stratum_band_label(
    stratum_id: int,
    tail_low: float,
    tail_high: float,
    bin_years: float,
) -> str:
    """Human-readable band for *unmerged* stratum ids (best-effort)."""
    if stratum_id == 0:
        return f"age < {tail_low:g} (young tail)"
    n_mid = max(1, int(np.ceil((tail_high - tail_low) / bin_years)))
    last_id = 1 + n_mid
    if stratum_id == last_id:
        return f"age >= {tail_high:g} (old tail)"
    if 1 <= stratum_id < last_id:
        lo = tail_low + (stratum_id - 1) * bin_years
        hi = min(tail_low + stratum_id * bin_years, tail_high)
        return f"[{lo:g}, {hi:g})"
    return f"id {stratum_id} (outside default band layout - merged or custom)"


def _quiet_pipeline_loggers() -> None:
    for name in (
        "",
        "cnn_age_project",
        "cnn_age_project.data",
        "cnn_age_project.workflow",
        "cnn_age_project.workflow.stages",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _count_train_subjects_per_stratum(
    train_age_map: dict[str, float],
    subject_stratum_merged: dict[str, int] | None,
) -> dict[int, int]:
    out: dict[int, int] = {}
    for sid in train_age_map:
        st = subject_stratum_from_merged_map(
            sid,
            subject_stratum_merged,
            train_age_map,
            STRATIFY_TAIL_LOW_MAX_AGE,
            STRATIFY_TAIL_HIGH_MIN_AGE,
            STRATIFY_AGE_BIN_YEARS,
        )
        out[int(st)] = out.get(int(st), 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count train windows per age stratum and suggest samples_per_epoch.",
    )
    parser.add_argument(
        "--no-auto-split",
        action="store_false",
        dest="auto_split",
        default=True,
        help="Use AgeTraining_Key.csv / AgeTesting_Key.csv instead of auto-split.",
    )
    parser.add_argument(
        "--no-age-stratified-split",
        dest="age_stratified_split",
        action="store_false",
        default=True,
        help="Disable age-stratified auto-split (random subject shuffle; auto-split only).",
    )
    parser.add_argument(
        "--validation-key",
        type=str,
        default=None,
        help="Optional validation key CSV filename in input/ (same as main CLI).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce pipeline logging to warnings only.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if args.quiet:
        _quiet_pipeline_loggers()

    cli = SimpleNamespace(
        auto_split=bool(args.auto_split),
        age_stratified_split=bool(args.age_stratified_split),
        validation_key=args.validation_key,
    )

    runtime_ctx = setup_runtime_context(cli)
    data_ctx = load_data_context(runtime_ctx, cli)

    lookup = build_subject_code_stratum_lookup(
        data_ctx.subject_codebook,
        data_ctx.train_age_map,
        data_ctx.subject_stratum_merged,
        STRATIFY_TAIL_LOW_MAX_AGE,
        STRATIFY_TAIL_HIGH_MIN_AGE,
        STRATIFY_AGE_BIN_YEARS,
    )

    train_idx = np.asarray(data_ctx.train_indices, dtype=np.int64)
    codes = np.asarray(data_ctx.subject_codes[train_idx], dtype=np.int64)

    n_skipped_invalid_code = int(np.sum(codes < 0))
    strata = np.full(train_idx.shape[0], -1, dtype=np.int64)
    valid_code = codes >= 0
    strata[valid_code] = lookup[codes[valid_code]]

    n_skipped_unknown_stratum = int(np.sum((strata < 0) & valid_code))
    # lookup leaves 0 for subjects not in train_age_map; still a valid bucket
    use = strata >= 0
    strata_u = strata[use]

    max_id = int(strata_u.max()) if strata_u.size else -1
    counts = np.bincount(strata_u, minlength=max_id + 1).astype(np.int64)
    nonempty = np.where(counts > 0)[0]
    B = int(nonempty.size)
    n_train_windows = int(train_idx.shape[0])
    n_min = int(counts[nonempty].min()) if B > 0 else 0
    sweet_spot = int(n_min * B) if B > 0 else 0

    raw_strata_count = n_strata(
        STRATIFY_TAIL_LOW_MAX_AGE,
        STRATIFY_TAIL_HIGH_MIN_AGE,
        STRATIFY_AGE_BIN_YEARS,
    )

    print()
    print("=" * 72)
    print("Train-window counts by age stratum (pipeline-consistent)")
    print("=" * 72)
    print(f"Split: auto_split={args.auto_split} | age_stratified_split={args.age_stratified_split}")
    print(
        f"Stratify params: bin_years={STRATIFY_AGE_BIN_YEARS:g} | "
        f"young_tail <{STRATIFY_TAIL_LOW_MAX_AGE:g} | "
        f"old_tail >={STRATIFY_TAIL_HIGH_MIN_AGE:g} | "
        f"min_subjects_per_stratum={STRATIFY_MIN_SUBJECTS_PER_STRATUM}"
    )
    print(f"Raw (pre-merge) stratum layout count: {raw_strata_count} (young + middle bands + old tail)")
    print(f"Train windows total: {n_train_windows:,}")
    if n_skipped_invalid_code:
        print(f"  (skipped rows with subject_code < 0: {n_skipped_invalid_code:,})")
    if n_skipped_unknown_stratum:
        print(f"  (warning: unexpected strata<0 with valid code: {n_skipped_unknown_stratum:,})")
    print(f"Strata with >=1 train window: B = {B}")
    print()

    subj_counts = _count_train_subjects_per_stratum(data_ctx.train_age_map, data_ctx.subject_stratum_merged)
    merged = data_ctx.subject_stratum_merged is not None

    print(f"{'Stratum':<8} {'Train subj':>12} {'Train windows':>14}  Label (best-effort)")
    print("-" * 72)
    for st in nonempty:
        lab = (
            _stratum_band_label(int(st), STRATIFY_TAIL_LOW_MAX_AGE, STRATIFY_TAIL_HIGH_MIN_AGE, STRATIFY_AGE_BIN_YEARS)
            if not merged
            else f"merged stratum id {int(st)}"
        )
        sc = subj_counts.get(int(st), 0)
        print(f"{int(st):<8} {sc:>12,} {int(counts[st]):>14,}  {lab}")
    print("-" * 72)
    print(f"Smallest stratum train-window count n_min: {n_min:,}")
    print()
    print("Suggested 'representative dose' epoch size (heuristic):")
    print(f"  samples_per_epoch ~= n_min * B = {n_min:,} * {B} = {sweet_spot:,}")
    if n_train_windows > 0 and sweet_spot > 0:
        ratio = n_train_windows / sweet_spot
        print(f"  vs. full train window count: {n_train_windows:,} / {sweet_spot:,} ~= {ratio:.2f}x longer per epoch today")
    print()
    print("Notes:")
    print("  - Set ``cnn_samples_per_epoch`` in config.py (or ``--cnn-samples-per-epoch``) to use this in training.")
    print("  - After sparse-stratum merging, stratum ids are contiguous; labels are approximate if merged.")
    print("  - Re-run after changing split caches, key files, or stratify config in config.py.")
    print("=" * 72)


if __name__ == "__main__":
    main()
