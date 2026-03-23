#!/usr/bin/env python3
"""Create one simple 5-year-binned age distribution plot by split.

Usage:
    python scripts/plot_age_distribution_by_split.py
    python scripts/plot_age_distribution_by_split.py --with-stratified-bins
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

# Repository root (parent of ``scripts/``)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cnn_age_project.workflow.stages import load_data_context, setup_runtime_context  # noqa: E402


def _ages_for_indices(y_mem: np.ndarray, idx: np.ndarray | None) -> np.ndarray:
    if idx is None:
        return np.empty(0, dtype=np.float64)
    vals = np.asarray(y_mem[np.asarray(idx, dtype=np.int64)], dtype=np.float64)
    return vals[np.isfinite(vals)]


def _plot_hist(train_ages: np.ndarray, val_ages: np.ndarray, test_ages: np.ndarray, bin_edges: np.ndarray, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_kw = dict(bins=bin_edges, density=False, alpha=0.35, edgecolor="none")
    ax.hist(train_ages, label=f"Train (n={train_ages.size:,})", color="#1f77b4", **hist_kw)
    ax.hist(val_ages, label=f"Validation (n={val_ages.size:,})", color="#ff7f0e", **hist_kw)
    ax.hist(test_ages, label=f"Test (n={test_ages.size:,})", color="#2ca02c", **hist_kw)

    ax.set_title(title)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Window count")
    ax.set_xticks(bin_edges)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot split age distributions with 5-year bins.")
    parser.add_argument(
        "--with-stratified-bins",
        action="store_true",
        help="Also save a second plot using age-stratification style bin edges.",
    )
    args = parser.parse_args()

    cli = SimpleNamespace(
        auto_split=True,
        age_stratified_split=True,
        validation_key=None,
    )
    runtime_ctx = setup_runtime_context(cli)
    data_ctx = load_data_context(runtime_ctx, cli)

    train_ages = _ages_for_indices(data_ctx.y_mem, data_ctx.train_indices)
    test_ages = _ages_for_indices(data_ctx.y_mem, data_ctx.test_indices)
    val_ages = _ages_for_indices(data_ctx.y_mem, data_ctx.val_indices)

    all_ages = np.concatenate([train_ages, val_ages, test_ages])
    if all_ages.size == 0:
        raise RuntimeError("No finite ages found in split indices.")

    # Fixed 5-year bins across the observed range.
    age_min = float(np.floor(all_ages.min() / 5.0) * 5.0)
    age_max = float(np.ceil(all_ages.max() / 5.0) * 5.0)
    if age_max <= age_min:
        age_max = age_min + 5.0
    bin_edges = np.arange(age_min, age_max + 5.0, 5.0)

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "age_distribution_5Ybins.png")
    _plot_hist(train_ages, val_ages, test_ages, bin_edges, out_path, "Age Distribution by Split (5-Year Bins)")
    print(f"Saved: {out_path}")

    if args.with_stratified_bins:
        from cnn_age_project.config import (  # Local import to keep script startup simple.
            STRATIFY_AGE_BIN_YEARS,
            STRATIFY_TAIL_HIGH_MIN_AGE,
            STRATIFY_TAIL_LOW_MAX_AGE,
        )

        mids = np.arange(float(STRATIFY_TAIL_LOW_MAX_AGE), float(STRATIFY_TAIL_HIGH_MIN_AGE), float(STRATIFY_AGE_BIN_YEARS))
        strat_edges = [age_min]
        if age_min < float(STRATIFY_TAIL_LOW_MAX_AGE):
            strat_edges.append(float(STRATIFY_TAIL_LOW_MAX_AGE))
        strat_edges.extend(mids.tolist())
        if age_max > float(STRATIFY_TAIL_HIGH_MIN_AGE):
            strat_edges.append(float(STRATIFY_TAIL_HIGH_MIN_AGE))
        strat_edges.append(age_max)
        strat_edges = np.asarray(sorted(set(strat_edges)), dtype=float)
        if strat_edges.size < 2:
            strat_edges = np.asarray([age_min, age_max], dtype=float)

        strat_out = os.path.join("output", "age_distribution_5Ystratifiedbins.png")
        _plot_hist(
            train_ages,
            val_ages,
            test_ages,
            strat_edges,
            strat_out,
            "Age Distribution by Split (Stratification Bins)",
        )
        print(f"Saved: {strat_out}")

    print(
        f"Summary | train={train_ages.size:,} val={val_ages.size:,} test={test_ages.size:,} "
        f"| age_range=[{all_ages.min():.2f}, {all_ages.max():.2f}]"
    )


if __name__ == "__main__":
    main()
