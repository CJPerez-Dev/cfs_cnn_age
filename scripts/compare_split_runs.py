#!/usr/bin/env python3
"""Compare MIL metrics from two CNN+MIL runs (e.g. age-stratified vs random subject split).

Reads ``cnn_mil_comparison_<run_tag>.json`` from full ``--model-mode both`` runs (``mil`` block only).

By default plots **subject-level** test MAE and R² only. For MIL eval with **one bag per test subject**,
``test_mae`` (bag-level mean error) and ``subject_mae`` are almost the same, so showing both is redundant.

Use ``--include-bag-level`` to also plot bag-level vs subject columns (older 4-group layout).

Usage:
    python scripts/compare_split_runs.py \\
        --a output/run_a/cnn_mil_comparison_run_a.json \\
        --b output/run_b/cnn_mil_comparison_run_b.json \\
        --label-a "Age-stratified" --label-b "Random split" \\
        -o output/split_strategy_comparison.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DEFAULT_TITLE = "CNN+MIL comparison: Age-stratified vs random subject split"


def _load_json(path: str) -> dict[str, Any]:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not a file: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_metric(branch: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(branch, dict):
        return None
    v = branch.get(key)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _split_hint(payload: dict[str, Any]) -> str:
    br = payload.get("mil")
    if isinstance(br, dict) and "age_stratified_split_applied" in br:
        on = bool(br.get("age_stratified_split_applied"))
        return "age-stratified" if on else "random (not age-stratified)"
    return "unknown"


def _collect_mil(payload: dict[str, Any]) -> dict[str, float | None]:
    branch = payload.get("mil")
    if not isinstance(branch, dict):
        return {k: None for k in ("test_mae", "subject_mae", "test_r2", "subject_r2")}
    return {
        "test_mae": _get_metric(branch, "test_mae"),
        "subject_mae": _get_metric(branch, "subject_mae"),
        "test_r2": _get_metric(branch, "test_r2"),
        "subject_r2": _get_metric(branch, "subject_r2"),
    }


def _panel_two_runs(
    ax,
    v_a: float | None,
    v_b: float | None,
    ylabel: str,
    title: str,
    label_a: str,
    label_b: str,
) -> None:
    heights = [
        float(v_a) if v_a is not None else np.nan,
        float(v_b) if v_b is not None else np.nan,
    ]
    ax.bar(
        [0, 1],
        heights,
        width=0.55,
        color=["#4E79A7", "#F28E2B"],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b], fontsize=10, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.set_xlim(-0.55, 1.55)
    ax.tick_params(axis="y", labelsize=10)


def _panel_grouped_metrics(
    ax,
    metric_keys: tuple[str, ...],
    xlabels: tuple[str, ...],
    ylabel: str,
    title: str,
    vals_a: dict[str, float | None],
    vals_b: dict[str, float | None],
) -> None:
    x = np.arange(len(metric_keys))
    width = 0.38
    ha = [vals_a.get(k) for k in metric_keys]
    hb = [vals_b.get(k) for k in metric_keys]
    heights_a = [float(v) if v is not None else np.nan for v in ha]
    heights_b = [float(v) if v is not None else np.nan for v in hb]

    ax.bar(x - width / 2, heights_a, width, color="#4E79A7", edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, heights_b, width, color="#F28E2B", edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.margins(x=0.12)
    ax.tick_params(axis="y", labelsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MIL metrics from two cnn_mil_comparison_*.json files (MIL stage only)."
    )
    parser.add_argument(
        "--a",
        dest="path_a",
        required=True,
        help="Path to first cnn_mil_comparison_*.json.",
    )
    parser.add_argument(
        "--b",
        dest="path_b",
        required=True,
        help="Path to second cnn_mil_comparison_*.json.",
    )
    parser.add_argument(
        "--label-a",
        default=None,
        help="X-axis label for --a (default: split hint).",
    )
    parser.add_argument(
        "--label-b",
        default=None,
        help="X-axis label for --b (default: split hint).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="split_strategy_comparison.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help=f"Figure title (default: {DEFAULT_TITLE!r}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution (default: 150).",
    )
    parser.add_argument(
        "--include-bag-level",
        action="store_true",
        help=(
            "Plot bag-level (test_*) and subject-level metrics side-by-side in each panel. "
            "For MIL with one eval bag per subject, test_mae and subject_mae are usually nearly identical."
        ),
    )
    args = parser.parse_args()

    pa = _load_json(args.path_a)
    pb = _load_json(args.path_b)

    for name, p in (("a", pa), ("b", pb)):
        if "mil" not in p or not isinstance(p.get("mil"), dict):
            raise ValueError(
                f"File --{name} must be a cnn_mil_comparison JSON with a non-empty 'mil' object."
            )

    label_a = args.label_a or f"A: {_split_hint(pa)}"
    label_b = args.label_b or f"B: {_split_hint(pb)}"

    mil_a, mil_b = _collect_mil(pa), _collect_mil(pb)

    fig_title = args.title if args.title is not None else DEFAULT_TITLE

    if args.include_bag_level:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=False)
        _panel_grouped_metrics(
            axes[0],
            ("test_mae", "subject_mae"),
            ("Bag-level MAE", "Subject MAE"),
            "MAE (years)",
            "MIL — MAE",
            mil_a,
            mil_b,
        )
        _panel_grouped_metrics(
            axes[1],
            ("test_r2", "subject_r2"),
            ("Bag-level R²", "Subject R²"),
            "R²",
            "MIL — R²",
            mil_a,
            mil_b,
        )
        legend_handles = [
            Patch(facecolor="#4E79A7", edgecolor="white", linewidth=0.6, label=label_a),
            Patch(facecolor="#F28E2B", edgecolor="white", linewidth=0.6, label=label_b),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            ncol=2,
            frameon=True,
            fontsize=10,
            borderaxespad=0.5,
        )
        footer = (
            f"Files: {os.path.basename(args.path_a)}  |  {os.path.basename(args.path_b)}\n"
            "Lower MAE is better; higher R² is better. MIL stage only. "
            "Bag-level vs subject metrics often match when eval uses one bag per subject."
        )
        tight_rect = (0, 0.10, 1, 0.82)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), constrained_layout=False)
        _panel_two_runs(
            axes[0],
            mil_a.get("subject_mae"),
            mil_b.get("subject_mae"),
            "MAE (years)",
            "MIL — subject-level test MAE",
            label_a,
            label_b,
        )
        _panel_two_runs(
            axes[1],
            mil_a.get("subject_r2"),
            mil_b.get("subject_r2"),
            "R²",
            "MIL — subject-level test R²",
            label_a,
            label_b,
        )
        footer = (
            f"Files: {os.path.basename(args.path_a)}  |  {os.path.basename(args.path_b)}\n"
            "Subject-level metrics on the held-out test split (one bag per subject). "
            "Use --include-bag-level to also show bag-level test_* columns."
        )
        tight_rect = (0, 0.10, 1, 0.92)

    fig.suptitle(fig_title, fontsize=14, y=0.98)
    fig.text(0.5, 0.02, footer, ha="center", fontsize=8, color="#333333")
    plt.tight_layout(rect=tight_rect)

    out = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
