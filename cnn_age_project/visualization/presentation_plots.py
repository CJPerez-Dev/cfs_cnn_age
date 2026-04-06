"""Presentation-quality figures comparing CNN vs CNN+MIL from run summary JSON."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Presentation-friendly defaults
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)


def load_run_summary(path: str) -> dict[str, Any]:
    """Load a ``cnn_run_summary_*.json`` file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _finite_epochs(y: list | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (epoch_indices, values) for finite entries."""
    if y is None:
        return np.array([]), np.array([])
    arr = np.asarray(y, dtype=np.float64)
    if arr.size == 0:
        return np.array([]), np.array([])
    idx = np.arange(len(arr))
    m = np.isfinite(arr)
    return idx[m], arr[m]


def plot_train_val_mae_side_by_side(
    cnn_summary: dict[str, Any],
    mil_summary: dict[str, Any],
    out_path: str,
) -> str:
    """Two panels: CNN train/val MAE (years) and CNN+MIL train/val MAE.

    Uses ``train_maes`` / ``val_maes`` from run summaries (window-level MAE per epoch).
    MIL runs currently log no per-epoch validation; the right panel shows train only
    unless ``val_maes`` is populated in future runs.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    def _plot_mae(ax, summary: dict[str, Any], title: str, *, tag: str) -> None:
        train = summary.get("train_maes") or []
        val_m = summary.get("val_maes")
        if val_m is None:
            val_m = []

        e, y = _finite_epochs(train)
        if e.size:
            ax.plot(e + 1, y, color="#4E79A7", linewidth=2, label=f"{tag} train MAE")

        if val_m and len(val_m) == len(train) and train:
            e2, y2 = _finite_epochs(np.asarray(val_m, dtype=np.float64))
            if e2.size:
                ax.plot(e2 + 1, y2, color="#E15759", linewidth=2, linestyle="--", label=f"{tag} validation MAE")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE (years)")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", framealpha=0.95)

    _plot_mae(ax0, cnn_summary, "CNN", tag="CNN")
    _plot_mae(ax1, mil_summary, "CNN+MIL", tag="CNN+MIL")

    vm = mil_summary.get("val_maes") or []
    if not vm:
        ax1.text(
            0.02,
            0.98,
            "MIL: no per-epoch validation curve in this run.",
            transform=ax1.transAxes,
            va="top",
            fontsize=9,
            color="#555555",
        )

    fig.suptitle("Training dynamics: MAE (years)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_train_mae_and_r2(
    cnn_summary: dict[str, Any],
    mil_summary: dict[str, Any],
    out_path: str,
    title: str = "Training-set MAE and R² (per epoch)",
) -> str:
    """Two panels: MAE (years) and R² over epochs for CNN vs MIL."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    mae_c = cnn_summary.get("train_maes") or []
    mae_m = mil_summary.get("train_maes") or []
    r2_c = cnn_summary.get("train_r2_scores") or []
    r2_m = mil_summary.get("train_r2_scores") or []

    for ax, data_c, data_m, ylabel, cnn_label, mil_label in (
        (ax0, mae_c, mae_m, "MAE (years)", "CNN train MAE", "CNN+MIL train MAE"),
        (ax1, r2_c, r2_m, "R²", "CNN train R²", "CNN+MIL train R²"),
    ):
        e, y = _finite_epochs(data_c)
        if e.size:
            ax.plot(e + 1, y, color="#4E79A7", linewidth=2, label=cnn_label)
        e2, y2 = _finite_epochs(data_m)
        if e2.size:
            ax.plot(e2 + 1, y2, color="#F28E2B", linewidth=2, label=mil_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", framealpha=0.95)

    ax1.set_xlabel("Epoch")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def _plot_metric_bars(
    cnn_summary: dict[str, Any],
    mil_summary: dict[str, Any],
    out_path: str,
    metrics: list[tuple[str, str, str]],
    ylabel: str,
    title: str,
    include_baseline: bool = True,
) -> str:
    """Shared bar chart for a small set of scalar metrics (CNN vs CNN+MIL vs baseline)."""
    baseline = None
    if include_baseline:
        baseline = {
            "test_mae": float(cnn_summary.get("baseline_mae", np.nan)),
            "test_r2": float(cnn_summary.get("baseline_r2", np.nan)),
            "subject_mae": float(cnn_summary.get("baseline_mae", np.nan)),
            "subject_r2": float(cnn_summary.get("baseline_r2", np.nan)),
        }

    n_met = len(metrics)
    x = np.arange(n_met, dtype=np.float64)
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 5))

    def bar_series(offset: float, summary: dict[str, Any], label: str, color: str):
        vals = []
        for _, key, _ in metrics:
            v = float(summary.get(key, np.nan))
            vals.append(v if np.isfinite(v) else np.nan)
        ax.bar(x + offset, vals, width, label=label, color=color)

    if baseline is not None:
        b_vals = []
        for _, key, _ in metrics:
            v = float(baseline.get(key, np.nan))
            b_vals.append(v if np.isfinite(v) else np.nan)
        ax.bar(x - width, b_vals, width, label="Baseline (train-mean age)", color="#BAB0AC")

    bar_series(0.0, cnn_summary, "CNN", "#4E79A7")
    bar_series(width, mil_summary, "CNN + MIL", "#F28E2B")

    ax.set_xticks(x + width / 2)
    if n_met == 1:
        ax.set_xticklabels([m[0] for m in metrics], rotation=0, ha="center")
    else:
        ax.set_xticklabels([m[0] for m in metrics], rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title + (" vs baseline" if include_baseline else ""))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        framealpha=0.95,
    )

    note = " | ".join(f"{m[0]}: {m[2]}" for m in metrics)
    ax.text(0.01, 0.99, note, transform=ax.transAxes, va="top", fontsize=8, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_test_subject_mae_only(
    cnn_summary: dict[str, Any],
    mil_summary: dict[str, Any],
    out_path: str,
    include_baseline: bool = True,
) -> str:
    """Subject-level MAE only (CNN vs CNN+MIL vs baseline)."""
    metrics = [
        ("Subject MAE (years)", "subject_mae", "lower is better"),
    ]
    return _plot_metric_bars(
        cnn_summary,
        mil_summary,
        out_path,
        metrics,
        ylabel="MAE (years)",
        title="Held-out subject-level MAE: CNN vs CNN+MIL",
        include_baseline=include_baseline,
    )


def plot_test_subject_r2(
    cnn_summary: dict[str, Any],
    mil_summary: dict[str, Any],
    out_path: str,
    include_baseline: bool = True,
) -> str:
    """Subject-level R² (CNN vs CNN+MIL vs constant-age baseline)."""
    metrics = [
        ("Subject R²", "subject_r2", "higher is better"),
    ]
    return _plot_metric_bars(
        cnn_summary,
        mil_summary,
        out_path,
        metrics,
        ylabel="R²",
        title="Held-out subject-level R²: CNN vs CNN+MIL",
        include_baseline=include_baseline,
    )


def plot_bootstrap_mae_ci(
    cnn_summary: dict[str, Any],
    mil_summary: dict[str, Any],
    out_path: str,
) -> str:
    """Point estimate + bootstrap CI for subject-level MAE (if CI present in summaries)."""
    ci_c = cnn_summary.get("bootstrap_mae_ci")
    ci_m = mil_summary.get("bootstrap_mae_ci")
    if not (ci_c and ci_m and len(ci_c) == 2 and len(ci_m) == 2):
        logger.info("Skipping bootstrap plot: missing bootstrap_mae_ci in one or both summaries.")
        return ""

    mae_c = float(cnn_summary.get("subject_mae", np.nan))
    mae_m = float(mil_summary.get("subject_mae", np.nan))
    if not (np.isfinite(mae_c) and np.isfinite(mae_m)):
        return ""

    low_c, high_c = float(ci_c[0]), float(ci_c[1])
    low_m, high_m = float(ci_m[0]), float(ci_m[1])
    means = [mae_c, mae_m]
    yerr = np.array([[mae_c - low_c, mae_m - low_m], [high_c - mae_c, high_m - mae_m]])
    labels = ["CNN", "CNN + MIL"]

    fig, ax = plt.subplots(figsize=(7, 5))
    xpos = np.arange(2)
    ax.bar(xpos, means, color=["#4E79A7", "#F28E2B"], width=0.5, yerr=yerr, capsize=6, ecolor="#333333")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Subject MAE (years)")
    ax.set_title("Subject-level MAE with bootstrap CI")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def render_presentation_bundle(
    cnn_summary: dict[str, Any],
    mil_summary: dict[str, Any],
    output_dir: str,
    prefix: str = "presentation",
) -> list[str]:
    """Write all comparison figures; return list of paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []
    paths.append(
        plot_train_val_mae_side_by_side(
            cnn_summary,
            mil_summary,
            os.path.join(output_dir, f"{prefix}_train_val_mae.png"),
        )
    )
    paths.append(
        plot_train_mae_and_r2(
            cnn_summary,
            mil_summary,
            os.path.join(output_dir, f"{prefix}_train_mae_r2.png"),
        )
    )
    paths.append(
        plot_test_subject_mae_only(
            cnn_summary,
            mil_summary,
            os.path.join(output_dir, f"{prefix}_test_subject_mae.png"),
        )
    )
    paths.append(
        plot_test_subject_r2(
            cnn_summary,
            mil_summary,
            os.path.join(output_dir, f"{prefix}_test_subject_r2.png"),
        )
    )
    boot = plot_bootstrap_mae_ci(
        cnn_summary,
        mil_summary,
        os.path.join(output_dir, f"{prefix}_bootstrap_test_mae.png"),
    )
    if boot:
        paths.append(boot)
    return paths


def plot_mil_best_bag_attention_figure(snapshot: dict[str, Any], out_path: str) -> str:
    """Scatter instance-level predicted age vs attention; second panel: attention vs window order.

    Gray dashed line: **mean-pooling reference** — if every window had the same weight, each
    would be ``1/n`` (attention still sums to 1). Compare learned weights to that baseline.

    Instance ages come from applying the bag regressor to each instance embedding (diagnostic
    only; training optimizes the attention-weighted bag prediction).

    Args:
        snapshot: Output of ``mil_best_bag_attention_snapshot`` from the trainer.
        out_path: PNG path.

    Returns:
        str: ``out_path`` if written; empty string if ``snapshot`` is empty.
    """
    if not snapshot:
        return ""

    inst = np.asarray(snapshot["instance_prediction_years"], dtype=np.float64)
    att = np.asarray(snapshot["attention_weights"], dtype=np.float64)
    win_idx = np.asarray(snapshot["window_index"], dtype=np.int32)
    true_age = float(snapshot["true_age_years"])
    bag_pred = float(snapshot["bag_prediction_years"])
    abs_err = float(snapshot["abs_error"])
    subj = snapshot.get("subject_id") or f"code={snapshot.get('subject_code', '?')}"
    seed = snapshot.get("diagnostic_seed", "")

    n = int(inst.size)
    uniform = 1.0 / max(1, n)
    # Attention is normalized to sum to 1; equal weights would be 1/n per window (mean pooling).
    baseline_label = f"Mean-pooling reference (equal weight / window)\n1/{n} ≈ {uniform:.4f} each"

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    ax0.scatter(inst, att, alpha=0.65, s=36, c="#4E79A7", edgecolors="white", linewidths=0.5, label="Learned attention")
    ax0.axhline(uniform, color="#BAB0AC", linestyle="--", linewidth=1, label=baseline_label)
    ax0.set_xlabel("Instance-level predicted age (years)")
    ax0.set_ylabel("Attention weight")
    ax0.set_title("Attention vs instance-level age (same MLP per window)")
    ax0.grid(True, alpha=0.35)
    ax0.legend(loc="best", fontsize=8)

    order = np.argsort(win_idx)
    ax1.plot(
        win_idx[order],
        att[order],
        "o-",
        color="#F28E2B",
        markersize=4,
        linewidth=1,
        label="Learned attention",
    )
    ax1.axhline(uniform, color="#BAB0AC", linestyle="--", linewidth=1, label=baseline_label)
    ax1.set_xlabel("Window index in bag (padded order)")
    ax1.set_ylabel("Attention weight")
    ax1.set_title("Attention vs window order")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best", fontsize=8)

    subtitle = (
        f"Subject: {subj} | true age: {true_age:.1f} y | bag pred: {bag_pred:.1f} y | "
        f"|error|: {abs_err:.2f} y | seed: {seed}"
    )
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved MIL attention figure to %s", out_path)
    return out_path


def demo_summaries() -> tuple[dict[str, Any], dict[str, Any]]:
    """Synthetic summaries for template plots when real JSON is unavailable."""
    rng = np.random.default_rng(42)
    n_c, n_m = 32, 24
    ep_c = np.arange(n_c)
    train_c = 400 * np.exp(-ep_c / 12) + 55 + rng.normal(0, 4, n_c)
    val_c = 420 * np.exp(-ep_c / 10) + 62 + rng.normal(0, 5, n_c)
    mae_c = 18 * np.exp(-ep_c / 14) + 5.2 + rng.normal(0, 0.15, n_c)
    val_mae_c = 19 * np.exp(-ep_c / 12) + 5.5 + rng.normal(0, 0.18, n_c)
    r2_c = 0.15 + 0.55 * (1 - np.exp(-ep_c / 15)) + rng.normal(0, 0.02, n_c)

    ep_m = np.arange(n_m)
    train_m = 380 * np.exp(-ep_m / 8) + 52 + rng.normal(0, 4, n_m)
    mae_m = 17 * np.exp(-ep_m / 12) + 4.7 + rng.normal(0, 0.12, n_m)
    r2_m = 0.18 + 0.58 * (1 - np.exp(-ep_m / 12)) + rng.normal(0, 0.02, n_m)

    base = {
        "baseline_mae": 12.5,
        "baseline_r2": 0.0,
        "baseline_loss": 180.0,
        "baseline_pred_age": 52.0,
        "test_loss": 78.0,
        "test_mae": 6.1,
        "test_r2": 0.62,
        "subject_mae": 5.9,
        "subject_r2": 0.65,
        "bootstrap_mae_ci": [5.7, 6.5],
    }
    cnn = {
        **base,
        "train_losses": train_c.tolist(),
        "val_losses": val_c.tolist(),
        "val_maes": val_mae_c.tolist(),
        "train_maes": mae_c.tolist(),
        "train_r2_scores": r2_c.tolist(),
        "model_label": "cnn",
    }
    mil = {
        **{k: v for k, v in base.items() if k != "bootstrap_mae_ci"},
        "test_loss": 72.0,
        "test_mae": 5.6,
        "test_r2": 0.66,
        "subject_mae": 5.4,
        "subject_r2": 0.69,
        "bootstrap_mae_ci": [5.2, 6.0],
        "train_losses": train_m.tolist(),
        "val_losses": [],
        "val_maes": [],
        "train_maes": mae_m.tolist(),
        "train_r2_scores": r2_m.tolist(),
        "model_label": "mil",
    }
    return cnn, mil
