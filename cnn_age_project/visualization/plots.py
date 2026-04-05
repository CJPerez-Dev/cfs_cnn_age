"""Plot-generation utilities for training and subject-level reports."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _mean_signed_error_by_true_age_band(
    true_years: np.ndarray,
    pred_years: np.ndarray,
    bin_width: float = 5.0,
):
    """Bin by true age and return (centers, mean(pred−true), labels) for non-empty bins."""
    ft = np.asarray(true_years, dtype=np.float64).ravel()
    fp = np.asarray(pred_years, dtype=np.float64).ravel()
    ok = np.isfinite(ft) & np.isfinite(fp)
    ft, fp = ft[ok], fp[ok]
    if ft.size == 0:
        return np.array([]), np.array([]), []
    err = fp - ft
    lo = max(0.0, float(np.floor(ft.min() / bin_width) * bin_width))
    hi = float(np.ceil(ft.max() / bin_width) * bin_width)
    if hi <= lo:
        hi = lo + bin_width
    edges = np.arange(lo, hi + bin_width, bin_width, dtype=np.float64)
    centers = []
    means = []
    for i in range(len(edges) - 1):
        e0, e1 = float(edges[i]), float(edges[i + 1])
        last = i == len(edges) - 2
        bin_m = (ft >= e0) & (ft <= e1) if last else (ft >= e0) & (ft < e1)
        if not np.any(bin_m):
            continue
        centers.append((e0 + e1) * 0.5)
        means.append(float(np.mean(err[bin_m])))
    return np.asarray(centers, dtype=np.float64), np.asarray(means, dtype=np.float64), float(bin_width)


def _plot_epoch_series(ax, values: list | np.ndarray, *, label: str, color: str, linestyle: str = "-") -> None:
    """Plot one per-epoch series, skipping non-finite points (e.g. skipped metric epochs)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return
    ep = np.arange(1, arr.size + 1, dtype=np.float64)
    m = np.isfinite(arr)
    if not np.any(m):
        return
    ax.plot(ep[m], arr[m], label=label, color=color, linestyle=linestyle, linewidth=2)


def save_training_report(
    run_output_dir,
    run_tag,
    report_save_name,
    train_maes,
    val_maes,
    r2_scores,
    baseline_mae,
    baseline_r2,
    test_mae,
    test_r2,
    final_targets,
    final_preds,
    subject_test_mae=None,
    subject_test_r2=None,
):
    """Create and save a four-panel training / held-out summary (MAE-focused).

    Panels:
        1. Train MAE vs epoch (and validation MAE when ``val_maes`` is non-empty).
        2. Train R² vs epoch (non-finite entries omitted).
        3. Predicted vs true age (sampled test windows).
        4. Mean signed error ``pred − true`` (years) by true-age band (diverging from 0).

    Args:
        run_output_dir (str): Output directory for this run.
        run_tag (str): Timestamp tag used in artifact names.
        report_save_name (str): Base filename for the report image.
        train_maes (list[float]): Per-epoch training MAE (years).
        val_maes (list[float] | None): Per-epoch validation MAE when a val split exists; else empty.
        r2_scores (list[float]): Per-epoch training R².
        baseline_mae (float): Baseline MAE (constant train-mean age).
        baseline_r2 (float): Baseline R² on test.
        test_mae (float): Final test MAE (window-level).
        test_r2 (float): Final test R² (window-level).
        final_targets (np.ndarray): Sampled true ages for scatter / age-band error profile.
        final_preds (np.ndarray): Sampled predictions for scatter / age-band error profile.
        subject_test_mae (float | None): Held-out subject-level test MAE (mean of per-subject means).
        subject_test_r2 (float | None): Held-out subject-level test R².

    Returns:
        str: Saved report image path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_mae, ax_r2, ax_scatter, ax_resid = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- MAE: train + optional validation (per-epoch val MAE = window-level for CNN, bag MAE for MIL) ---
    _plot_epoch_series(ax_mae, train_maes, label="Train MAE", color="#1f77b4")
    vm = val_maes if val_maes is not None else []
    vm_has_series = bool(vm) and any(np.isfinite(v) for v in vm)
    if vm_has_series and len(vm) == len(train_maes) and len(train_maes) > 0:
        _plot_epoch_series(ax_mae, vm, label="Validation MAE", color="#d62728", linestyle="--")
    elif vm_has_series and len(vm) > 0:
        _plot_epoch_series(ax_mae, vm, label="Validation MAE", color="#d62728", linestyle="--")

    try:
        if np.isfinite(baseline_mae):
            ax_mae.axhline(y=float(baseline_mae), color="#7f7f7f", linestyle=":", linewidth=1.5, label="Baseline MAE")
    except Exception:
        pass
    ax_mae.set_title("MAE (years) — training" + (" / validation" if vm_has_series else ""))
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE (years)")
    ax_mae.grid(True, alpha=0.35)
    ax_mae.legend(loc="upper right", framealpha=0.95)

    # --- R² ---
    _plot_epoch_series(ax_r2, r2_scores, label="Train R²", color="#2ca02c")
    try:
        if np.isfinite(baseline_r2):
            ax_r2.axhline(y=float(baseline_r2), color="#7f7f7f", linestyle=":", linewidth=1.5, label="Baseline R² (test)")
    except Exception:
        pass
    ax_r2.set_title("R² — training (window-level on train sample)")
    ax_r2.set_xlabel("Epoch")
    ax_r2.set_ylabel("R²")
    ax_r2.grid(True, alpha=0.35)
    ax_r2.legend(loc="lower right", framealpha=0.95)

    subj_line = ""
    if subject_test_mae is not None and np.isfinite(subject_test_mae):
        sr = float(subject_test_r2) if subject_test_r2 is not None and np.isfinite(subject_test_r2) else float("nan")
        subj_line = f"Test (subject): MAE = {subject_test_mae:.2f} y  |  R² = {sr:.4f}\n"
    summary = (
        f"Test (window): MAE = {test_mae:.2f} y  |  R² = {test_r2:.4f}\n"
        f"{subj_line}"
        f"Baseline: MAE = {baseline_mae:.2f} y  |  R² = {baseline_r2:.4f}\n"
        f"ΔMAE vs baseline = {baseline_mae - test_mae:.2f} y"
    )
    ax_r2.text(
        0.02,
        0.98,
        summary,
        transform=ax_r2.transAxes,
        bbox=dict(facecolor="white", alpha=0.88),
        verticalalignment="top",
        fontsize=9,
        family="monospace",
    )

    # --- Predicted vs true (test sample) ---
    if len(final_targets) > 0 and len(final_preds) > 0:
        ft = np.asarray(final_targets, dtype=np.float64).ravel()
        fp = np.asarray(final_preds, dtype=np.float64).ravel()
        ax_scatter.scatter(ft, fp, alpha=0.38, s=8, c="#0d47a1", edgecolors="none")
        min_age = float(min(ft.min(), fp.min()))
        max_age = float(max(ft.max(), fp.max()))
        ax_scatter.plot([min_age, max_age], [min_age, max_age], color="black", linewidth=1.5, linestyle="-", label="Identity")
        ax_scatter.legend(loc="upper left", fontsize=8)
    ax_scatter.set_title("Predicted vs true age (held-out test, window sample)")
    ax_scatter.set_xlabel("True age (years)")
    ax_scatter.set_ylabel("Predicted age (years)")
    ax_scatter.grid(True, alpha=0.35)
    ax_scatter.set_aspect("equal", adjustable="box")

    # --- Mean signed error by true age band (pred − true); 0 = unbiased in that band ---
    if len(final_targets) > 0 and len(final_preds) > 0:
        ft = np.asarray(final_targets, dtype=np.float64).ravel()
        fp = np.asarray(final_preds, dtype=np.float64).ravel()
        centers, means, bw = _mean_signed_error_by_true_age_band(ft, fp, bin_width=5.0)
        if centers.size > 0:
            w = float(bw) * 0.82
            for xc, ym in zip(centers, means):
                col = "#2ca02c" if ym >= 0.0 else "#9467bd"
                ax_resid.bar(xc, ym, width=w, color=col, alpha=0.88, edgecolor="white", linewidth=0.6, align="center")
            ax_resid.axhline(0.0, color="black", linewidth=1.3)
            ok = np.isfinite(ft) & np.isfinite(fp)
            if np.any(ok):
                overall = float(np.mean((fp - ft)[ok]))
                ax_resid.axhline(overall, color="#ff7f0e", linestyle="--", linewidth=1.0, label=f"Global mean error = {overall:+.2f} y")
            ax_resid.legend(loc="upper right", fontsize=8)
    ax_resid.set_title("Mean error by true age (pred − true, held-out sample)")
    ax_resid.set_xlabel("True age (years, band center)")
    ax_resid.set_ylabel("Mean error (years)")
    ax_resid.grid(True, axis="y", alpha=0.35)

    fig.suptitle(f"Training report — {run_tag}", fontsize=13, y=1.02)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    report_name, report_ext = os.path.splitext(report_save_name)
    report_path = os.path.join(run_output_dir, f"{report_name}_{run_tag}{report_ext}")
    plt.savefig(report_path, dpi=300)
    logger.info("Training report saved to: %s", report_path)
    plt.close(fig)
    return report_path


def save_subject_examples_report(run_output_dir, run_tag, subject_report_save_name, subject_examples):
    """Create and save a bar chart comparing true vs predicted age by subject.

    Args:
        run_output_dir (str): Output directory for this run.
        run_tag (str): Timestamp tag used in artifact names.
        subject_report_save_name (str): Base filename for the subject report image.
        subject_examples (list[dict[str, Any]]): Subject summary rows.

    Returns:
        str: Saved report path, or ``"not_generated"`` if no examples are available.
    """
    if len(subject_examples) <= 0:
        return "not_generated"

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    labels = [entry["subject_id"] for entry in subject_examples]
    pred_vals = [entry["pred_mean"] for entry in subject_examples]
    true_vals = [entry["true_mean"] for entry in subject_examples]
    windows_used = [entry["windows_used"] for entry in subject_examples]

    x_pos = np.arange(len(labels))
    width = 0.35
    ax2.bar(x_pos - width / 2, true_vals, width, label="True age")
    ax2.bar(x_pos + width / 2, pred_vals, width, label="Predicted age")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Age (years)")
    ax2.set_title("Random Test Subjects: True vs Predicted Age")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    subj_name, subj_ext = os.path.splitext(subject_report_save_name)
    subject_report_path = os.path.join(run_output_dir, f"{subj_name}_{run_tag}{subj_ext}")
    plt.savefig(subject_report_path, dpi=300)
    logger.info("Subject example report saved to: %s", subject_report_path)
    plt.close(fig2)
    return subject_report_path


def save_model_comparison_report(run_output_dir, run_tag, cnn_summary, mil_summary):
    """Save a metric comparison chart for Baseline vs CNN vs MIL.

    Includes window-level test metrics and subject-level test metrics when present
    in the run summaries (``subject_mae``, ``subject_r2``).

    Args:
        run_output_dir (str): Output directory for comparison artifacts.
        run_tag (str): Timestamp tag used in output filename.
        cnn_summary (dict[str, Any]): Summary payload for baseline CNN run.
        mil_summary (dict[str, Any]): Summary payload for MIL run.

    Returns:
        str: Saved comparison figure path.
    """
    # Use the baseline metrics stored in the per-run summary (baseline is the
    # constant train-mean age predictor). Baseline should be identical for CNN
    # and MIL runs given the same split; we take it from the CNN summary.
    baseline = {
        "test_loss": float(cnn_summary.get("baseline_loss", np.nan)),
        "test_mae": float(cnn_summary.get("baseline_mae", np.nan)),
        "test_r2": float(cnn_summary.get("baseline_r2", np.nan)),
        "subject_mae": float(cnn_summary.get("baseline_mae", np.nan)),
        "subject_r2": float(cnn_summary.get("baseline_r2", np.nan)),
    }

    metrics = [
        ("Test Loss (MSE)", "test_loss", False),
        ("Test MAE (window)", "test_mae", False),
        ("Subject MAE", "subject_mae", False),
        ("Test R² (window)", "test_r2", True),
        ("Subject R²", "subject_r2", True),
    ]

    baseline_values = [float(baseline.get(key, np.nan)) for _, key, _ in metrics]
    cnn_values = [float(cnn_summary.get(key, np.nan)) for _, key, _ in metrics]
    mil_values = [float(mil_summary.get(key, np.nan)) for _, key, _ in metrics]
    labels = [name for name, _, _ in metrics]

    x = np.arange(len(metrics), dtype=np.float32)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, baseline_values, width, label="Baseline", color="#BAB0AC")
    ax.bar(x, cnn_values, width, label="CNN", color="#4E79A7")
    ax.bar(x + width, mil_values, width, label="CNN + MIL", color="#F28E2B")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Metric Value")
    ax.set_title("Baseline vs CNN vs CNN+MIL (window + subject test metrics)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")

    summary_lines = []
    for label, key, higher_is_better in metrics:
        b_v = float(baseline.get(key, np.nan))
        cnn_v = float(cnn_summary.get(key, np.nan))
        mil_v = float(mil_summary.get(key, np.nan))
        if not (np.isfinite(b_v) and np.isfinite(cnn_v) and np.isfinite(mil_v)):
            continue
        # Compare improvements over baseline for each model.
        if higher_is_better:
            cnn_gain = cnn_v - b_v
            mil_gain = mil_v - b_v
        else:
            cnn_gain = b_v - cnn_v
            mil_gain = b_v - mil_v
        winner = "CNN+MIL" if mil_gain > cnn_gain else ("CNN" if cnn_gain > mil_gain else "Tie")
        summary_lines.append(
            f"{label}: baseline={b_v:.4f} | CNN={cnn_v:.4f} | CNN+MIL={mil_v:.4f} | better={winner}"
        )

    if summary_lines:
        ax.text(
            0.01,
            0.99,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85),
        )

    plt.tight_layout()
    path = os.path.join(run_output_dir, f"cnn_mil_comparison_{run_tag}.png")
    plt.savefig(path, dpi=300)
    logger.info("CNN vs MIL comparison report saved to: %s", path)
    plt.close(fig)
    return path
