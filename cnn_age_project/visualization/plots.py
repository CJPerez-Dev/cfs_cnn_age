"""Plot-generation utilities for training and subject-level reports."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def save_training_report(
    run_output_dir,
    run_tag,
    report_save_name,
    train_losses,
    r2_scores,
    maes,
    baseline_loss,
    baseline_mae,
    baseline_r2,
    test_mae,
    final_targets,
    final_preds,
):
    """Create and save the multi-panel training/evaluation summary figure.

    Args:
        run_output_dir (str): Output directory for this run.
        run_tag (str): Timestamp tag used in artifact names.
        report_save_name (str): Base filename for the report image.
        train_losses (list[float]): Epoch loss history.
        r2_scores (list[float]): Epoch R² history.
        maes (list[float]): Epoch MAE history.
        baseline_loss (float): Baseline MSE.
        baseline_mae (float): Baseline MAE.
        baseline_r2 (float): Baseline R².
        test_mae (float): Final test MAE.
        final_targets (np.ndarray): Sampled target points for scatter plot.
        final_preds (np.ndarray): Sampled prediction points for scatter plot.

    Returns:
        str: Saved report image path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(train_losses, linewidth=2)
    axes[0, 0].set_title("Training Loss (MSE)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(True)
    try:
        axes[0, 0].axhline(y=baseline_loss, color="r", linestyle="--", linewidth=1, label="Baseline Loss (MSE)")
        axes[0, 0].legend(loc="upper right")
    except Exception:
        pass

    axes[0, 1].plot(r2_scores, linewidth=2)
    axes[0, 1].set_title("R² Score")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True)
    axes[0, 1].text(
        0.02,
        0.05,
        (
            f"Test MAE: {test_mae:.2f}\n"
            f"Baseline MAE: {baseline_mae:.2f}\n"
            f"Baseline Loss (MSE): {baseline_loss:.2f}\n"
            f"Baseline R²: {baseline_r2:.4f}\n"
            f"ΔMAE: {baseline_mae - test_mae:.2f}"
        ),
        transform=axes[0, 1].transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    try:
        if np.isfinite(baseline_r2):
            axes[0, 1].axhline(y=baseline_r2, color="tab:orange", linestyle="--", linewidth=1, label="Baseline R²")
            axes[0, 1].legend(loc="lower right")
    except Exception:
        pass

    axes[1, 0].plot(maes, linewidth=2)
    axes[1, 0].set_title("Mean Absolute Error")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Years")
    axes[1, 0].grid(True)
    try:
        axes[1, 0].axhline(y=baseline_mae, color="r", linestyle="--", linewidth=1, label="Baseline MAE")
        axes[1, 0].legend(loc="upper right")
    except Exception:
        pass

    if len(final_targets) > 0 and len(final_preds) > 0:
        axes[1, 1].scatter(final_targets, final_preds, alpha=0.15)
        min_age = min(final_targets.min(), final_preds.min())
        max_age = max(final_targets.max(), final_preds.max())
        axes[1, 1].plot([min_age, max_age], [min_age, max_age], linewidth=2)
        try:
            axes[1, 1].text(
                0.02,
                0.95,
                (
                    f"Baseline MSE: {baseline_loss:.2f}\n"
                    f"Baseline MAE: {baseline_mae:.2f}\n"
                    f"Baseline R²: {baseline_r2:.4f}"
                ),
                transform=axes[1, 1].transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
                verticalalignment="top",
            )
        except Exception:
            pass
    axes[1, 1].set_title("Predicted vs True Age")
    axes[1, 1].set_xlabel("True Age")
    axes[1, 1].set_ylabel("Predicted Age")
    axes[1, 1].grid(True)

    plt.tight_layout()
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
    }

    metrics = [
        ("Test Loss (MSE)", "test_loss", False),
        ("Test MAE", "test_mae", False),
        ("Test R²", "test_r2", True),
    ]

    baseline_values = [float(baseline.get(key, np.nan)) for _, key, _ in metrics]
    cnn_values = [float(cnn_summary.get(key, np.nan)) for _, key, _ in metrics]
    mil_values = [float(mil_summary.get(key, np.nan)) for _, key, _ in metrics]
    labels = [name for name, _, _ in metrics]

    x = np.arange(len(metrics), dtype=np.float32)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, baseline_values, width, label="Baseline", color="#BAB0AC")
    ax.bar(x, cnn_values, width, label="CNN", color="#4E79A7")
    ax.bar(x + width, mil_values, width, label="CNN + MIL", color="#F28E2B")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Metric Value")
    ax.set_title("Baseline vs CNN vs CNN+MIL (Test Metrics)")
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
