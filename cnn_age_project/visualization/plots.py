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
        if final_preds.min() < 0:
            logger.warning("Visualization includes negative predictions (min=%.4f).", float(final_preds.min()))
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
