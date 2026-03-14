"""Utilities for hyperparameter persistence and run-summary artifact writing."""

import json
import logging
import os

import numpy as np

from cnn_age_project.config import TUNING_RESULTS_FILE

logger = logging.getLogger(__name__)


def _to_serializable(value):
    """Recursively convert numpy values into JSON-serializable Python primitives.

    Args:
        value: Arbitrary Python/numpy value.

    Returns:
        Any: JSON-serializable equivalent.
    """
    if isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


def get_default_hyperparameters(
    lr,
    use_huber_loss,
    huber_beta,
    max_windows_per_subject_per_epoch,
    cnn_embedding_dim=128,
    cnn_dropout=0.0,
    mil_attention_dim=128,
    mil_attention_dropout=0.1,
    mil_pooling_type="gated",
    mil_bag_size=256,
    mil_sampling_strategy="random",
    mil_encoder_lr=1e-5,
    mil_head_lr=5e-5,
    mil_weight_decay=1e-2,
    mil_regressor_hidden_dim=64,
    mil_allow_replacement_when_small=True,
):
    """Return baseline hyperparameters used when no tuned config exists.

    Args:
        lr (float): Learning rate.
        use_huber_loss (bool): Whether to use Huber loss.
        huber_beta (float): Huber beta value.
        max_windows_per_subject_per_epoch (int): Per-subject sampling cap.

    Returns:
        dict[str, Any]: Default hyperparameter dictionary.
    """
    return {
        "learning_rate": lr,
        "use_huber_loss": use_huber_loss,
        "huber_beta": huber_beta,
        "max_windows_per_subject_per_epoch": max_windows_per_subject_per_epoch,
        "cnn_embedding_dim": int(cnn_embedding_dim),
        "cnn_dropout": float(cnn_dropout),
        "mil_attention_dim": mil_attention_dim,
        "mil_attention_dropout": mil_attention_dropout,
        "mil_pooling_type": mil_pooling_type,
        "mil_bag_size": mil_bag_size,
        "mil_sampling_strategy": mil_sampling_strategy,
        "mil_encoder_lr": mil_encoder_lr,
        "mil_head_lr": mil_head_lr,
        "mil_weight_decay": mil_weight_decay,
        "mil_regressor_hidden_dim": mil_regressor_hidden_dim,
        "mil_allow_replacement_when_small": bool(mil_allow_replacement_when_small),
    }


def load_best_hyperparameters_if_available(path, defaults):
    """Load saved hyperparameters from disk and merge onto defaults.

    Args:
        path (str): Path to best-hyperparameter JSON file.
        defaults (dict[str, Any]): Fallback defaults.

    Returns:
        dict[str, Any]: Loaded/merged hyperparameters.
    """
    if not os.path.exists(path):
        logger.info("No saved hyperparameter file found. Using defaults.")
        return defaults.copy()

    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        merged = defaults.copy()
        merged.update({k: loaded[k] for k in defaults.keys() if k in loaded})
        logger.info("Loaded saved hyperparameters from: %s", path)
        return merged
    except Exception as exc:
        logger.warning("Failed to load saved hyperparameters (%s). Using defaults.", exc)
        return defaults.copy()


def save_best_hyperparameters(path, hparams, tuning_results=None):
    """Persist selected best hyperparameters and optional trial-by-trial results.

    Args:
        path (str): Destination JSON path for best hyperparameters.
        hparams (dict[str, Any]): Best hyperparameters.
        tuning_results (list[dict[str, Any]] | None): Optional trial metrics.

    Returns:
        None
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(hparams), f, indent=2)
    logger.info("Saved best hyperparameters to: %s", path)

    if tuning_results is not None:
        # Save tuning results next to the best-hyperparameter file using a
        # matching basename to avoid clobbering a global tuning_results.json.
        base_name = os.path.splitext(os.path.basename(path))[0]
        results_filename = f"{base_name}_tuning_results.json"
        results_path = os.path.join(os.path.dirname(path), results_filename)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(tuning_results), f, indent=2)
        logger.info("Saved tuning trial results to: %s", results_path)


def build_tuning_candidates(defaults, model_mode="cnn"):
    """Generate a model-mode aware hyperparameter candidate grid.

    Args:
        defaults (dict[str, Any]): Default hyperparameter configuration.

    Returns:
        list[dict[str, Any]]: Candidate hyperparameter dictionaries.
    """
    mode = str(model_mode).strip().lower()
    candidates = []

    # CNN-focused dimensions.
    cnn_lrs = [1e-4, 3e-4, 8e-4]
    cnn_huber = [0.7, 1.0, 1.5]
    cnn_max_per_subj = [512, 1024]
    cnn_embedding_dims = [64, 128, 256]
    cnn_dropouts = [0.0, 0.1, 0.2, 0.3, 0.5]

    # MIL-focused dimensions.
    mil_attention_dims = [64, 128, 256]
    mil_attention_dropouts = [0.0, 0.1, 0.2]
    mil_pooling_types = ["gated", "mean"]
    mil_bag_sizes = [128, 256, 512]
    mil_sampling_strategies = ["random", "sequential"]
    mil_encoder_lrs = [1e-6, 1e-5]
    mil_head_lrs = [1e-4, 1e-3]
    mil_weight_decays = [1e-4, 1e-3, 1e-2]
    mil_regressor_hidden_dims = [64, 128]

    if mode in {"cnn", "both"}:
        for lr in cnn_lrs:
            for huber_beta in cnn_huber:
                for max_per_subj in cnn_max_per_subj:
                    for emb in cnn_embedding_dims:
                        for dp in cnn_dropouts:
                            base = defaults.copy()
                            base.update(
                                {
                                    "learning_rate": lr,
                                    "use_huber_loss": True,
                                    "huber_beta": huber_beta,
                                    "max_windows_per_subject_per_epoch": max_per_subj,
                                    "cnn_embedding_dim": int(emb),
                                    "cnn_dropout": float(dp),
                                }
                            )
                            candidates.append(base)

    if mode in {"mil", "both"}:
        for attention_dim in mil_attention_dims:
            for pooling in mil_pooling_types:
                # Keep mean pooling simpler by not exploding its dropout axis.
                dropout_grid = [0.0] if pooling == "mean" else mil_attention_dropouts
                for attention_dropout in dropout_grid:
                    for bag_size in mil_bag_sizes:
                        for sampling_strategy in mil_sampling_strategies:
                            for enc_lr in mil_encoder_lrs:
                                for head_lr in mil_head_lrs:
                                    for weight_decay in mil_weight_decays:
                                        for reg_hidden in mil_regressor_hidden_dims:
                                            base = defaults.copy()
                                            base.update(
                                                {
                                                    "mil_attention_dim": attention_dim,
                                                    "mil_attention_dropout": attention_dropout,
                                                    "mil_pooling_type": pooling,
                                                    "mil_bag_size": bag_size,
                                                    "mil_sampling_strategy": sampling_strategy,
                                                    "mil_encoder_lr": enc_lr,
                                                    "mil_head_lr": head_lr,
                                                    "mil_weight_decay": weight_decay,
                                                    "mil_regressor_hidden_dim": reg_hidden,
                                                }
                                            )
                                            candidates.append(base)

    if defaults not in candidates:
        candidates.append(defaults.copy())
    return candidates


def build_optuna_candidate(defaults, trial, model_mode="cnn"):
    """Build one hyperparameter candidate from an Optuna trial.

    Args:
        defaults (dict[str, Any]): Default hyperparameter dictionary.
        trial: Optuna trial object with suggest_* methods.
        model_mode (str): One of ``cnn``, ``mil``, or ``both``.

    Returns:
        dict[str, Any]: Candidate hyperparameter dictionary.
    """
    mode = str(model_mode).strip().lower()
    candidate = defaults.copy()

    if mode in {"cnn", "both"}:
        candidate.update(
            {
                "learning_rate": float(trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True)),
                "use_huber_loss": True,
                "huber_beta": float(trial.suggest_categorical("huber_beta", [0.7, 1.0, 1.5])),
                "max_windows_per_subject_per_epoch": int(
                    trial.suggest_categorical("max_windows_per_subject_per_epoch", [512, 1024])
                ),
                "cnn_embedding_dim": int(trial.suggest_categorical("cnn_embedding_dim", [64, 128, 256])),
                "cnn_dropout": float(trial.suggest_categorical("cnn_dropout", [0.0, 0.1, 0.2, 0.3, 0.5])),
            }
        )

    if mode in {"mil", "both"}:
        pooling_type = str(trial.suggest_categorical("mil_pooling_type", ["gated", "mean"]))
        if pooling_type == "mean":
            attention_dropout = 0.0
        else:
            attention_dropout = float(trial.suggest_categorical("mil_attention_dropout", [0.0, 0.1, 0.2]))

        candidate.update(
            {
                "mil_attention_dim": int(trial.suggest_categorical("mil_attention_dim", [64, 128, 256])),
                "mil_attention_dropout": attention_dropout,
                "mil_pooling_type": pooling_type,
                "mil_bag_size": int(trial.suggest_categorical("mil_bag_size", [128, 256, 512])),
                "mil_sampling_strategy": str(
                    trial.suggest_categorical("mil_sampling_strategy", ["random", "sequential"])
                ),
                "mil_encoder_lr": float(trial.suggest_categorical("mil_encoder_lr", [1e-6, 1e-5])),
                "mil_head_lr": float(trial.suggest_categorical("mil_head_lr", [1e-4, 1e-3])),
                "mil_weight_decay": float(trial.suggest_categorical("mil_weight_decay", [1e-4, 1e-3, 1e-2])),
                "mil_regressor_hidden_dim": int(trial.suggest_categorical("mil_regressor_hidden_dim", [64, 128])),
            }
        )

    return candidate


def save_run_summary(data_dir, summary_dict, run_tag):
    """Write machine-readable JSON and human-readable TXT run summaries.

    Args:
        data_dir (str): Destination directory for summary artifacts.
        summary_dict (dict[str, Any]): Summary payload.
        run_tag (str): Timestamp tag used in filenames.

    Returns:
        tuple[str, str]: ``(txt_path, json_path)``.
    """
    serializable = _to_serializable(summary_dict)

    txt_path = os.path.join(data_dir, f"cnn_run_summary_{run_tag}.txt")
    json_path = os.path.join(data_dir, f"cnn_run_summary_{run_tag}.json")

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(serializable, jf, indent=2)

    lines = [
        "EEG CNN Age Run Summary",
        "=" * 80,
        f"Run Timestamp: {serializable.get('run_timestamp', 'n/a')}",
        f"Duration Seconds: {serializable.get('duration_seconds', 'n/a')}",
        f"Device: {serializable.get('device', 'n/a')}",
        f"GPU: {serializable.get('gpu_name', 'n/a')}",
        "",
        "Config",
        "-" * 80,
        f"Batch Size: {serializable.get('batch_size', 'n/a')}",
        f"Max Epochs: {serializable.get('max_epochs', 'n/a')}",
        f"Learning Rate: {serializable.get('learning_rate', 'n/a')}",
        f"TF32 Enabled: {serializable.get('tf32_enabled', 'n/a')}",
        f"Torch Compile Enabled: {serializable.get('torch_compile_enabled', 'n/a')}",
        f"Triton Available: {serializable.get('triton_available', 'n/a')}",
        f"Early Stopping Enabled: {serializable.get('early_stopping_enabled', 'n/a')}",
        f"Bootstrap Enabled: {serializable.get('bootstrap_enabled', 'n/a')}",
        f"Bootstrap Iterations: {serializable.get('bootstrap_iterations', 'n/a')}",
        f"Bootstrap Confidence: {serializable.get('bootstrap_confidence', 'n/a')}",
        "",
        "Data Split",
        "-" * 80,
        f"Total Windows: {serializable.get('n_samples', 'n/a')}",
        f"Train Windows: {serializable.get('n_train_windows', 'n/a')}",
        f"Test Windows: {serializable.get('n_test_windows', 'n/a')}",
        f"Train Subjects (CSV): {serializable.get('n_train_subjects_csv', 'n/a')}",
        f"Test Subjects (CSV): {serializable.get('n_test_subjects_csv', 'n/a')}",
        f"Test Subjects (Evaluated): {serializable.get('n_test_subjects_evaluated', 'n/a')}",
        "",
        "Training",
        "-" * 80,
        f"Epochs Completed: {serializable.get('epochs_completed', 'n/a')}",
        f"Best Epoch: {serializable.get('best_epoch', 'n/a')}",
        f"Best Train Loss: {serializable.get('best_train_loss', 'n/a')}",
        "",
        "Window-Level Test Metrics",
        "-" * 80,
        f"Test Loss: {serializable.get('test_loss', 'n/a')}",
        f"Test R2: {serializable.get('test_r2', 'n/a')}",
        f"Test MAE: {serializable.get('test_mae', 'n/a')}",
        "",
        "Baseline (Constant Train-Mean Age)",
        "-" * 80,
        f"Baseline Prediction Age: {serializable.get('baseline_pred_age', 'n/a')}",
        f"Baseline Loss: {serializable.get('baseline_loss', 'n/a')}",
        f"Baseline R2: {serializable.get('baseline_r2', 'n/a')}",
        f"Baseline MAE: {serializable.get('baseline_mae', 'n/a')}",
        f"Delta MAE (Baseline - Model): {serializable.get('delta_mae', 'n/a')}",
        f"Delta R2 (Model - Baseline): {serializable.get('delta_r2', 'n/a')}",
        "",
        "Subject-Level Metrics",
        "-" * 80,
        f"Subject R2: {serializable.get('subject_r2', 'n/a')}",
        f"Subject MAE: {serializable.get('subject_mae', 'n/a')}",
        f"Bootstrap MAE CI: {serializable.get('bootstrap_mae_ci', 'n/a')}",
        f"Bootstrap R2 CI: {serializable.get('bootstrap_r2_ci', 'n/a')}",
        "",
        "Artifacts",
        "-" * 80,
        f"Model: {serializable.get('model_path', 'n/a')}",
        f"Training Report: {serializable.get('report_path', 'n/a')}",
        f"Subject Report: {serializable.get('subject_report_path', 'n/a')}",
        f"Summary JSON: {json_path}",
    ]

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write("\n".join(lines) + "\n")

    logger.info("Run summary saved to: %s", txt_path)
    logger.info("Run summary JSON saved to: %s", json_path)
    return txt_path, json_path


def save_model_comparison_summary(data_dir, run_tag, cnn_summary, mil_summary):
    """Write JSON/TXT summary comparing baseline CNN and MIL runs.

    Args:
        data_dir (str): Destination directory for comparison artifacts.
        run_tag (str): Timestamp tag used in comparison filenames.
        cnn_summary (dict[str, Any]): Per-run summary payload for CNN.
        mil_summary (dict[str, Any]): Per-run summary payload for MIL.

    Returns:
        tuple[str, str, dict[str, Any]]: ``(txt_path, json_path, payload)``.
    """
    metrics = [
        ("test_mae", False),
        ("test_r2", True),
        ("subject_mae", False),
        ("subject_r2", True),
        ("best_train_loss", False),
    ]

    deltas = {}
    for metric_name, _ in metrics:
        cnn_value = float(cnn_summary.get(metric_name, np.nan))
        mil_value = float(mil_summary.get(metric_name, np.nan))
        delta = mil_value - cnn_value
        deltas[metric_name] = {
            "cnn": cnn_value,
            "mil": mil_value,
            "delta_mil_minus_cnn": float(delta) if np.isfinite(delta) else np.nan,
        }

    payload = {
        "run_timestamp": run_tag,
        "cnn": _to_serializable(cnn_summary),
        "mil": _to_serializable(mil_summary),
        "deltas": _to_serializable(deltas),
    }

    json_path = os.path.join(data_dir, f"cnn_mil_comparison_{run_tag}.json")
    txt_path = os.path.join(data_dir, f"cnn_mil_comparison_{run_tag}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "CNN vs MIL Comparison",
        "=" * 80,
        f"Run Tag: {run_tag}",
        "",
    ]

    for metric_name, higher_is_better in metrics:
        cnn_value = deltas[metric_name]["cnn"]
        mil_value = deltas[metric_name]["mil"]
        delta = deltas[metric_name]["delta_mil_minus_cnn"]
        if np.isfinite(delta):
            if higher_is_better:
                winner = "MIL" if delta > 0 else ("CNN" if delta < 0 else "Tie")
            else:
                winner = "MIL" if delta < 0 else ("CNN" if delta > 0 else "Tie")
            lines.append(
                f"{metric_name}: CNN={cnn_value:.6f} | MIL={mil_value:.6f} | "
                f"delta(MIL-CNN)={delta:+.6f} | better={winner}"
            )
        else:
            lines.append(f"{metric_name}: unavailable")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Model comparison summary saved to: %s", txt_path)
    logger.info("Model comparison JSON saved to: %s", json_path)
    return txt_path, json_path, payload
