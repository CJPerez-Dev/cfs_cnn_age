"""CLI entrypoint for running the EEG CNN age-prediction workflow."""

import argparse
import logging
import warnings

from cnn_age_project.config import LOG_LEVEL
from cnn_age_project.utils.utils import configure_logging
from cnn_age_project.workflow.stages import execute_full_workflow

warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line options for standard run or tuning mode.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed CLI arguments containing tuning options.
    """
    parser = argparse.ArgumentParser(
        description="EEG age regression: CNN encoder + MIL subject-level model (optional CNN-only or MIL-only runs)."
    )
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning mode before final training")
    parser.add_argument(
        "--tune-and-train",
        action="store_true",
        help="After tuning completes, continue into full training/evaluation. By default, `--tune` stops after saving best hyperparameters.",
    )
    parser.add_argument("--tune-epochs", type=int, default=4, help="Epochs per tuning trial")
    parser.add_argument("--tune-max-trials", type=int, default=8, help="Maximum number of hyperparameter trials")
    parser.add_argument(
        "--tune-backend",
        type=str,
        choices=["grid", "optuna"],
        default="optuna",
        help="Hyperparameter tuner backend (default: optuna). Use 'grid' for shuffled grid search.",
    )
    parser.add_argument(
        "--optuna-sampler",
        type=str,
        choices=["tpe", "random"],
        default="tpe",
        help="Optuna sampler to use when --tune-backend optuna.",
    )
    parser.add_argument(
        "--optuna-pruner",
        type=str,
        choices=["none", "median"],
        default="median",
        help=(
            "Optuna pruner when --tune-backend optuna. ``median`` uses MedianPruner with intermediate "
            "metrics each epoch (val MAE when a val split exists). Use ``none`` to disable pruning."
        ),
    )
    parser.add_argument(
        "--optuna-startup-trials",
        type=int,
        default=10,
        help="Number of startup trials before TPE adapts (Optuna only).",
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=42,
        help="Random seed for Optuna sampler.",
    )
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help=(
            "Optuna RDB URL or path to a SQLite file on **shared** storage (e.g. /scratch/alpine/$USER/optuna/cnn.db). "
            "When set, multiple processes/jobs can load the same study (use the same --optuna-study-name). "
            "Each process still runs --tune-max-trials trials (e.g. 3 jobs × 16 = 48 trials total). "
            "Prefer `--tune` without `--tune-and-train` on workers, then one training job with --hparams-file."
        ),
    )
    parser.add_argument(
        "--optuna-study-name",
        type=str,
        default=None,
        help="Optuna study name when using --optuna-storage (default: --tune-name or cfs_cnn_optuna).",
    )
    parser.add_argument("--hparams-file", type=str, default=None, help="Path to a JSON file with hyperparameters to use for training (overrides saved defaults).")
    parser.add_argument(
        "--cnn-hparams-file",
        type=str,
        default=None,
        help="Optional CNN hyperparameters JSON (used when --model-mode both). If set, overrides --hparams-file for the CNN run only.",
    )
    parser.add_argument(
        "--mil-hparams-file",
        type=str,
        default=None,
        help="Optional MIL hyperparameters JSON (used when --model-mode both). If set, overrides --hparams-file for the MIL run only.",
    )
    parser.add_argument("--tune-name", type=str, default=None, help="Optional name to attach to tuning results/best-hparams output when --tune is used.")
    parser.add_argument(
        "--model-mode",
        type=str,
        choices=["cnn", "mil", "both"],
        default="both",
        help=(
            "Default `both`: train CNN then MIL (same run; subject-level prediction + CNN vs MIL comparison). "
            "`cnn` = window-level encoder only; `mil` = MIL fine-tune from --mil-pretrained-model or defaults/default_model.pt."
        ),
    )
    parser.add_argument(
        "--validation-key",
        type=str,
        default=None,
        help=(
            "Legacy (--no-auto-split) only: validation subject key CSV in input/ (e.g. AgeValidation_Key.csv). "
            "Ignored when auto-split is on (val fold comes from 70/15/15)."
        ),
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        default=True,
        help="Use 70/15/15 subject-level train/val/test split from AgeKey.csv or metadata (default).",
    )
    parser.add_argument(
        "--no-auto-split",
        action="store_false",
        dest="auto_split",
        help="Use AgeTraining_Key.csv and AgeTesting_Key.csv for train/test split instead of auto-split.",
    )
    parser.add_argument(
        "--no-age-stratified-split",
        dest="age_stratified_split",
        action="store_false",
        default=True,
        help="Disable age-stratified train/val/test subject split (auto-split only; uses random shuffle).",
    )
    parser.add_argument(
        "--no-age-weighted-window-sampling",
        dest="age_weighted_window_sampling",
        action="store_false",
        default=True,
        help="Disable inverse-frequency age-stratum sampling for CNN training windows (WeightedRandomSampler-style).",
    )
    parser.add_argument(
        "--cnn-samples-per-epoch",
        type=int,
        default=None,
        help=(
            "CNN age-weighted training: stochastic draws per epoch (default: config cnn_samples_per_epoch). "
            "Each epoch resamples with replacement using inverse-stratum weights. Use 0 to use the full train pool per epoch."
        ),
    )
    parser.add_argument(
        "--no-mil-inverse-frequency-subject-sampling",
        dest="mil_inverse_frequency_subject_sampling",
        action="store_false",
        default=True,
        help="Disable MIL train: inverse-frequency subject draws (use one bag per subject per epoch).",
    )
    parser.add_argument(
        "--mil-subject-draws-per-epoch",
        type=int,
        default=None,
        help=(
            "MIL (inverse-frequency train): number of subject pseudo-bags sampled per epoch. "
            "Default: config or one draw per eligible train subject. Overrides MIL hyperparameter JSON."
        ),
    )
    parser.add_argument(
        "--mil-subject-draws-multiplier",
        type=float,
        default=None,
        help=(
            "MIL (inverse-frequency train): set draws/epoch = round(multiplier × train subjects with ≥1 window), "
            "e.g. 2 for 2×. Overrides --mil-subject-draws-per-epoch and JSON."
        ),
    )

    parser.add_argument("--mil-finetune", action="store_true", help="Enable MIL Step 3 fine-tuning mode (unfreeze encoder + train full MIL model).")
    parser.add_argument(
        "--mil-pretrained-model",
        type=str,
        default=None,
        help="CNN checkpoint for MIL encoder init. If omitted: MIL-only uses defaults/default_model.pt when present; "
        "`--model-mode both` uses the CNN weights saved in the same run (after the CNN stage).",
    )
    parser.add_argument("--mil-attention-dim", type=int, default=None, help="Hidden size for MIL gated attention head.")
    parser.add_argument("--mil-attention-dropout", type=float, default=None, help="Dropout within MIL attention head.")
    parser.add_argument("--mil-pooling-type", type=str, choices=["gated", "mean"], default=None, help="MIL bag pooling type.")
    parser.add_argument("--mil-regressor-hidden-dim", type=int, default=None, help="Hidden size for MIL bag-level regressor.")
    parser.add_argument("--mil-regressor-dropout", type=float, default=None, help="Dropout in MIL bag regressor after hidden ReLU.")
    parser.add_argument("--mil-bag-batch-size", type=int, default=None, help="Number of pseudo-bags per optimizer step in MIL fine-tuning.")
    parser.add_argument(
        "--mil-pseudo-bag-min-windows",
        type=int,
        default=None,
        help="Minimum random pseudo-bag size per subject.",
    )
    parser.add_argument(
        "--mil-pseudo-bag-max-windows",
        type=int,
        default=None,
        help="Maximum random pseudo-bag size per subject.",
    )
    parser.add_argument(
        "--mil-sampling-strategy",
        type=str,
        choices=["random", "sequential"],
        default=None,
        help="Pseudo-bag sampling strategy for MIL training/evaluation.",
    )
    parser.add_argument(
        "--mil-allow-replacement-when-small",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow pseudo-bag sampling with replacement when a subject has fewer windows than requested.",
    )
    parser.add_argument("--mil-encoder-lr", type=float, default=None, help="Fine-tuning LR for pretrained CNN encoder parameters.")
    parser.add_argument("--mil-head-lr", type=float, default=None, help="Fine-tuning LR for MIL attention/predictor parameters.")
    parser.add_argument("--mil-weight-decay", type=float, default=None, help="Weight decay used by MIL fine-tuning AdamW optimizer.")
    parser.add_argument(
        "--mil-early-stopping-patience",
        type=int,
        default=None,
        help="MIL fine-tune: epochs with no val-loss improvement before stopping (default: config mil_early_stopping_patience).",
    )
    parser.add_argument(
        "--mil-early-stopping-min-epochs",
        type=int,
        default=None,
        help="MIL fine-tune: minimum epochs before early stopping can trigger (default: config mil_early_stopping_min_epochs).",
    )
    parser.add_argument(
        "--mil-early-stopping-monitor",
        type=str,
        choices=["auto", "train", "val"],
        default="auto",
        help="MIL fine-tune: early stopping on train loss, validation bag MAE (matches Optuna), or auto (val when a val split exists).",
    )
    return parser.parse_args()


def run():
    """Configure logging and execute the full training/evaluation workflow.

    Args:
        None

    Returns:
        None
    """
    configure_logging(LOG_LEVEL)
    args = parse_args()

    # Backward compatibility: legacy `--mil-finetune` used to imply MIL-only when the old default was `cnn`.
    if args.mil_finetune and args.model_mode == "cnn":
        args.model_mode = "mil"

    execute_full_workflow(args)


if __name__ == "__main__":
    run()
