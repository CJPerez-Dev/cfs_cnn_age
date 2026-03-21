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
    parser = argparse.ArgumentParser(description="EEG CNN age prediction with optional hyperparameter tuning")
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
        default="grid",
        help="Hyperparameter tuner backend. 'grid' uses shuffled candidate search, 'optuna' uses adaptive search.",
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
        help="Optuna pruner to use when --tune-backend optuna.",
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
        default="cnn",
        help="Select execution mode: baseline CNN only, MIL only, or both back-to-back with comparison artifacts.",
    )
    parser.add_argument(
        "--validation-key",
        type=str,
        default=None,
        help="Optional validation subject key CSV filename (e.g. AgeValidation_Key.csv) in input/. When set, a validation set is used for early stopping and LR scheduling.",
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

    parser.add_argument("--mil-finetune", action="store_true", help="Enable MIL Step 3 fine-tuning mode (unfreeze encoder + train full MIL model).")
    parser.add_argument("--mil-pretrained-model", type=str, default=None, help="Optional path to pretrained CNN checkpoint used to initialize MIL encoder.")
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

    # Backward compatibility: if legacy flag is set and model-mode was left at
    # default CNN, switch to MIL mode.
    if args.mil_finetune and args.model_mode == "cnn":
        args.model_mode = "mil"

    execute_full_workflow(args)


if __name__ == "__main__":
    run()
