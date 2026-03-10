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
    parser.add_argument("--tune-epochs", type=int, default=4, help="Epochs per tuning trial")
    parser.add_argument("--tune-max-trials", type=int, default=8, help="Maximum number of hyperparameter trials")
    parser.add_argument("--hparams-file", type=str, default=None, help="Path to a JSON file with hyperparameters to use for training (overrides saved defaults).")
    parser.add_argument("--tune-name", type=str, default=None, help="Optional name to attach to tuning results/best-hparams output when --tune is used.")
    return parser.parse_args()


def run():
    """Configure logging and execute the full training/evaluation workflow.

    Args:
        None

    Returns:
        None
    """
    configure_logging(LOG_LEVEL)
    execute_full_workflow(parse_args())


if __name__ == "__main__":
    run()
