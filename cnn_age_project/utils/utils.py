"""Shared utility helpers for logging, progress bars, and device selection."""

import logging
import torch
from tqdm import tqdm


def configure_logging(log_level):
    """Initialize global logging format and level for the application.

    Args:
        log_level (str): Log level name such as "INFO" or "DEBUG".

    Returns:
        None
    """
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def log_stage(stage_name, logger):
    """Emit a visual stage divider to improve run-log readability.

    Args:
        stage_name (str): Human-readable pipeline stage name.
        logger (logging.Logger): Logger used for output.

    Returns:
        None
    """
    logger.info("================== %s ==================", stage_name)


def make_tqdm(iterable, total=None, desc="", unit="it", position=0, leave=True):
    """Create a consistent tqdm progress bar wrapper.

    Args:
        iterable: Iterable object to wrap.
        total (int | None): Explicit total count for progress display.
        desc (str): Progress bar description text.
        unit (str): Unit label shown by tqdm.
        position (int): Row position for stacked progress bars.
        leave (bool): Whether to keep bar rendered after completion.

    Returns:
        tqdm.tqdm: Configured tqdm iterator.
    """
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        position=position,
        leave=leave,
    )


def select_device(use_tf32, logger):
    """Select CPU/GPU device and apply CUDA TF32 performance settings.

    Args:
        use_tf32 (bool): Whether to enable TF32 on supported CUDA hardware.
        logger (logging.Logger): Logger used for environment messages.

    Returns:
        tuple[torch.device, str]: Selected device and detected GPU name (or "none").
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    gpu_name = "none"

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("GPU: %s", gpu_name)
        torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(use_tf32)
        if use_tf32:
            logger.info("TF32 enabled via allow_tf32 for matmul/cuDNN.")
        else:
            logger.info("TF32 disabled.")

    torch.backends.cudnn.benchmark = (device.type == "cuda")
    return device, gpu_name
