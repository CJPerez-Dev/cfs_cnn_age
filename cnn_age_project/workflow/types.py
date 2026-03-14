"""Typed dataclass contracts exchanged between workflow stages."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class RunContext:
    """Run-scoped metadata such as timestamp, RNG, and output folder.

    Attributes:
        run_start_time: Start timestamp of the current run.
        run_tag: Filesystem-safe timestamp tag.
        rng: Reproducible random number generator for the run.
        run_output_dir: Output folder for artifacts generated in this run.
    """
    run_start_time: datetime
    run_tag: str
    rng: np.random.Generator
    run_output_dir: str


@dataclass
class RuntimeContext:
    """Runtime system resources and cache-path locations.

    Attributes:
        memmap_root: Directory containing source memmaps.
        device: Torch device selected for execution.
        gpu_name: Human-readable GPU name, or fallback label.
        split_cache_path: Path to split-code cache memmap.
        age_cache_path: Path to age-target cache memmap.
        subject_code_cache_path: Path to subject-code cache memmap.
        subject_codebook_path: Path to subject-codebook JSON.
    """
    memmap_root: str
    device: Any
    gpu_name: str
    split_cache_path: str
    age_cache_path: str
    subject_code_cache_path: str
    subject_codebook_path: str


@dataclass
class DataContext:
    """Loaded arrays and split metadata required for training/evaluation.

    Attributes:
        x_mem: Input window memmap.
        y_mem: Age-target memmap.
        n_samples: Number of rows/windows.
        window_len: Inferred window length.
        meta_path: Metadata CSV path.
        train_age_map: Train subject->age mapping.
        test_age_map: Test subject->age mapping.
        train_indices: Global indices for training windows.
        test_indices: Global indices for test windows.
        subject_codes: Subject code per global index.
        subject_codebook: Subject ID lookup by code.
        val_age_map: Optional validation subject->age mapping (None if no validation set).
        val_indices: Optional validation window indices (None or empty if no validation set).
    """
    x_mem: np.memmap
    y_mem: np.memmap
    n_samples: int
    window_len: int
    meta_path: str
    train_age_map: dict[str, float]
    test_age_map: dict[str, float]
    train_indices: np.ndarray
    test_indices: np.ndarray
    subject_codes: np.memmap
    subject_codebook: list[str]
    val_age_map: dict[str, float] | None = None
    val_indices: np.ndarray | None = None


@dataclass
class NormalizationContext:
    """Input/target normalization stats and optional balanced-sampling indices.

    Attributes:
        x_mean: Input normalization mean.
        x_std: Input normalization std.
        y_mean: Target normalization mean.
        y_std: Target normalization std.
        balanced_sorted_indices: Subject-sorted training indices.
        balanced_offsets: Per-subject offsets into sorted indices.
        balanced_counts: Per-subject training-window counts.
    """
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float
    balanced_sorted_indices: np.ndarray | None
    balanced_offsets: np.ndarray | None
    balanced_counts: np.ndarray | None


@dataclass
class HyperparameterContext:
    """Active hyperparameter set chosen for the current run.

    Attributes:
        active_hparams: Raw hyperparameter dictionary.
        active_lr: Active learning rate.
        active_use_huber: Whether Huber loss is enabled.
        active_huber_beta: Huber beta value.
        active_max_windows_per_subject: Per-subject sampling cap.
    """
    active_hparams: dict[str, Any]
    active_lr: float
    active_use_huber: bool
    active_huber_beta: float
    active_max_windows_per_subject: int


@dataclass
class TrainingContext:
    """Objects and metrics produced by the training stage.

    Attributes:
        model: Trained model instance.
        criterion: Loss criterion.
        optimizer: Optimizer used for training.
        scaler: AMP gradient scaler.
        amp_enabled: Whether AMP was enabled.
        triton_available: Whether Triton was available at runtime.
        compile_applied: Whether ``torch.compile`` was applied.
        train_losses: Epoch training losses.
        r2_scores: Epoch training R² values.
        maes: Epoch training MAE values.
        best_loss: Best observed training loss.
        best_epoch: Epoch number corresponding to best_loss.
    """
    model: Any
    criterion: Any
    optimizer: Any
    scaler: Any
    amp_enabled: bool
    triton_available: bool
    compile_applied: bool
    train_losses: list[float]
    r2_scores: list[float]
    maes: list[float]
    best_loss: float
    best_epoch: int


@dataclass
class EvaluationContext:
    """Window-level, subject-level, and baseline metrics from evaluation.

    Attributes:
        test_loss: Window-level test loss.
        test_r2: Window-level test R².
        test_mae: Window-level test MAE.
        final_targets: Sampled targets for visualization.
        final_preds: Sampled predictions for visualization.
        subj_mae: Subject-level MAE.
        subj_r2: Subject-level R².
        subj_ids: Evaluated subject IDs.
        mae_ci: Bootstrap confidence interval for subject MAE.
        r2_ci: Bootstrap confidence interval for subject R².
        baseline_pred: Constant baseline prediction value.
        baseline_loss: Baseline loss.
        baseline_r2: Baseline R².
        baseline_mae: Baseline MAE.
        subject_examples: Random subject summaries for plotting.
    """
    test_loss: float
    test_r2: float
    test_mae: float
    final_targets: np.ndarray
    final_preds: np.ndarray
    subj_mae: float
    subj_r2: float
    subj_ids: list[str]
    mae_ci: tuple[float, float]
    r2_ci: tuple[float, float]
    baseline_pred: float
    baseline_loss: float
    baseline_r2: float
    baseline_mae: float
    subject_examples: list[dict[str, Any]]
