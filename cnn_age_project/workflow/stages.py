"""Top-level staged workflow for training, evaluation, and artifact export."""

import logging
import os
from datetime import datetime
import copy
from time import perf_counter

import numpy as np
import torch

from cnn_age_project.config import (
    AGE_TARGET_CACHE_FILE,
    BATCH_SIZE,
    BEST_HPARAMS_FILE,
    CLIP_PREDICTED_AGE_AFTER_DENORM,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_ENABLED,
    BOOTSTRAP_ITERATIONS,
    DEBUG_CHUNK_LOG_EVERY,
    EARLY_STOPPING_ENABLED,
    EARLY_STOPPING_MIN_DELTA_ABS,
    EARLY_STOPPING_MIN_DELTA_REL,
    EARLY_STOPPING_MIN_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    HUBER_BETA,
    INPUT_DIR,
    LOG_LEVEL,
    LR,
    MIL_ALLOW_REPLACEMENT_WHEN_SMALL,
    MIL_BAG_BATCH_SIZE,
    MIL_FINETUNE_ENCODER_LR,
    MIL_FINETUNE_ENCODER_WARMUP_EPOCHS,
    MIL_FINETUNE_HEAD_LR,
    MIL_FINETUNE_WEIGHT_DECAY,
    MIL_INVERSE_FREQUENCY_SUBJECT_SAMPLING,
    MIL_SUBJECT_DRAWS_PER_EPOCH,
    MIL_PSEUDO_BAG_MAX_WINDOWS,
    MIL_PSEUDO_BAG_MIN_WINDOWS,
    MAX_WINDOWS_PER_SUBJECT_PER_EPOCH,
    METRIC_EVERY_N_EPOCHS,
    MIN_PREDICTED_AGE_YEARS,
    MODEL_SAVE_NAME,
    NORM_EPS,
    NORMALIZE_INPUT,
    NORMALIZE_TARGET,
    NORM_SAMPLE_WINDOWS,
    OUTPUT_DIR,
    PLOT_MAX_POINTS,
    RANDOM_SEED,
    REPORT_SAVE_NAME,
    SPLIT_CACHE_FILE,
    SUBJECT_BALANCED_TRAINING,
    SUBJECT_CODEBOOK_FILE,
    SUBJECT_CODE_CACHE_FILE,
    SUBJECT_EXAMPLE_COUNT,
    SUBJECT_EXAMPLE_MAX_WINDOWS,
    SUBJECT_REPORT_SAVE_NAME,
    TEST_KEY_CSV,
    TORCH_COMPILE_DYNAMIC,
    TORCH_COMPILE_MODE,
    TRAIN_KEY_CSV,
    USE_HUBER_LOSS,
    USE_TF32,
    USE_TORCH_COMPILE,
    USE_AUTO_SPLIT,
    USE_AGE_STRATIFIED_SPLIT,
    STRATIFY_AGE_BIN_YEARS,
    STRATIFY_TAIL_LOW_MAX_AGE,
    STRATIFY_TAIL_HIGH_MIN_AGE,
    STRATIFY_MIN_SUBJECTS_PER_STRATUM,
    AGE_WEIGHTED_WINDOW_SAMPLING,
    VALIDATION_KEY_CSV,
    SPLIT_RATIO_TRAIN,
    SPLIT_RATIO_VAL,
    SPLIT_RATIO_TEST,
    SUBJECT_KEY_CSV,
    LR_SCHEDULER,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR,
    GRAD_CLIP_NORM,
    CNN_WEIGHT_DECAY,
    MIL_EARLY_STOPPING_ENABLED,
    MIL_EARLY_STOPPING_PATIENCE,
    MIL_EARLY_STOPPING_MIN_EPOCHS,
    MIL_EARLY_STOPPING_MIN_DELTA_ABS,
    MIL_EARLY_STOPPING_MIN_DELTA_REL,
    MIL_REGRESSOR_DROPOUT,
    PROJECT_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_HPARAMS_PATH,
)
from cnn_age_project.data.age_strata import (
    build_merged_strata_for_subjects,
    build_subject_code_stratum_lookup,
    log_stratum_summary,
    split_subjects_stratified_train_val_test,
)
from cnn_age_project.data.data_io import (
    build_or_load_targets_and_split,
    build_subject_age_map_from_metadata,
    load_memmap_arrays,
    load_subject_age_map,
    split_subjects_ratio,
    validate_subject_wise_split_integrity,
    validate_required_project_files,
)
from cnn_age_project.data.dataset import (
    build_subject_group_index,
    compute_train_window_stratum_sample_weights,
)
from cnn_age_project.data.preprocessing import compute_target_norm_stats, estimate_input_norm_stats
from cnn_age_project.evaluation.evaluation import (
    bootstrap_ci_subject_metrics,
    build_subject_examples,
    compute_constant_baseline,
    compute_r2_from_arrays,
    compute_subject_level_metrics,
    run_epoch_metrics,
)
from cnn_age_project.experiments.experiment_logger import (
    build_optuna_candidate,
    build_tuning_candidates,
    get_default_hyperparameters,
    load_best_hyperparameters_if_available,
    save_model_comparison_summary,
    save_best_hyperparameters,
    save_run_summary,
)
from cnn_age_project.models.mil import MILAgeRegressor
from cnn_age_project.training.trainer import (
    build_mil_gated_attention_model,
    configure_mil_finetune_optimizer,
    evaluate_mil_on_subject_bags,
    run_mil_finetune_training,
    run_tuning_trial,
    setup_model_and_optimizers,
    train_model,
)
from cnn_age_project.training.losses import get_loss_function
from cnn_age_project.utils.utils import log_stage, make_tqdm, select_device
from cnn_age_project.visualization.plots import (
    save_model_comparison_report,
    save_subject_examples_report,
    save_training_report,
)
from cnn_age_project.workflow.types import (
    DataContext,
    EvaluationContext,
    HyperparameterContext,
    NormalizationContext,
    RunContext,
    RuntimeContext,
    TrainingContext,
)

logger = logging.getLogger(__name__)


def _run_candidate_trial(
    mode_for_tune,
    trial_idx,
    total_trials,
    candidate,
    args,
    data_ctx,
    norm_ctx,
    runtime_ctx,
    base_cnn_state_dict,
):
    """Run one candidate in cnn/mil/both mode and return tuning metrics."""
    if mode_for_tune == "cnn":
        return run_tuning_trial(
            trial_idx=trial_idx,
            total_trials=total_trials,
            hparams=candidate,
            tune_epochs=args.tune_epochs,
            x_mem=data_ctx.x_mem,
            y_mem=data_ctx.y_mem,
            train_indices=data_ctx.train_indices,
            test_indices=data_ctx.test_indices,
            subject_codes=data_ctx.subject_codes,
            subject_codebook=data_ctx.subject_codebook,
            x_mean=norm_ctx.x_mean,
            x_std=norm_ctx.x_std,
            y_mean=norm_ctx.y_mean,
            y_std=norm_ctx.y_std,
            balanced_sorted_indices=norm_ctx.balanced_sorted_indices,
            balanced_offsets=norm_ctx.balanced_offsets,
            balanced_counts=norm_ctx.balanced_counts,
            device=runtime_ctx.device,
            amp_enabled=(runtime_ctx.device.type == "cuda"),
            rng=np.random.default_rng(RANDOM_SEED + trial_idx),
            window_len=data_ctx.window_len,
            batch_size=BATCH_SIZE,
            normalize_target=NORMALIZE_TARGET,
            subject_balanced_training=SUBJECT_BALANCED_TRAINING,
            max_windows_per_subject_per_epoch=MAX_WINDOWS_PER_SUBJECT_PER_EPOCH,
            debug_chunk_log_every=DEBUG_CHUNK_LOG_EVERY,
            plot_max_points=PLOT_MAX_POINTS,
            grad_clip_norm=GRAD_CLIP_NORM,
            val_indices=data_ctx.val_indices,
            train_window_sample_weights=norm_ctx.train_window_sample_weights,
            use_age_weighted_window_sampling=bool(
                getattr(args, "age_weighted_window_sampling", AGE_WEIGHTED_WINDOW_SAMPLING)
            ),
        )

    if mode_for_tune == "mil":
        return _run_mil_tuning_trial(
            trial_idx=trial_idx,
            total_trials=total_trials,
            hparams=candidate,
            tune_epochs=args.tune_epochs,
            data_ctx=data_ctx,
            norm_ctx=norm_ctx,
            runtime_ctx=runtime_ctx,
            rng=np.random.default_rng(RANDOM_SEED + trial_idx),
            base_cnn_state_dict=base_cnn_state_dict,
            args=args,
        )

    if mode_for_tune == "both":
        cnn_result = run_tuning_trial(
            trial_idx=trial_idx,
            total_trials=total_trials,
            hparams=candidate,
            tune_epochs=args.tune_epochs,
            x_mem=data_ctx.x_mem,
            y_mem=data_ctx.y_mem,
            train_indices=data_ctx.train_indices,
            test_indices=data_ctx.test_indices,
            subject_codes=data_ctx.subject_codes,
            subject_codebook=data_ctx.subject_codebook,
            x_mean=norm_ctx.x_mean,
            x_std=norm_ctx.x_std,
            y_mean=norm_ctx.y_mean,
            y_std=norm_ctx.y_std,
            balanced_sorted_indices=norm_ctx.balanced_sorted_indices,
            balanced_offsets=norm_ctx.balanced_offsets,
            balanced_counts=norm_ctx.balanced_counts,
            device=runtime_ctx.device,
            amp_enabled=(runtime_ctx.device.type == "cuda"),
            rng=np.random.default_rng(RANDOM_SEED + trial_idx),
            window_len=data_ctx.window_len,
            batch_size=BATCH_SIZE,
            normalize_target=NORMALIZE_TARGET,
            subject_balanced_training=SUBJECT_BALANCED_TRAINING,
            max_windows_per_subject_per_epoch=MAX_WINDOWS_PER_SUBJECT_PER_EPOCH,
            debug_chunk_log_every=DEBUG_CHUNK_LOG_EVERY,
            plot_max_points=PLOT_MAX_POINTS,
            grad_clip_norm=GRAD_CLIP_NORM,
            val_indices=data_ctx.val_indices,
            train_window_sample_weights=norm_ctx.train_window_sample_weights,
            use_age_weighted_window_sampling=bool(
                getattr(args, "age_weighted_window_sampling", AGE_WEIGHTED_WINDOW_SAMPLING)
            ),
        )
        mil_result = _run_mil_tuning_trial(
            trial_idx=trial_idx,
            total_trials=total_trials,
            hparams=candidate,
            tune_epochs=args.tune_epochs,
            data_ctx=data_ctx,
            norm_ctx=norm_ctx,
            runtime_ctx=runtime_ctx,
            rng=np.random.default_rng(RANDOM_SEED + 10_000 + trial_idx),
            base_cnn_state_dict=base_cnn_state_dict,
            args=args,
        )
        cnn_sel = float(cnn_result.get("selection_mae", cnn_result["test_mae"]))
        mil_sel = float(mil_result.get("selection_mae", mil_result["test_mae"]))
        combined_selection = 0.5 * (cnn_sel + mil_sel)
        combined_test = 0.5 * (float(cnn_result["test_mae"]) + float(mil_result["test_mae"]))
        return {
            "hparams": candidate,
            "selection_mae": float(combined_selection),
            "test_mae": float(combined_test),
            "test_r2": np.nan,
            "test_loss": np.nan,
            "trial_seconds": float(cnn_result["trial_seconds"] + mil_result["trial_seconds"]),
            "combined_metric": "mean_val_mae",
            "cnn": cnn_result,
            "mil": mil_result,
            "model_type": "both",
        }

    raise ValueError(f"Unsupported tuning mode: {mode_for_tune}")


def initialize_run_context():
    """Create run tag/RNG/output folder and log startup configuration.

    Args:
        None

    Returns:
        RunContext: Initialized run metadata and output folder information.
    """
    run_start_time = datetime.now()
    run_tag = run_start_time.strftime("%Y-%m-%d_%H-%M-%S")
    rng = np.random.default_rng(RANDOM_SEED)

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_output_dir = os.path.join(OUTPUT_DIR, run_tag)
    os.makedirs(run_output_dir, exist_ok=True)

    logging.info("Pipeline started | output=%s", run_output_dir)

    return RunContext(
        run_start_time=run_start_time,
        run_tag=run_tag,
        rng=rng,
        run_output_dir=run_output_dir,
    )


def setup_runtime_context(args=None):
    """Validate filesystem prerequisites and initialize device/runtime paths.

    Args:
        args (argparse.Namespace | None): Optional CLI args; --auto-split overrides config.

    Returns:
        RuntimeContext: Runtime device information and cache paths.
    """
    use_auto_split = getattr(args, "auto_split", USE_AUTO_SPLIT) if args is not None else USE_AUTO_SPLIT
    use_age_stratified = USE_AGE_STRATIFIED_SPLIT
    if args is not None:
        use_age_stratified = bool(getattr(args, "age_stratified_split", use_age_stratified))

    log_stage("Preflight Checks", logger)
    memmap_root = validate_required_project_files(INPUT_DIR, OUTPUT_DIR, use_auto_split=use_auto_split)

    log_stage("Device Setup", logger)
    device, gpu_name = select_device(USE_TF32, logger)

    if use_auto_split:
        base_s, ext_s = os.path.splitext(SPLIT_CACHE_FILE)
        base_a, ext_a = os.path.splitext(AGE_TARGET_CACHE_FILE)
        base_c, ext_c = os.path.splitext(SUBJECT_CODE_CACHE_FILE)
        base_cb, ext_cb = os.path.splitext(SUBJECT_CODEBOOK_FILE)
        auto_tag = "_auto_strat" if use_age_stratified else "_auto"
        split_cache_path = os.path.join(OUTPUT_DIR, base_s + auto_tag + ext_s)
        age_cache_path = os.path.join(OUTPUT_DIR, base_a + auto_tag + ext_a)
        subject_code_cache_path = os.path.join(OUTPUT_DIR, base_c + auto_tag + ext_c)
        subject_codebook_path = os.path.join(OUTPUT_DIR, base_cb + auto_tag + ext_cb)
    else:
        split_cache_path = os.path.join(OUTPUT_DIR, SPLIT_CACHE_FILE)
        age_cache_path = os.path.join(OUTPUT_DIR, AGE_TARGET_CACHE_FILE)
        subject_code_cache_path = os.path.join(OUTPUT_DIR, SUBJECT_CODE_CACHE_FILE)
        subject_codebook_path = os.path.join(OUTPUT_DIR, SUBJECT_CODEBOOK_FILE)

    return RuntimeContext(
        memmap_root=memmap_root,
        device=device,
        gpu_name=gpu_name,
        split_cache_path=split_cache_path,
        age_cache_path=age_cache_path,
        subject_code_cache_path=subject_code_cache_path,
        subject_codebook_path=subject_codebook_path,
    )


def load_data_context(runtime_ctx: RuntimeContext, args=None):
    """Load memmaps, build/load split caches, and return train/test indices.

    Args:
        runtime_ctx (RuntimeContext): Runtime context with memmap root and cache paths.
        args (argparse.Namespace | None): Optional CLI args; --validation-key overrides config.

    Returns:
        DataContext: Loaded arrays, split indices, and subject metadata.
    """
    log_stage("Memmap Loading", logger)
    memmaps = load_memmap_arrays(runtime_ctx.memmap_root)
    x_mem = memmaps["x_mem"]
    n_samples = memmaps["n_samples"]
    window_len = memmaps["window_len"]
    meta_path = memmaps["meta_path"]
    logging.info("Samples: %d | window length: %d", n_samples, window_len)

    log_stage("Label + Split Preparation", logger)
    use_auto_split = getattr(args, "auto_split", USE_AUTO_SPLIT) if args is not None else USE_AUTO_SPLIT
    use_age_stratified = USE_AGE_STRATIFIED_SPLIT
    if args is not None:
        use_age_stratified = bool(getattr(args, "age_stratified_split", use_age_stratified))

    subject_stratum_merged: dict[str, int] | None = None

    if use_auto_split:
        # 70/15/15 (or configured ratios) subject split from metadata; key files ignored.
        subject_age = None
        key_path = SUBJECT_KEY_CSV
        if key_path is None or not os.path.exists(key_path):
            key_path = os.path.join(INPUT_DIR, "AgeKey.csv")
        if key_path and os.path.exists(key_path):
            subject_age = load_subject_age_map(key_path)
        if not subject_age:
            subject_age = build_subject_age_map_from_metadata(meta_path, n_samples)
        if subject_age:
            logger.debug("Subject ages: %d subjects", len(subject_age))
        if not subject_age:
            raise ValueError(
                "use_auto_split is True but no subject ages found. "
                "Provide a single key CSV (config.subject_key_filename) with SubjectID, VariableValue, "
                "or add an age column (age, Age, or VariableValue) to the metadata CSV."
            )
        rng_split = np.random.default_rng(RANDOM_SEED)
        if use_age_stratified:
            subject_stratum_merged = build_merged_strata_for_subjects(
                subject_age,
                STRATIFY_TAIL_LOW_MAX_AGE,
                STRATIFY_TAIL_HIGH_MIN_AGE,
                STRATIFY_AGE_BIN_YEARS,
                STRATIFY_MIN_SUBJECTS_PER_STRATUM,
            )
            log_stratum_summary(subject_age, subject_stratum_merged)
            train_ids, val_ids, test_ids = split_subjects_stratified_train_val_test(
                subject_age,
                subject_stratum_merged,
                SPLIT_RATIO_TRAIN,
                SPLIT_RATIO_VAL,
                SPLIT_RATIO_TEST,
                rng_split,
            )
            logger.info(
                "Age-stratified split | train=%d val=%d test=%d subjects | bands=%.0fy tails <%.0f / >=%.0f min/stratum=%d",
                len(train_ids),
                len(val_ids),
                len(test_ids),
                STRATIFY_AGE_BIN_YEARS,
                STRATIFY_TAIL_LOW_MAX_AGE,
                STRATIFY_TAIL_HIGH_MIN_AGE,
                STRATIFY_MIN_SUBJECTS_PER_STRATUM,
            )
        else:
            subject_ids = list(subject_age.keys())
            train_ids, val_ids, test_ids = split_subjects_ratio(
                subject_ids,
                SPLIT_RATIO_TRAIN,
                SPLIT_RATIO_VAL,
                SPLIT_RATIO_TEST,
                rng_split,
            )
            logger.info("Random split | train=%d val=%d test=%d subjects", len(train_ids), len(val_ids), len(test_ids))
        train_age_map = {s: subject_age[s] for s in train_ids}
        test_age_map = {s: subject_age[s] for s in test_ids}
        val_age_map = {s: subject_age[s] for s in val_ids}
    else:
        if not os.path.exists(TRAIN_KEY_CSV) or not os.path.exists(TEST_KEY_CSV):
            raise FileNotFoundError("Training/testing key CSV files were not found in input folder.")
        train_age_map = load_subject_age_map(TRAIN_KEY_CSV)
        test_age_map = load_subject_age_map(TEST_KEY_CSV)
        overlap_subjects = set(train_age_map).intersection(set(test_age_map))
        if len(overlap_subjects) > 0:
            raise ValueError(f"Train/test subject overlap detected: {len(overlap_subjects)} subjects.")
        val_age_map = None
        subject_stratum_merged = None
        validation_csv_path = None
        if getattr(args, "validation_key", None):
            validation_csv_path = os.path.join(INPUT_DIR, args.validation_key)
        elif VALIDATION_KEY_CSV is not None:
            validation_csv_path = VALIDATION_KEY_CSV
        if validation_csv_path is not None and os.path.exists(validation_csv_path):
            val_age_map = load_subject_age_map(validation_csv_path)
            for name, other in [("train", train_age_map), ("test", test_age_map)]:
                overlap = set(val_age_map).intersection(set(other))
                if overlap:
                    raise ValueError(
                        f"Validation subjects must not overlap with {name}: {len(overlap)} overlapping."
                    )
            logger.info("Validation set: %d subjects", len(val_age_map))

    split_codes, age_targets, subject_codes, subject_codebook = build_or_load_targets_and_split(
        meta_path=meta_path,
        n_samples=n_samples,
        train_age_map=train_age_map,
        test_age_map=test_age_map,
        split_cache_path=runtime_ctx.split_cache_path,
        age_cache_path=runtime_ctx.age_cache_path,
        subject_code_cache_path=runtime_ctx.subject_code_cache_path,
        subject_codebook_path=runtime_ctx.subject_codebook_path,
        val_age_map=val_age_map,
    )

    valid_age = np.isfinite(age_targets)
    train_indices = np.where((split_codes == 1) & valid_age)[0]
    test_indices = np.where((split_codes == 2) & valid_age)[0]
    val_indices = np.where((split_codes == 3) & valid_age)[0]
    if val_indices.size == 0:
        val_indices = None
    else:
        val_indices = np.sort(val_indices)

    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError("No train/test windows with valid age labels found.")

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    y_mem = age_targets

    validate_subject_wise_split_integrity(
        train_indices=train_indices,
        test_indices=test_indices,
        subject_codes=subject_codes,
        subject_codebook=subject_codebook,
        val_indices=val_indices,
    )

    logging.info(
        "Windows | train=%d test=%d%s",
        train_indices.size,
        test_indices.size,
        f" val={val_indices.size}" if val_indices is not None and val_indices.size > 0 else "",
    )

    return DataContext(
        x_mem=x_mem,
        y_mem=y_mem,
        n_samples=n_samples,
        window_len=window_len,
        meta_path=meta_path,
        train_age_map=train_age_map,
        test_age_map=test_age_map,
        train_indices=train_indices,
        test_indices=test_indices,
        subject_codes=subject_codes,
        subject_codebook=subject_codebook,
        val_age_map=val_age_map,
        val_indices=val_indices,
        subject_stratum_merged=subject_stratum_merged,
    )


def setup_normalization_context(data_ctx: DataContext, rng, args=None):
    """Compute normalization statistics and optional subject-balanced index structures.

    Args:
        data_ctx (DataContext): Loaded training/testing data context.
        rng (np.random.Generator): Random generator for sampling normalization windows.
        args (argparse.Namespace | None): Optional CLI args for sampling overrides.

    Returns:
        NormalizationContext: Normalization values and balanced-sampling structures.
    """
    log_stage("Normalization Setup", logger)
    y_mean, y_std = compute_target_norm_stats(data_ctx.y_mem, data_ctx.train_indices, norm_eps=NORM_EPS)
    if not NORMALIZE_TARGET:
        y_mean, y_std = 0.0, 1.0
    if y_std < 1e-6:
        logger.warning("Target std was extremely small (%.6g). Clipping to 1e-3 for stability.", y_std)
        y_std = 1e-3
    logger.debug("Target norm: enabled=%s mean=%.4f std=%.4f", NORMALIZE_TARGET, y_mean, y_std)

    if NORMALIZE_INPUT:
        x_mean, x_std = estimate_input_norm_stats(
            x_mem=data_ctx.x_mem,
            train_indices=data_ctx.train_indices,
            rng=rng,
            norm_eps=NORM_EPS,
            sample_windows=NORM_SAMPLE_WINDOWS,
        )
    else:
        x_mean, x_std = 0.0, 1.0
    if x_std < 1e-6:
        logger.warning("Input std was extremely small (%.6g). Clipping to 1e-3 for stability.", x_std)
        x_std = 1e-3
    logger.debug("Input norm: enabled=%s mean=%.6f std=%.6f", NORMALIZE_INPUT, x_mean, x_std)

    balanced_sorted_indices = None
    balanced_offsets = None
    balanced_counts = None
    if SUBJECT_BALANCED_TRAINING:
        balanced_sorted_indices, balanced_offsets, balanced_counts = build_subject_group_index(
            train_indices=data_ctx.train_indices,
            subject_codes=data_ctx.subject_codes,
            n_subjects=len(data_ctx.subject_codebook),
        )
        logger.debug("Subject-balanced: max_windows/subject=%d subjects=%d", MAX_WINDOWS_PER_SUBJECT_PER_EPOCH, int(np.sum(balanced_counts > 0)))

    use_age_weighted = AGE_WEIGHTED_WINDOW_SAMPLING
    if args is not None:
        use_age_weighted = bool(getattr(args, "age_weighted_window_sampling", use_age_weighted))

    train_window_sample_weights = None
    if use_age_weighted:
        merged = data_ctx.subject_stratum_merged
        if merged is None:
            merged = build_merged_strata_for_subjects(
                dict(data_ctx.train_age_map),
                STRATIFY_TAIL_LOW_MAX_AGE,
                STRATIFY_TAIL_HIGH_MIN_AGE,
                STRATIFY_AGE_BIN_YEARS,
                STRATIFY_MIN_SUBJECTS_PER_STRATUM,
            )
        lookup = build_subject_code_stratum_lookup(
            data_ctx.subject_codebook,
            data_ctx.train_age_map,
            merged,
            STRATIFY_TAIL_LOW_MAX_AGE,
            STRATIFY_TAIL_HIGH_MIN_AGE,
            STRATIFY_AGE_BIN_YEARS,
        )
        train_window_sample_weights = compute_train_window_stratum_sample_weights(
            data_ctx.train_indices,
            data_ctx.subject_codes,
            lookup,
        )
        logger.info(
            "Age-weighted window sampling (inverse stratum frequency) | n_train_windows=%d mean_weight=%.4f",
            int(data_ctx.train_indices.shape[0]),
            float(np.mean(train_window_sample_weights)),
        )
        if SUBJECT_BALANCED_TRAINING:
            logger.info(
                "Subject-balanced sampling is superseded by age-weighted sampling for CNN epoch indices "
                "(subject-balanced structures are still built for MIL)."
            )

    return NormalizationContext(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        balanced_sorted_indices=balanced_sorted_indices,
        balanced_offsets=balanced_offsets,
        balanced_counts=balanced_counts,
        train_window_sample_weights=train_window_sample_weights,
    )


def _resolve_mil_checkpoint_path(args):
    """Resolve MIL pretrained checkpoint path from CLI or defaults folder."""
    cli_path = getattr(args, "mil_pretrained_model", None)
    if cli_path:
        return cli_path
    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    return None


def _resolve_cli_or_hparam(args, cli_attr, hparams, hparam_key, fallback):
    """Resolve value with precedence CLI -> tuned hparams -> fallback constant."""
    cli_value = getattr(args, cli_attr, None)
    if cli_value is not None:
        return cli_value
    if isinstance(hparams, dict) and hparam_key in hparams:
        return hparams[hparam_key]
    return fallback


def _run_mil_tuning_trial(
    trial_idx,
    total_trials,
    hparams,
    tune_epochs,
    data_ctx,
    norm_ctx,
    runtime_ctx,
    rng,
    base_cnn_state_dict,
    args,
):
    """Run one MIL hyperparameter tuning trial and return held-out metrics.

    Model selection for tuning uses **validation** subject-bag MAE when a val split
    exists; otherwise falls back to test MAE (with a warning).
    """
    trial_start = perf_counter()
    logger.info("MIL tune trial %d/%d", trial_idx, total_trials)

    # When a pretrained CNN checkpoint is provided, its architecture
    # (embedding dimension) must match the CNN we instantiate for MIL.
    # Infer the encoder embedding dimension from the checkpoint if
    # available; otherwise fall back to candidate hparams/defaults.
    if base_cnn_state_dict is not None:
        # `features.8.bias` has shape [embedding_dim].
        try:
            embedding_dim = int(base_cnn_state_dict["features.8.bias"].shape[0])
        except Exception:
            embedding_dim = int(hparams.get("cnn_embedding_dim", 128))
    else:
        embedding_dim = int(hparams.get("cnn_embedding_dim", 128))

    mil_model = build_mil_gated_attention_model(
        window_len=data_ctx.window_len,
        device=runtime_ctx.device,
        freeze_encoder=True,
        base_cnn_state_dict=base_cnn_state_dict,
        feature_dim=embedding_dim,
        cnn_embedding_dim=embedding_dim,
        cnn_dropout=float(hparams.get("cnn_dropout", 0.0)),
        attention_dim=int(hparams.get("mil_attention_dim", 128)),
        regressor_hidden_dim=int(hparams.get("mil_regressor_hidden_dim", 64)),
        pooling_type=str(hparams.get("mil_pooling_type", "gated")),
        attention_dropout=float(hparams.get("mil_attention_dropout", 0.0)),
        regressor_dropout=float(hparams.get("mil_regressor_dropout", 0.1)),
    )

    criterion = get_loss_function(
        use_huber_loss=bool(hparams.get("use_huber_loss", USE_HUBER_LOSS)),
        huber_beta=float(hparams.get("huber_beta", HUBER_BETA)),
    )
    optimizer = configure_mil_finetune_optimizer(
        mil_model=mil_model,
        device=runtime_ctx.device,
        encoder_learning_rate=float(hparams.get("mil_encoder_lr", MIL_FINETUNE_ENCODER_LR)),
        mil_head_learning_rate=float(hparams.get("mil_head_lr", MIL_FINETUNE_HEAD_LR)),
        weight_decay=float(hparams.get("mil_weight_decay", MIL_FINETUNE_WEIGHT_DECAY)),
    )
    amp_enabled = runtime_ctx.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    bag_size = int(hparams.get("mil_bag_size", MIL_PSEUDO_BAG_MIN_WINDOWS))
    sampling_strategy = str(hparams.get("mil_sampling_strategy", "random"))

    train_sorted = norm_ctx.balanced_sorted_indices
    train_offsets = norm_ctx.balanced_offsets
    train_counts = norm_ctx.balanced_counts
    if train_sorted is None or train_offsets is None or train_counts is None:
        train_sorted, train_offsets, train_counts = build_subject_group_index(
            train_indices=data_ctx.train_indices,
            subject_codes=data_ctx.subject_codes,
            n_subjects=len(data_ctx.subject_codebook),
        )

    training = run_mil_finetune_training(
        mil_model=mil_model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=runtime_ctx.device,
        x_mem=data_ctx.x_mem,
        y_mem=data_ctx.y_mem,
        balanced_sorted_indices=train_sorted,
        balanced_offsets=train_offsets,
        balanced_counts=train_counts,
        rng=rng,
        epochs=tune_epochs,
        bag_batch_size=MIL_BAG_BATCH_SIZE,
        x_mean=norm_ctx.x_mean,
        x_std=norm_ctx.x_std,
        y_mean=norm_ctx.y_mean,
        y_std=norm_ctx.y_std,
        normalize_target=NORMALIZE_TARGET,
        pseudo_bag_min_windows=bag_size,
        pseudo_bag_max_windows=bag_size,
        allow_replacement_when_small=MIL_ALLOW_REPLACEMENT_WHEN_SMALL,
        sampling_strategy=sampling_strategy,
        debug_chunk_log_every=DEBUG_CHUNK_LOG_EVERY,
        amp_enabled=amp_enabled,
        subject_codebook=data_ctx.subject_codebook,
        train_age_map=data_ctx.train_age_map,
        subject_stratum_merged=data_ctx.subject_stratum_merged,
        mil_inverse_frequency_subject_sampling=bool(
            getattr(args, "mil_inverse_frequency_subject_sampling", MIL_INVERSE_FREQUENCY_SUBJECT_SAMPLING)
        ),
        mil_subject_draws_per_epoch=MIL_SUBJECT_DRAWS_PER_EPOCH,
    )

    # Test split: reported only (do not use for hyperparameter selection).
    eval_metrics = evaluate_mil_on_subject_bags(
        mil_model=training["model"],
        criterion=criterion,
        x_mem=data_ctx.x_mem,
        y_mem=data_ctx.y_mem,
        eval_indices=data_ctx.test_indices,
        subject_codes=data_ctx.subject_codes,
        n_subjects=len(data_ctx.subject_codebook),
        batch_size=MIL_BAG_BATCH_SIZE,
        device=runtime_ctx.device,
        x_mean=norm_ctx.x_mean,
        x_std=norm_ctx.x_std,
        y_mean=norm_ctx.y_mean,
        y_std=norm_ctx.y_std,
        normalize_target=NORMALIZE_TARGET,
        rng=rng,
        pseudo_bag_min_windows=bag_size,
        pseudo_bag_max_windows=bag_size,
        allow_replacement_when_small=MIL_ALLOW_REPLACEMENT_WHEN_SMALL,
        sampling_strategy=sampling_strategy,
    )

    val_metrics = None
    if data_ctx.val_indices is not None and data_ctx.val_indices.size > 0:
        val_metrics = evaluate_mil_on_subject_bags(
            mil_model=training["model"],
            criterion=criterion,
            x_mem=data_ctx.x_mem,
            y_mem=data_ctx.y_mem,
            eval_indices=data_ctx.val_indices,
            subject_codes=data_ctx.subject_codes,
            n_subjects=len(data_ctx.subject_codebook),
            batch_size=MIL_BAG_BATCH_SIZE,
            device=runtime_ctx.device,
            x_mean=norm_ctx.x_mean,
            x_std=norm_ctx.x_std,
            y_mean=norm_ctx.y_mean,
            y_std=norm_ctx.y_std,
            normalize_target=NORMALIZE_TARGET,
            rng=rng,
            pseudo_bag_min_windows=bag_size,
            pseudo_bag_max_windows=bag_size,
            allow_replacement_when_small=MIL_ALLOW_REPLACEMENT_WHEN_SMALL,
            sampling_strategy=sampling_strategy,
        )
        selection_mae = float(val_metrics["test_mae"])
    else:
        selection_mae = float(eval_metrics["test_mae"])

    trial_seconds = perf_counter() - trial_start
    if val_metrics is not None:
        logger.info(
            "[Tune %d/%d] MIL trial completed in %.1fs | val_mae=%.3f (selection) test_mae=%.3f test_r2=%.4f",
            trial_idx,
            total_trials,
            trial_seconds,
            selection_mae,
            eval_metrics["test_mae"],
            eval_metrics["test_r2"],
        )
    else:
        logger.info(
            "[Tune %d/%d] MIL trial completed in %.1fs | selection_mae=%.3f test_mae=%.3f test_r2=%.4f",
            trial_idx,
            total_trials,
            trial_seconds,
            selection_mae,
            eval_metrics["test_mae"],
            eval_metrics["test_r2"],
        )

    out = {
        "hparams": hparams,
        "test_loss": float(eval_metrics["test_loss"]),
        "test_r2": float(eval_metrics["test_r2"]) if np.isfinite(eval_metrics["test_r2"]) else np.nan,
        "test_mae": float(eval_metrics["test_mae"]),
        "selection_mae": selection_mae,
        "trial_seconds": float(trial_seconds),
        "model_type": "mil",
    }
    if val_metrics is not None:
        out["val_mae"] = selection_mae
        out["val_loss"] = float(val_metrics["test_loss"])
        out["val_r2"] = float(val_metrics["test_r2"]) if np.isfinite(val_metrics["test_r2"]) else np.nan
    return out


def select_hyperparameters(args, data_ctx: DataContext, norm_ctx: NormalizationContext, runtime_ctx: RuntimeContext, run_ctx: RunContext = None):
    """Select active hyperparameters via tuning mode or previously saved best values.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        data_ctx (DataContext): Data context used by tuning trials.
        norm_ctx (NormalizationContext): Normalization and sampling structures.
        runtime_ctx (RuntimeContext): Runtime context containing device info.

    Returns:
        HyperparameterContext: Selected active hyperparameters for training.
    """
    defaults = get_default_hyperparameters(
        lr=LR,
        use_huber_loss=USE_HUBER_LOSS,
        huber_beta=HUBER_BETA,
        max_windows_per_subject_per_epoch=MAX_WINDOWS_PER_SUBJECT_PER_EPOCH,
        cnn_weight_decay=float(getattr(args, "cnn_weight_decay", CNN_WEIGHT_DECAY) or CNN_WEIGHT_DECAY),
        mil_attention_dim=int(getattr(args, "mil_attention_dim", 128) or 128),
        mil_attention_dropout=float(getattr(args, "mil_attention_dropout", 0.1) or 0.1),
        mil_pooling_type=str(getattr(args, "mil_pooling_type", "gated") or "gated"),
        mil_bag_size=int(getattr(args, "mil_pseudo_bag_min_windows", MIL_PSEUDO_BAG_MIN_WINDOWS) or MIL_PSEUDO_BAG_MIN_WINDOWS),
        mil_sampling_strategy=str(getattr(args, "mil_sampling_strategy", "random") or "random"),
        mil_encoder_lr=float(getattr(args, "mil_encoder_lr", MIL_FINETUNE_ENCODER_LR) or MIL_FINETUNE_ENCODER_LR),
        mil_head_lr=float(getattr(args, "mil_head_lr", MIL_FINETUNE_HEAD_LR) or MIL_FINETUNE_HEAD_LR),
        mil_weight_decay=float(getattr(args, "mil_weight_decay", MIL_FINETUNE_WEIGHT_DECAY) or MIL_FINETUNE_WEIGHT_DECAY),
        mil_regressor_hidden_dim=int(getattr(args, "mil_regressor_hidden_dim", 64) or 64),
        mil_regressor_dropout=float(getattr(args, "mil_regressor_dropout", MIL_REGRESSOR_DROPOUT) or MIL_REGRESSOR_DROPOUT),
        mil_allow_replacement_when_small=bool(
            getattr(args, "mil_allow_replacement_when_small", MIL_ALLOW_REPLACEMENT_WHEN_SMALL)
            if getattr(args, "mil_allow_replacement_when_small", None) is not None
            else MIL_ALLOW_REPLACEMENT_WHEN_SMALL
        ),
    )
    # Determine where to source or save best hyperparameters.
    # Priority order when not tuning:
    # 1) `--hparams-file` provided by user
    # 2) repo-provided `defaults/default_hyperparameters.json` (root defaults)
    # 3) repo-provided `output/hparams/best_hyperparameters.json`
    # 4) output folder `output/hparams/best_hyperparameters.json`
    best_hparams_path = os.path.join(OUTPUT_DIR, "hparams", BEST_HPARAMS_FILE)
    repo_hparams_path = os.path.join(PROJECT_DIR, "output", "hparams", BEST_HPARAMS_FILE)
    default_hparams_path = DEFAULT_HPARAMS_PATH
    if args.hparams_file:
        # Accept absolute paths, repo-relative paths, and POSIX-style "/output/..." paths
        # that commonly appear when copying commands across shells/OSes.
        candidate = str(args.hparams_file)
        if not os.path.exists(candidate):
            candidate_rel = candidate.lstrip("/\\")
            candidate2 = os.path.join(PROJECT_DIR, candidate_rel)
            if os.path.exists(candidate2):
                candidate = candidate2
        best_hparams_path = candidate

    log_stage("Hyperparameter Selection", logger)
    if args.tune:
        mode_for_tune = str(getattr(args, "model_mode", "cnn") or "cnn")
        tune_backend = str(getattr(args, "tune_backend", "grid") or "grid").lower()
        max_trials = int(args.tune_max_trials)
        logger.info(
            "Hyperparameter tuning enabled | mode=%s | backend=%s | trials=%d | tune_epochs=%d",
            mode_for_tune,
            tune_backend,
            max_trials,
            args.tune_epochs,
        )
        val_ok = data_ctx.val_indices is not None and data_ctx.val_indices.size > 0
        if val_ok:
            logger.info(
                "Tuning minimizes selection_mae on the **validation** set (CNN: window MAE; MIL: subject-bag MAE; "
                "test metrics are logged only)."
            )
        else:
            logger.warning(
                "No validation split — selection_mae falls back to **test** MAE for CNN/MIL/both (test-informed "
                "model choice). Prefer auto-split with val or AgeValidation_Key.csv."
            )
        tuning_results = []
        tuning_start = perf_counter()
        best_so_far = None

        mil_checkpoint_path = None
        base_cnn_state_dict = None
        if mode_for_tune in {"mil", "both"}:
            mil_checkpoint_path = _resolve_mil_checkpoint_path(args)
            if not mil_checkpoint_path or not os.path.exists(mil_checkpoint_path):
                raise FileNotFoundError(
                    "MIL tuning requires a pretrained CNN checkpoint.\n"
                    "Provide one with `--mil-pretrained-model PATH` or place\n"
                    f"`{os.path.basename(DEFAULT_MODEL_PATH)}` in `{os.path.dirname(DEFAULT_MODEL_PATH)}`."
                )
            checkpoint = torch.load(mil_checkpoint_path, map_location="cpu")
            base_cnn_state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            logger.info("Loaded MIL tuning checkpoint: %s", mil_checkpoint_path)

        if tune_backend == "optuna":
            try:
                import optuna
            except Exception as exc:
                raise ImportError(
                    "Optuna backend selected but optuna is not installed. "
                    "Install with `pip install optuna` or switch to `--tune-backend grid`."
                ) from exc

            sampler_name = str(getattr(args, "optuna_sampler", "tpe") or "tpe").lower()
            pruner_name = str(getattr(args, "optuna_pruner", "median") or "median").lower()
            optuna_seed = int(getattr(args, "optuna_seed", RANDOM_SEED))
            startup_trials = max(0, int(getattr(args, "optuna_startup_trials", 10)))

            if sampler_name == "random":
                sampler = optuna.samplers.RandomSampler(seed=optuna_seed)
            else:
                sampler = optuna.samplers.TPESampler(seed=optuna_seed, n_startup_trials=startup_trials)

            if pruner_name == "none":
                pruner = optuna.pruners.NopPruner()
            else:
                pruner = optuna.pruners.MedianPruner(n_startup_trials=max(1, startup_trials))

            study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

            def _objective(trial):
                trial_idx = int(trial.number) + 1
                candidate = build_optuna_candidate(defaults, trial=trial, model_mode=mode_for_tune)
                trial_result = _run_candidate_trial(
                    mode_for_tune=mode_for_tune,
                    trial_idx=trial_idx,
                    total_trials=max_trials,
                    candidate=candidate,
                    args=args,
                    data_ctx=data_ctx,
                    norm_ctx=norm_ctx,
                    runtime_ctx=runtime_ctx,
                    base_cnn_state_dict=base_cnn_state_dict,
                )
                trial_result["trial_number"] = int(trial.number)
                trial_result["search_backend"] = "optuna"
                tuning_results.append(trial_result)
                selection_mae = trial_result.get("selection_mae", trial_result["test_mae"])
                trial.set_user_attr("selection_mae", float(selection_mae))
                return float(selection_mae)

            study.optimize(_objective, n_trials=max_trials, show_progress_bar=False)
            logger.info(
                "Optuna study complete | best_trial=%d best_mae=%.3f",
                int(study.best_trial.number),
                float(study.best_value),
            )
        else:
            candidates = build_tuning_candidates(defaults, model_mode=mode_for_tune)
            # Shuffle candidates so low trial counts still sample across the space.
            shuffle_rng = np.random.default_rng(RANDOM_SEED)
            candidate_order = shuffle_rng.permutation(len(candidates))
            candidates = [candidates[int(i)] for i in candidate_order]
            max_trials = min(max_trials, len(candidates))

            trial_bar = make_tqdm(
                enumerate(candidates[:max_trials], start=1),
                total=max_trials,
                desc="Hyperparameter Tuning",
                unit="trial",
                position=0,
                leave=True,
            )

            for trial_idx, candidate in trial_bar:
                trial_result = _run_candidate_trial(
                    mode_for_tune=mode_for_tune,
                    trial_idx=trial_idx,
                    total_trials=max_trials,
                    candidate=candidate,
                    args=args,
                    data_ctx=data_ctx,
                    norm_ctx=norm_ctx,
                    runtime_ctx=runtime_ctx,
                    base_cnn_state_dict=base_cnn_state_dict,
                )
                trial_result["search_backend"] = "grid"
                tuning_results.append(trial_result)

                selection_mae = trial_result.get("selection_mae", trial_result["test_mae"])
                if best_so_far is None or selection_mae < best_so_far.get("selection_mae", best_so_far["test_mae"]):
                    best_so_far = trial_result
                    logger.info(
                        "[Tune] New best at trial %d | MAE=%.3f R2=%.4f hparams=%s",
                        trial_idx,
                        selection_mae,
                        trial_result["test_r2"],
                        trial_result["hparams"],
                    )

                best_sel = best_so_far.get("selection_mae", best_so_far["test_mae"]) if best_so_far else None
                trial_bar.set_postfix(best_mae=f"{best_sel:.3f}" if best_sel is not None else "n/a")

        def _selection_mae(item):
            return item.get("selection_mae", item["test_mae"])
        best_trial = min(tuning_results, key=_selection_mae)
        active_hparams = best_trial["hparams"]
        # If the user provided a tune name, incorporate it into the saved filename.
        hparams_dir = os.path.join(OUTPUT_DIR, "hparams")
        os.makedirs(hparams_dir, exist_ok=True)
        if args.tune_name:
            dest_path = os.path.join(hparams_dir, f"best_hyperparameters_{args.tune_name}.json")
        else:
            # Use a run-unique filename (timestamp run_tag) to avoid overwriting
            # any existing `best_hyperparameters.json` provided by the repository.
            tag = run_ctx.run_tag if run_ctx is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dest_path = os.path.join(hparams_dir, f"best_hyperparameters_{tag}.json")
        save_best_hyperparameters(dest_path, active_hparams, tuning_results=tuning_results)
        logger.info(
            "Best tune | selection_mae=%.3f test_mae=%.3f (%.1fs)",
            _selection_mae(best_trial),
            best_trial["test_mae"],
            perf_counter() - tuning_start,
        )
    else:
        # Prefer explicit CLI file, then repository defaults (defaults/), then repo hparams output.
        if not args.hparams_file:
            if os.path.exists(default_hparams_path):
                best_hparams_path = default_hparams_path
            elif os.path.exists(repo_hparams_path):
                best_hparams_path = repo_hparams_path
        active_hparams = load_best_hyperparameters_if_available(best_hparams_path, defaults)
        logger.debug("Active hyperparameters: %s", active_hparams)

    return HyperparameterContext(
        active_hparams=active_hparams,
        active_lr=float(active_hparams.get("learning_rate", LR)),
        active_use_huber=bool(active_hparams.get("use_huber_loss", USE_HUBER_LOSS)),
        active_huber_beta=float(active_hparams.get("huber_beta", HUBER_BETA)),
        active_max_windows_per_subject=int(
            active_hparams.get("max_windows_per_subject_per_epoch", MAX_WINDOWS_PER_SUBJECT_PER_EPOCH)
        ),
    )


def run_training_stage(
    data_ctx: DataContext,
    norm_ctx: NormalizationContext,
    hparam_ctx: HyperparameterContext,
    runtime_ctx: RuntimeContext,
    rng,
    args,
):
    """Initialize trainable objects and execute the full training phase.

    Args:
        data_ctx (DataContext): Loaded data context.
        norm_ctx (NormalizationContext): Normalization and sampling context.
        hparam_ctx (HyperparameterContext): Selected hyperparameter context.
        runtime_ctx (RuntimeContext): Runtime context with selected device.
        rng (np.random.Generator): Random generator used during sampling.
        args (argparse.Namespace): Parsed CLI args for optional MIL mode.

    Returns:
        TrainingContext: Training artifacts and metrics histories.
    """
    if getattr(args, "mil_finetune", False):
        log_stage("MIL Model Setup (Step 3)", logger)

        mil_checkpoint_path = _resolve_mil_checkpoint_path(args)
        if mil_checkpoint_path == DEFAULT_MODEL_PATH:
            logger.info("No --mil-pretrained-model provided; using default model: %s", mil_checkpoint_path)
        base_cnn_state_dict = None
        if mil_checkpoint_path:
            if not os.path.exists(mil_checkpoint_path):
                raise FileNotFoundError(f"MIL pretrained model checkpoint not found: {mil_checkpoint_path}")
            checkpoint = torch.load(mil_checkpoint_path, map_location="cpu")
            base_cnn_state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            logger.info("Loaded MIL pretrained checkpoint: %s", mil_checkpoint_path)
        else:
            logger.warning(
                "MIL fine-tuning started without --mil-pretrained-model. "
                "Encoder will still be trainable, but initialization is from random/default weights."
            )

        active_hparams = hparam_ctx.active_hparams
        mil_attention_dim = int(
            _resolve_cli_or_hparam(args, "mil_attention_dim", active_hparams, "mil_attention_dim", 128)
        )
        mil_regressor_hidden_dim = int(
            _resolve_cli_or_hparam(args, "mil_regressor_hidden_dim", active_hparams, "mil_regressor_hidden_dim", 64)
        )
        mil_pooling_type = str(
            _resolve_cli_or_hparam(args, "mil_pooling_type", active_hparams, "mil_pooling_type", "gated")
        )
        cnn_embedding_dim = int(
            _resolve_cli_or_hparam(args, "cnn_embedding_dim", active_hparams, "cnn_embedding_dim", 128)
        )
        # If we're loading a pretrained CNN checkpoint, its embedding dimension
        # must match the instantiated encoder architecture. Infer it from the
        # checkpoint and override any hparam/CLI value to avoid size mismatches.
        if base_cnn_state_dict is not None:
            try:
                inferred_dim = int(base_cnn_state_dict["features.8.bias"].shape[0])
                if inferred_dim != cnn_embedding_dim:
                    logger.info(
                        "MIL encoder embedding dim overridden by checkpoint | hparams=%d checkpoint=%d",
                        cnn_embedding_dim,
                        inferred_dim,
                    )
                cnn_embedding_dim = inferred_dim
            except Exception:
                logger.warning(
                    "Could not infer cnn_embedding_dim from checkpoint; using configured value: %d",
                    cnn_embedding_dim,
                )
        cnn_dropout = float(
            _resolve_cli_or_hparam(args, "cnn_dropout", active_hparams, "cnn_dropout", 0.0)
        )
        mil_attention_dropout = float(
            _resolve_cli_or_hparam(args, "mil_attention_dropout", active_hparams, "mil_attention_dropout", 0.0)
        )
        mil_regressor_dropout = float(
            _resolve_cli_or_hparam(args, "mil_regressor_dropout", active_hparams, "mil_regressor_dropout", MIL_REGRESSOR_DROPOUT)
        )
        mil_encoder_lr = float(
            _resolve_cli_or_hparam(args, "mil_encoder_lr", active_hparams, "mil_encoder_lr", MIL_FINETUNE_ENCODER_LR)
        )
        mil_head_lr = float(
            _resolve_cli_or_hparam(args, "mil_head_lr", active_hparams, "mil_head_lr", MIL_FINETUNE_HEAD_LR)
        )
        mil_weight_decay = float(
            _resolve_cli_or_hparam(args, "mil_weight_decay", active_hparams, "mil_weight_decay", MIL_FINETUNE_WEIGHT_DECAY)
        )
        mil_bag_min = int(
            _resolve_cli_or_hparam(
                args,
                "mil_pseudo_bag_min_windows",
                active_hparams,
                "mil_bag_size",
                MIL_PSEUDO_BAG_MIN_WINDOWS,
            )
        )
        mil_bag_max = int(
            _resolve_cli_or_hparam(
                args,
                "mil_pseudo_bag_max_windows",
                active_hparams,
                "mil_bag_size",
                MIL_PSEUDO_BAG_MAX_WINDOWS,
            )
        )
        mil_sampling_strategy = str(
            _resolve_cli_or_hparam(args, "mil_sampling_strategy", active_hparams, "mil_sampling_strategy", "random")
        )
        mil_allow_replacement = bool(
            _resolve_cli_or_hparam(
                args,
                "mil_allow_replacement_when_small",
                active_hparams,
                "mil_allow_replacement_when_small",
                MIL_ALLOW_REPLACEMENT_WHEN_SMALL,
            )
        )

        mil_model = build_mil_gated_attention_model(
            window_len=data_ctx.window_len,
            device=runtime_ctx.device,
            freeze_encoder=True,
            base_cnn_state_dict=base_cnn_state_dict,
            feature_dim=cnn_embedding_dim,
            cnn_embedding_dim=cnn_embedding_dim,
            cnn_dropout=cnn_dropout,
            attention_dim=mil_attention_dim,
            regressor_hidden_dim=mil_regressor_hidden_dim,
            pooling_type=mil_pooling_type,
            attention_dropout=mil_attention_dropout,
            regressor_dropout=mil_regressor_dropout,
        )

        mil_sorted_indices = norm_ctx.balanced_sorted_indices
        mil_offsets = norm_ctx.balanced_offsets
        mil_counts = norm_ctx.balanced_counts
        if mil_sorted_indices is None or mil_offsets is None or mil_counts is None:
            logger.debug("Building MIL subject group index.")
            mil_sorted_indices, mil_offsets, mil_counts = build_subject_group_index(
                train_indices=data_ctx.train_indices,
                subject_codes=data_ctx.subject_codes,
                n_subjects=len(data_ctx.subject_codebook),
            )

        criterion = get_loss_function(
            use_huber_loss=hparam_ctx.active_use_huber,
            huber_beta=hparam_ctx.active_huber_beta,
        )
        optimizer = configure_mil_finetune_optimizer(
            mil_model=mil_model,
            device=runtime_ctx.device,
            encoder_learning_rate=mil_encoder_lr,
            mil_head_learning_rate=mil_head_lr,
            weight_decay=mil_weight_decay,
        )

        amp_enabled = runtime_ctx.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        log_stage("MIL Fine-Tuning", logger)
        training = run_mil_finetune_training(
            mil_model=mil_model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=runtime_ctx.device,
            x_mem=data_ctx.x_mem,
            y_mem=data_ctx.y_mem,
            balanced_sorted_indices=mil_sorted_indices,
            balanced_offsets=mil_offsets,
            balanced_counts=mil_counts,
            rng=rng,
            epochs=EPOCHS,
            bag_batch_size=int(getattr(args, "mil_bag_batch_size", None) or MIL_BAG_BATCH_SIZE),
            x_mean=norm_ctx.x_mean,
            x_std=norm_ctx.x_std,
            y_mean=norm_ctx.y_mean,
            y_std=norm_ctx.y_std,
            normalize_target=NORMALIZE_TARGET,
            pseudo_bag_min_windows=mil_bag_min,
            pseudo_bag_max_windows=mil_bag_max,
            allow_replacement_when_small=mil_allow_replacement,
            sampling_strategy=mil_sampling_strategy,
            debug_chunk_log_every=DEBUG_CHUNK_LOG_EVERY,
            amp_enabled=amp_enabled,
            early_stopping_enabled=MIL_EARLY_STOPPING_ENABLED,
            early_stopping_patience=MIL_EARLY_STOPPING_PATIENCE,
            early_stopping_min_epochs=MIL_EARLY_STOPPING_MIN_EPOCHS,
            early_stopping_min_delta_abs=MIL_EARLY_STOPPING_MIN_DELTA_ABS,
            early_stopping_min_delta_rel=MIL_EARLY_STOPPING_MIN_DELTA_REL,
            grad_clip_norm=GRAD_CLIP_NORM,
            encoder_warmup_epochs=int(MIL_FINETUNE_ENCODER_WARMUP_EPOCHS),
            encoder_learning_rate=float(active_hparams.get("mil_encoder_lr", MIL_FINETUNE_ENCODER_LR)),
            subject_codebook=data_ctx.subject_codebook,
            train_age_map=data_ctx.train_age_map,
            subject_stratum_merged=data_ctx.subject_stratum_merged,
            mil_inverse_frequency_subject_sampling=bool(
                getattr(args, "mil_inverse_frequency_subject_sampling", MIL_INVERSE_FREQUENCY_SUBJECT_SAMPLING)
            ),
            mil_subject_draws_per_epoch=(
                int(active_hparams["mil_subject_draws_per_epoch"])
                if active_hparams.get("mil_subject_draws_per_epoch") is not None
                else MIL_SUBJECT_DRAWS_PER_EPOCH
            ),
        )

        return TrainingContext(
            model=training["model"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            triton_available=False,
            compile_applied=False,
            train_losses=training["train_losses"],
            r2_scores=training["r2_scores"],
            maes=training["maes"],
            best_loss=training["best_loss"],
            best_epoch=training["best_epoch"],
        )

    log_stage("Model Setup", logger)
    model, criterion, optimizer, scaler, amp_enabled, triton_available, compile_applied = setup_model_and_optimizers(
        window_len=data_ctx.window_len,
        device=runtime_ctx.device,
        use_torch_compile=USE_TORCH_COMPILE,
        torch_compile_mode=TORCH_COMPILE_MODE,
        torch_compile_dynamic=TORCH_COMPILE_DYNAMIC,
        use_huber_loss=hparam_ctx.active_use_huber,
        huber_beta=hparam_ctx.active_huber_beta,
        learning_rate=hparam_ctx.active_lr,
        cnn_embedding_dim=int(hparam_ctx.active_hparams.get("cnn_embedding_dim", 128)),
        cnn_dropout=float(hparam_ctx.active_hparams.get("cnn_dropout", 0.0)),
        cnn_weight_decay=float(hparam_ctx.active_hparams.get("cnn_weight_decay", CNN_WEIGHT_DECAY)),
    )

    log_stage("Model Training", logger)
    logging.info("Starting training...")

    use_aw = AGE_WEIGHTED_WINDOW_SAMPLING
    if args is not None:
        use_aw = bool(getattr(args, "age_weighted_window_sampling", use_aw))

    training = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=runtime_ctx.device,
        x_mem=data_ctx.x_mem,
        y_mem=data_ctx.y_mem,
        train_indices=data_ctx.train_indices,
        balanced_sorted_indices=norm_ctx.balanced_sorted_indices,
        balanced_offsets=norm_ctx.balanced_offsets,
        balanced_counts=norm_ctx.balanced_counts,
        rng=rng,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        x_mean=norm_ctx.x_mean,
        x_std=norm_ctx.x_std,
        y_mean=norm_ctx.y_mean,
        y_std=norm_ctx.y_std,
        normalize_target=NORMALIZE_TARGET,
        subject_balanced_training=SUBJECT_BALANCED_TRAINING,
        active_max_windows_per_subject=hparam_ctx.active_max_windows_per_subject,
        plot_max_points=PLOT_MAX_POINTS,
        metric_every_n_epochs=METRIC_EVERY_N_EPOCHS,
        debug_chunk_log_every=DEBUG_CHUNK_LOG_EVERY,
        early_stopping_enabled=EARLY_STOPPING_ENABLED,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_epochs=EARLY_STOPPING_MIN_EPOCHS,
        early_stopping_min_delta_abs=EARLY_STOPPING_MIN_DELTA_ABS,
        early_stopping_min_delta_rel=EARLY_STOPPING_MIN_DELTA_REL,
        amp_enabled=amp_enabled,
        val_indices=data_ctx.val_indices,
        lr_scheduler_type=LR_SCHEDULER,
        reduce_lr_patience=REDUCE_LR_PATIENCE,
        reduce_lr_factor=REDUCE_LR_FACTOR,
        min_lr=MIN_LR,
        grad_clip_norm=GRAD_CLIP_NORM,
        train_window_sample_weights=norm_ctx.train_window_sample_weights,
        use_age_weighted_window_sampling=use_aw,
    )

    return TrainingContext(
        model=training["model"],
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        amp_enabled=amp_enabled,
        triton_available=triton_available,
        compile_applied=compile_applied,
        train_losses=training["train_losses"],
        r2_scores=training["r2_scores"],
        maes=training["maes"],
        best_loss=training["best_loss"],
        best_epoch=training["best_epoch"],
    )


def run_evaluation_stage(
    data_ctx: DataContext,
    norm_ctx: NormalizationContext,
    train_ctx: TrainingContext,
    hparam_ctx: HyperparameterContext | None,
    runtime_ctx: RuntimeContext,
    rng,
):
    """Run held-out evaluation, baseline comparison, and subject-example selection.

    Args:
        data_ctx (DataContext): Loaded data context.
        norm_ctx (NormalizationContext): Normalization context.
        train_ctx (TrainingContext): Outputs from training stage.
        runtime_ctx (RuntimeContext): Runtime context with selected device.
        rng (np.random.Generator): Random generator for bootstrap/examples.

    Returns:
        EvaluationContext: Evaluation metrics, baseline metrics, and subject examples.
    """
    log_stage("Held-out Evaluation", logger)
    logging.info("Test evaluation...")
    if isinstance(train_ctx.model, MILAgeRegressor):
        active_hparams = hparam_ctx.active_hparams if hparam_ctx is not None else {}
        # Keep MIL evaluation consistent with MIL tuning: use the tuned bag size
        # and sampling strategy when available.
        eval_bag_size = int(active_hparams.get("mil_bag_size", MIL_PSEUDO_BAG_MIN_WINDOWS))
        eval_sampling_strategy = str(active_hparams.get("mil_sampling_strategy", "sequential"))
        eval_allow_replacement = bool(
            active_hparams.get(
                "mil_allow_replacement_when_small",
                MIL_ALLOW_REPLACEMENT_WHEN_SMALL,
            )
        )
        logging.info(
            "MIL eval bags | bag_size=%d sampling_strategy=%s allow_replacement_when_small=%s",
            eval_bag_size,
            eval_sampling_strategy,
            eval_allow_replacement,
        )
        mil_eval = evaluate_mil_on_subject_bags(
            mil_model=train_ctx.model,
            criterion=train_ctx.criterion,
            x_mem=data_ctx.x_mem,
            y_mem=data_ctx.y_mem,
            eval_indices=data_ctx.test_indices,
            subject_codes=data_ctx.subject_codes,
            n_subjects=len(data_ctx.subject_codebook),
            batch_size=MIL_BAG_BATCH_SIZE,
            device=runtime_ctx.device,
            x_mean=norm_ctx.x_mean,
            x_std=norm_ctx.x_std,
            y_mean=norm_ctx.y_mean,
            y_std=norm_ctx.y_std,
            normalize_target=NORMALIZE_TARGET,
            rng=rng,
            pseudo_bag_min_windows=eval_bag_size,
            pseudo_bag_max_windows=eval_bag_size,
            allow_replacement_when_small=eval_allow_replacement,
            sampling_strategy=eval_sampling_strategy,
        )
        test_loss = mil_eval["test_loss"]
        test_r2 = mil_eval["test_r2"]
        test_mae = mil_eval["test_mae"]
        final_targets = mil_eval["final_targets"]
        final_preds = mil_eval["final_preds"]
        sum_true_by_subject = mil_eval["sum_true_by_subject"]
        sum_pred_by_subject = mil_eval["sum_pred_by_subject"]
        count_by_subject = mil_eval["count_by_subject"]
    else:
        test_loss, test_r2, test_mae, final_targets, final_preds, sum_true_by_subject, sum_pred_by_subject, count_by_subject = run_epoch_metrics(
            model=train_ctx.model,
            x_mem=data_ctx.x_mem,
            y_mem=data_ctx.y_mem,
            indices=data_ctx.test_indices,
            batch_size=BATCH_SIZE,
            device=runtime_ctx.device,
            criterion=train_ctx.criterion,
            amp_enabled=train_ctx.amp_enabled,
            subject_codes=data_ctx.subject_codes,
            n_subjects=len(data_ctx.subject_codebook),
            x_mean=norm_ctx.x_mean,
            x_std=norm_ctx.x_std,
            y_mean=norm_ctx.y_mean,
            y_std=norm_ctx.y_std,
            normalize_target=NORMALIZE_TARGET,
            plot_max_points=PLOT_MAX_POINTS,
            debug_chunk_log_every=DEBUG_CHUNK_LOG_EVERY,
        )
    logging.info("Test results | loss=%.4f R2=%.4f MAE=%.2f", test_loss, test_r2, test_mae)

    subj_mae = np.nan
    subj_r2 = np.nan
    subj_ids = []
    mae_ci = (np.nan, np.nan)
    r2_ci = (np.nan, np.nan)

    if sum_true_by_subject is not None:
        subj_true, subj_pred, subj_ids = compute_subject_level_metrics(
            sum_true_by_subject=sum_true_by_subject,
            sum_pred_by_subject=sum_pred_by_subject,
            count_by_subject=count_by_subject,
            subject_codebook=data_ctx.subject_codebook,
        )
        subj_mae = float(np.mean(np.abs(subj_true - subj_pred))) if subj_true.size > 0 else np.nan
        subj_r2 = compute_r2_from_arrays(subj_true, subj_pred)
        logging.info("Subject-level test | N=%d | R2: %.4f | MAE: %.2f", len(subj_ids), subj_r2, subj_mae)

        if BOOTSTRAP_ENABLED and subj_true.size > 1:
            mae_ci, r2_ci = bootstrap_ci_subject_metrics(
                y_true_subject=subj_true,
                y_pred_subject=subj_pred,
                rng=rng,
                n_boot=BOOTSTRAP_ITERATIONS,
                confidence=BOOTSTRAP_CONFIDENCE,
            )
            logging.info(
                "Bootstrap %.1f%% CI (subject-level) | MAE: [%.2f, %.2f] | R2: [%.4f, %.4f]",
                BOOTSTRAP_CONFIDENCE * 100.0,
                mae_ci[0],
                mae_ci[1],
                r2_ci[0],
                r2_ci[1],
            )
    else:
        logging.warning("Subject-level metrics/CI skipped: subject aggregation not available.")

    log_stage("Baseline Comparison", logger)
    baseline_pred, baseline_loss, baseline_r2, baseline_mae = compute_constant_baseline(
        y_mem=data_ctx.y_mem,
        train_indices=data_ctx.train_indices,
        test_indices=data_ctx.test_indices,
        debug_chunk_log_every=DEBUG_CHUNK_LOG_EVERY,
    )
    logging.info(
        "Baseline (constant train-mean age %.2f) | Loss: %.4f | R2: %.4f | MAE: %.2f",
        baseline_pred,
        baseline_loss,
        baseline_r2,
        baseline_mae,
    )
    logging.info(
        "Model lift over baseline | ΔMAE: %.2f | ΔR2: %.4f",
        baseline_mae - test_mae,
        test_r2 - baseline_r2 if np.isfinite(test_r2) and np.isfinite(baseline_r2) else np.nan,
    )

    subject_examples = []
    if len(data_ctx.subject_codebook) > 0:
        subject_examples = build_subject_examples(
            model=train_ctx.model,
            x_mem=data_ctx.x_mem,
            y_mem=data_ctx.y_mem,
            test_indices=data_ctx.test_indices,
            subject_codes=data_ctx.subject_codes,
            subject_codebook=data_ctx.subject_codebook,
            batch_size=BATCH_SIZE,
            device=runtime_ctx.device,
            amp_enabled=train_ctx.amp_enabled,
            x_mean=norm_ctx.x_mean,
            x_std=norm_ctx.x_std,
            y_mean=norm_ctx.y_mean,
            y_std=norm_ctx.y_std,
            normalize_target=NORMALIZE_TARGET,
            rng=rng,
            subject_count=max(2, SUBJECT_EXAMPLE_COUNT),
            max_windows=SUBJECT_EXAMPLE_MAX_WINDOWS,
        )
        logger.debug("Subject examples for plot: %d", len(subject_examples))
    else:
        logging.warning("Subject example plot skipped: no subject codes available.")

    return EvaluationContext(
        test_loss=float(test_loss),
        test_r2=float(test_r2),
        test_mae=float(test_mae),
        final_targets=final_targets,
        final_preds=final_preds,
        subj_mae=float(subj_mae) if np.isfinite(subj_mae) else np.nan,
        subj_r2=float(subj_r2) if np.isfinite(subj_r2) else np.nan,
        subj_ids=subj_ids,
        mae_ci=(float(mae_ci[0]), float(mae_ci[1])),
        r2_ci=(float(r2_ci[0]), float(r2_ci[1])),
        baseline_pred=float(baseline_pred),
        baseline_loss=float(baseline_loss),
        baseline_r2=float(baseline_r2) if np.isfinite(baseline_r2) else np.nan,
        baseline_mae=float(baseline_mae),
        subject_examples=subject_examples,
    )


def save_artifacts_and_summary(
    run_ctx: RunContext,
    runtime_ctx: RuntimeContext,
    data_ctx: DataContext,
    norm_ctx: NormalizationContext,
    hparam_ctx: HyperparameterContext,
    train_ctx: TrainingContext,
    eval_ctx: EvaluationContext,
    model_label="cnn",
):
    """Persist model weights, plots, and textual/JSON run summaries.

    Args:
        run_ctx (RunContext): Run metadata context.
        runtime_ctx (RuntimeContext): Runtime context.
        data_ctx (DataContext): Data context.
        norm_ctx (NormalizationContext): Normalization / sampling context.
        hparam_ctx (HyperparameterContext): Hyperparameter context.
        train_ctx (TrainingContext): Training stage outputs.
        eval_ctx (EvaluationContext): Evaluation stage outputs.

    Returns:
        dict[str, Any]: Summary payload saved to disk.
    """
    log_stage("Artifacts + Reports", logger)
    model_to_save = train_ctx.model._orig_mod if hasattr(train_ctx.model, "_orig_mod") else train_ctx.model
    model_name, model_ext = os.path.splitext(MODEL_SAVE_NAME)
    model_path = os.path.join(run_ctx.run_output_dir, f"{model_name}_{model_label}_{run_ctx.run_tag}{model_ext}")
    torch.save(model_to_save.state_dict(), model_path)
    logging.info("Model saved to: %s", model_path)

    report_path = save_training_report(
        run_output_dir=run_ctx.run_output_dir,
        run_tag=run_ctx.run_tag,
        report_save_name=REPORT_SAVE_NAME,
        train_losses=train_ctx.train_losses,
        r2_scores=train_ctx.r2_scores,
        maes=train_ctx.maes,
        baseline_loss=eval_ctx.baseline_loss,
        baseline_mae=eval_ctx.baseline_mae,
        baseline_r2=eval_ctx.baseline_r2,
        test_mae=eval_ctx.test_mae,
        final_targets=eval_ctx.final_targets,
        final_preds=eval_ctx.final_preds,
    )

    subject_report_path = save_subject_examples_report(
        run_output_dir=run_ctx.run_output_dir,
        run_tag=run_ctx.run_tag,
        subject_report_save_name=SUBJECT_REPORT_SAVE_NAME,
        subject_examples=eval_ctx.subject_examples,
    )

    run_end_time = datetime.now()
    duration_seconds = (run_end_time - run_ctx.run_start_time).total_seconds()

    summary_payload = {
        "model_label": model_label,
        "run_timestamp": run_end_time.isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
        "device": str(runtime_ctx.device),
        "gpu_name": runtime_ctx.gpu_name,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "learning_rate": hparam_ctx.active_lr,
        "active_hyperparameters": hparam_ctx.active_hparams,
        "use_huber_loss": hparam_ctx.active_use_huber,
        "huber_beta": hparam_ctx.active_huber_beta,
        "normalize_input": NORMALIZE_INPUT,
        "normalize_target": NORMALIZE_TARGET,
        "clip_predicted_age_after_denorm": CLIP_PREDICTED_AGE_AFTER_DENORM,
        "min_predicted_age_years": MIN_PREDICTED_AGE_YEARS,
        "use_age_stratified_split": USE_AGE_STRATIFIED_SPLIT,
        "age_stratified_split_applied": data_ctx.subject_stratum_merged is not None,
        "stratify_age_bin_years": STRATIFY_AGE_BIN_YEARS,
        "stratify_tail_low_max_age": STRATIFY_TAIL_LOW_MAX_AGE,
        "stratify_tail_high_min_age": STRATIFY_TAIL_HIGH_MIN_AGE,
        "stratify_min_subjects_per_stratum": STRATIFY_MIN_SUBJECTS_PER_STRATUM,
        "age_weighted_window_sampling": AGE_WEIGHTED_WINDOW_SAMPLING,
        "age_weighted_window_weights_built": norm_ctx.train_window_sample_weights is not None,
        "mil_inverse_frequency_subject_sampling": MIL_INVERSE_FREQUENCY_SUBJECT_SAMPLING,
        "mil_subject_draws_per_epoch_config": MIL_SUBJECT_DRAWS_PER_EPOCH,
        "subject_balanced_training": SUBJECT_BALANCED_TRAINING,
        "max_windows_per_subject_per_epoch": hparam_ctx.active_max_windows_per_subject,
        "tf32_enabled": USE_TF32,
        "torch_compile_enabled": train_ctx.compile_applied,
        "triton_available": train_ctx.triton_available,
        "early_stopping_enabled": EARLY_STOPPING_ENABLED,
        "bootstrap_enabled": BOOTSTRAP_ENABLED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "n_samples": int(data_ctx.n_samples),
        "n_train_windows": int(data_ctx.train_indices.size),
        "n_test_windows": int(data_ctx.test_indices.size),
        "n_val_windows": int(data_ctx.val_indices.size) if data_ctx.val_indices is not None else 0,
        "n_train_subjects_csv": int(len(data_ctx.train_age_map)),
        "n_test_subjects_csv": int(len(data_ctx.test_age_map)),
        "n_test_subjects_evaluated": int(len(eval_ctx.subj_ids)),
        "epochs_completed": int(len(train_ctx.train_losses)),
        "best_epoch": int(train_ctx.best_epoch),
        "best_train_loss": float(train_ctx.best_loss),
        "test_loss": float(eval_ctx.test_loss),
        "test_r2": float(eval_ctx.test_r2) if np.isfinite(eval_ctx.test_r2) else np.nan,
        "test_mae": float(eval_ctx.test_mae),
        "baseline_pred_age": float(eval_ctx.baseline_pred),
        "baseline_loss": float(eval_ctx.baseline_loss),
        "baseline_r2": float(eval_ctx.baseline_r2) if np.isfinite(eval_ctx.baseline_r2) else np.nan,
        "baseline_mae": float(eval_ctx.baseline_mae),
        "delta_mae": float(eval_ctx.baseline_mae - eval_ctx.test_mae),
        "delta_r2": float(eval_ctx.test_r2 - eval_ctx.baseline_r2)
        if np.isfinite(eval_ctx.test_r2) and np.isfinite(eval_ctx.baseline_r2)
        else np.nan,
        "subject_r2": float(eval_ctx.subj_r2) if np.isfinite(eval_ctx.subj_r2) else np.nan,
        "subject_mae": float(eval_ctx.subj_mae) if np.isfinite(eval_ctx.subj_mae) else np.nan,
        "bootstrap_mae_ci": [float(eval_ctx.mae_ci[0]), float(eval_ctx.mae_ci[1])],
        "bootstrap_r2_ci": [float(eval_ctx.r2_ci[0]), float(eval_ctx.r2_ci[1])],
        "model_path": model_path,
        "report_path": report_path,
        "subject_report_path": subject_report_path,
        "run_output_dir": run_ctx.run_output_dir,
        "memmap_source_dir": runtime_ctx.memmap_root,
        "age_target_cache": runtime_ctx.age_cache_path,
    }
    log_stage("Run Summary", logger)
    save_run_summary(run_ctx.run_output_dir, summary_payload, run_ctx.run_tag)

    logging.info("Training complete.")
    return summary_payload


def _resolve_model_modes(args):
    """Resolve requested model execution mode into ordered run labels.

    Args:
        args (argparse.Namespace): Parsed CLI args.

    Returns:
        list[str]: Ordered list of labels from {"cnn", "mil"}.
    """
    mode = getattr(args, "model_mode", None)
    if mode is None:
        # Backward compatibility with older --mil-finetune behavior.
        mode = "mil" if bool(getattr(args, "mil_finetune", False)) else "cnn"

    if mode == "cnn":
        return ["cnn"]
    if mode == "mil":
        return ["mil"]
    if mode == "both":
        return ["cnn", "mil"]
    raise ValueError(f"Unsupported model mode: {mode}")


def _build_model_args(args, model_label):
    """Clone args and set per-model runtime flags.

    Args:
        args (argparse.Namespace): Parsed base CLI args.
        model_label (str): Either ``cnn`` or ``mil``.

    Returns:
        argparse.Namespace: Adjusted args for the selected model branch.
    """
    model_args = copy.deepcopy(args)
    model_args.mil_finetune = (model_label == "mil")
    return model_args


def _resolve_model_hparams_file(args, model_label: str) -> str | None:
    """Resolve per-model hyperparameter file for `both` runs.

    Precedence (per model):
    - --cnn-hparams-file / --mil-hparams-file (when provided)
    - --hparams-file (shared)
    - defaults/default_hyperparameters.json (handled by select_hyperparameters)
    """
    if model_label == "cnn":
        return getattr(args, "cnn_hparams_file", None) or getattr(args, "hparams_file", None)
    if model_label == "mil":
        return getattr(args, "mil_hparams_file", None) or getattr(args, "hparams_file", None)
    return getattr(args, "hparams_file", None)


def execute_full_workflow(args):
    """Execute all pipeline stages from setup through artifact generation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    run_ctx = initialize_run_context()
    runtime_ctx = setup_runtime_context(args)
    data_ctx = load_data_context(runtime_ctx, args)
    norm_ctx = setup_normalization_context(data_ctx, run_ctx.rng, args=args)
    model_labels = _resolve_model_modes(args)

    # Preconditions for defaults: if the run includes MIL, require a pretrained
    # CNN checkpoint either via CLI or the repository `defaults/` folder. If the
    # run mode is `both`, prefer explicit hyperparameters via `--hparams-file`
    # or a `defaults/default_hyperparameters.json` file in the repo root.
    if "mil" in model_labels:
        mil_candidate = getattr(args, "mil_pretrained_model", None) or DEFAULT_MODEL_PATH
        if not mil_candidate or not os.path.exists(mil_candidate):
            raise FileNotFoundError(
                "MIL mode requires a pretrained CNN checkpoint.\n"
                "Provide one with the CLI flag `--mil-pretrained-model PATH` or\n"
                f"place a file named '{os.path.basename(DEFAULT_MODEL_PATH)}' in the repository 'defaults' folder: {os.path.dirname(DEFAULT_MODEL_PATH)}\n"
                "Example: defaults/default_model.pt"
            )

    if (getattr(args, "model_mode", None) == "both") and (not bool(getattr(args, "tune", False))):
        # When running both modes back-to-back we expect a hyperparameters JSON
        # to be provided either via `--hparams-file` or defaults/default_hyperparameters.json.
        has_any_cli_hparams = bool(
            getattr(args, "hparams_file", None)
            or getattr(args, "cnn_hparams_file", None)
            or getattr(args, "mil_hparams_file", None)
        )
        if (not has_any_cli_hparams) and (not os.path.exists(DEFAULT_HPARAMS_PATH)):
            repo_hparams_path = os.path.join(PROJECT_DIR, "output", "hparams", BEST_HPARAMS_FILE)
            if not os.path.exists(repo_hparams_path):
                raise FileNotFoundError(
                    "`both` mode requires a hyperparameters JSON file.\n"
                    "Provide one with the CLI flag `--hparams-file PATH` (shared) or\n"
                    "provide per-model files with `--cnn-hparams-file PATH` and/or `--mil-hparams-file PATH`, or\n"
                    f"place a file named '{os.path.basename(DEFAULT_HPARAMS_PATH)}' in the repository 'defaults' folder: {os.path.dirname(DEFAULT_HPARAMS_PATH)}\n"
                    "Example: defaults/default_hyperparameters.json"
                )

    # Global hyperparameters context:
    # - For single-mode runs this is the only context.
    # - For `both` mode we still build it once so downstream summary keys are
    #   consistent, but each model run may override it with per-model files.
    hparam_ctx = select_hyperparameters(args, data_ctx, norm_ctx, runtime_ctx, run_ctx=run_ctx)

    # Default behavior: when tuning is requested, stop after saving tuned
    # hyperparameters so the user can run a separate non-tuning training pass.
    if bool(getattr(args, "tune", False)) and not bool(getattr(args, "tune_and_train", False)):
        logger.info(
            "Tuning-only run complete. Skipping training/evaluation. "
            "Use --tune-and-train to continue into full model training after tuning."
        )
        return {
            "status": "tune_only_complete",
            "active_hyperparameters": hparam_ctx.active_hparams,
            "run_output_dir": run_ctx.run_output_dir,
        }

    model_summaries = {}

    if (getattr(args, "model_mode", None) == "both") and (not bool(getattr(args, "tune", False))):
        logger.info(
            "Both mode: per-model hparams | cnn=%s | mil=%s",
            _resolve_model_hparams_file(args, "cnn") or "defaults",
            _resolve_model_hparams_file(args, "mil") or "defaults",
        )

    for model_label in model_labels:
        log_stage(f"Model Run: {model_label.upper()}", logger)
        model_output_dir = os.path.join(run_ctx.run_output_dir, model_label)
        os.makedirs(model_output_dir, exist_ok=True)

        model_run_ctx = RunContext(
            run_start_time=run_ctx.run_start_time,
            run_tag=f"{run_ctx.run_tag}_{model_label}",
            rng=run_ctx.rng,
            run_output_dir=model_output_dir,
        )
        model_args = _build_model_args(args, model_label)
        # Allow separate tuned hyperparameter files per model in `both` mode.
        # This enables a fair "best CNN" vs "best CNN+MIL" comparison.
        if (getattr(args, "model_mode", None) == "both") and (not bool(getattr(args, "tune", False))):
            model_args.hparams_file = _resolve_model_hparams_file(args, model_label)
            model_hparam_ctx = select_hyperparameters(model_args, data_ctx, norm_ctx, runtime_ctx, run_ctx=run_ctx)
        else:
            model_hparam_ctx = hparam_ctx

        train_ctx = run_training_stage(data_ctx, norm_ctx, model_hparam_ctx, runtime_ctx, run_ctx.rng, model_args)
        eval_ctx = run_evaluation_stage(data_ctx, norm_ctx, train_ctx, model_hparam_ctx, runtime_ctx, run_ctx.rng)
        summary_payload = save_artifacts_and_summary(
            model_run_ctx,
            runtime_ctx,
            data_ctx,
            norm_ctx,
            model_hparam_ctx,
            train_ctx,
            eval_ctx,
            model_label=model_label,
        )
        model_summaries[model_label] = summary_payload

    if "cnn" in model_summaries and "mil" in model_summaries:
        log_stage("CNN vs MIL Comparison", logger)
        comparison_txt, comparison_json, _ = save_model_comparison_summary(
            data_dir=run_ctx.run_output_dir,
            run_tag=run_ctx.run_tag,
            cnn_summary=model_summaries["cnn"],
            mil_summary=model_summaries["mil"],
        )
        comparison_plot = save_model_comparison_report(
            run_output_dir=run_ctx.run_output_dir,
            run_tag=run_ctx.run_tag,
            cnn_summary=model_summaries["cnn"],
            mil_summary=model_summaries["mil"],
        )
        logger.info(
            "CNN vs MIL comparison artifacts saved | txt=%s | json=%s | plot=%s",
            comparison_txt,
            comparison_json,
            comparison_plot,
        )
