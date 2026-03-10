"""Top-level staged workflow for training, evaluation, and artifact export."""

import logging
import os
from datetime import datetime
from time import perf_counter

import numpy as np
import torch

from cnn_age_project.config import (
    AGE_TARGET_CACHE_FILE,
    BATCH_SIZE,
    BEST_HPARAMS_FILE,
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
    MAX_WINDOWS_PER_SUBJECT_PER_EPOCH,
    METRIC_EVERY_N_EPOCHS,
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
    PROJECT_DIR,
)
from cnn_age_project.data.data_io import (
    build_or_load_targets_and_split,
    load_memmap_arrays,
    load_subject_age_map,
    validate_required_project_files,
)
from cnn_age_project.data.dataset import build_subject_group_index
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
    build_tuning_candidates,
    get_default_hyperparameters,
    load_best_hyperparameters_if_available,
    save_best_hyperparameters,
    save_run_summary,
)
from cnn_age_project.training.trainer import run_tuning_trial, setup_model_and_optimizers, train_model
from cnn_age_project.utils.utils import log_stage, make_tqdm, select_device
from cnn_age_project.visualization.plots import save_subject_examples_report, save_training_report
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

    logging.info("==== EEG CNN Age Prediction Pipeline Started ====")
    logging.info(
        "Config | batch=%d epochs=%d lr=%.4g tf32=%s compile=%s huber=%s norm_x=%s norm_y=%s balanced=%s log_level=%s",
        BATCH_SIZE,
        EPOCHS,
        LR,
        USE_TF32,
        USE_TORCH_COMPILE,
        USE_HUBER_LOSS,
        NORMALIZE_INPUT,
        NORMALIZE_TARGET,
        SUBJECT_BALANCED_TRAINING,
        LOG_LEVEL,
    )
    logging.info("Project folders | input=%s | output=%s", INPUT_DIR, OUTPUT_DIR)
    logging.info("Run artifact folder: %s", run_output_dir)

    return RunContext(
        run_start_time=run_start_time,
        run_tag=run_tag,
        rng=rng,
        run_output_dir=run_output_dir,
    )


def setup_runtime_context():
    """Validate filesystem prerequisites and initialize device/runtime paths.

    Args:
        None

    Returns:
        RuntimeContext: Runtime device information and cache paths.
    """
    log_stage("Preflight Checks", logger)
    memmap_root = validate_required_project_files(INPUT_DIR, OUTPUT_DIR)
    logging.info("Memmap source directory: %s", memmap_root)

    log_stage("Device Setup", logger)
    device, gpu_name = select_device(USE_TF32, logger)

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


def load_data_context(runtime_ctx: RuntimeContext):
    """Load memmaps, build/load split caches, and return train/test indices.

    Args:
        runtime_ctx (RuntimeContext): Runtime context with memmap root and cache paths.

    Returns:
        DataContext: Loaded arrays, split indices, and subject metadata.
    """
    log_stage("Memmap Loading", logger)
    logging.info("Opening memmap files (sequential streaming mode)...")
    memmaps = load_memmap_arrays(runtime_ctx.memmap_root)
    x_mem = memmaps["x_mem"]
    n_samples = memmaps["n_samples"]
    window_len = memmaps["window_len"]
    meta_path = memmaps["meta_path"]

    logging.info("Detected samples: %d", n_samples)
    logging.info("Detected window length: %d", window_len)
    logging.info("Memmap tensors ready (contiguous streaming batches).")

    if not os.path.exists(TRAIN_KEY_CSV) or not os.path.exists(TEST_KEY_CSV):
        raise FileNotFoundError("Training/testing key CSV files were not found in input folder.")

    log_stage("Label + Split Preparation", logger)
    train_age_map = load_subject_age_map(TRAIN_KEY_CSV)
    test_age_map = load_subject_age_map(TEST_KEY_CSV)
    overlap_subjects = set(train_age_map).intersection(set(test_age_map))
    if len(overlap_subjects) > 0:
        raise ValueError(f"Train/test subject overlap detected: {len(overlap_subjects)} subjects.")

    split_codes, age_targets, subject_codes, subject_codebook = build_or_load_targets_and_split(
        meta_path=meta_path,
        n_samples=n_samples,
        train_age_map=train_age_map,
        test_age_map=test_age_map,
        split_cache_path=runtime_ctx.split_cache_path,
        age_cache_path=runtime_ctx.age_cache_path,
        subject_code_cache_path=runtime_ctx.subject_code_cache_path,
        subject_codebook_path=runtime_ctx.subject_codebook_path,
    )

    valid_age = np.isfinite(age_targets)
    train_indices = np.where((split_codes == 1) & valid_age)[0]
    test_indices = np.where((split_codes == 2) & valid_age)[0]

    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError("No train/test windows with valid age labels found.")

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    y_mem = age_targets

    logging.info("Split windows with age labels | train: %d | test: %d", train_indices.size, test_indices.size)
    logger.info("Unique subjects discovered in metadata: %d", len(subject_codebook))

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
    )


def setup_normalization_context(data_ctx: DataContext, rng):
    """Compute normalization statistics and optional subject-balanced index structures.

    Args:
        data_ctx (DataContext): Loaded training/testing data context.
        rng (np.random.Generator): Random generator for sampling normalization windows.

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
    logger.info("Target normalization | enabled=%s mean=%.4f std=%.4f", NORMALIZE_TARGET, y_mean, y_std)

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
    logger.info("Input normalization | enabled=%s mean=%.6f std=%.6f", NORMALIZE_INPUT, x_mean, x_std)

    balanced_sorted_indices = None
    balanced_offsets = None
    balanced_counts = None
    if SUBJECT_BALANCED_TRAINING:
        balanced_sorted_indices, balanced_offsets, balanced_counts = build_subject_group_index(
            train_indices=data_ctx.train_indices,
            subject_codes=data_ctx.subject_codes,
            n_subjects=len(data_ctx.subject_codebook),
        )
        logger.info(
            "Subject-balanced training enabled | max windows/subject/epoch=%d | subjects with windows=%d",
            MAX_WINDOWS_PER_SUBJECT_PER_EPOCH,
            int(np.sum(balanced_counts > 0)),
        )

    return NormalizationContext(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        balanced_sorted_indices=balanced_sorted_indices,
        balanced_offsets=balanced_offsets,
        balanced_counts=balanced_counts,
    )


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
    )
    # Determine where to source or save best hyperparameters.
    # Priority order when not tuning:
    # 1) `--hparams-file` provided by user
    # 2) repo-provided `output/hparams/best_hyperparameters.json`
    # 3) output folder `output/hparams/best_hyperparameters.json`
    best_hparams_path = os.path.join(OUTPUT_DIR, "hparams", BEST_HPARAMS_FILE)
    repo_hparams_path = os.path.join(PROJECT_DIR, "output", "hparams", BEST_HPARAMS_FILE)
    if args.hparams_file:
        best_hparams_path = args.hparams_file

    log_stage("Hyperparameter Selection", logger)
    if args.tune:
        candidates = build_tuning_candidates(defaults)
        max_trials = min(args.tune_max_trials, len(candidates))
        logger.info("Hyperparameter tuning enabled | trials=%d | tune_epochs=%d", max_trials, args.tune_epochs)
        tuning_results = []
        tuning_start = perf_counter()
        best_so_far = None

        trial_bar = make_tqdm(
            enumerate(candidates[:max_trials], start=1),
            total=max_trials,
            desc="Hyperparameter Tuning",
            unit="trial",
            position=0,
            leave=True,
        )

        for trial_idx, candidate in trial_bar:
            trial_result = run_tuning_trial(
                trial_idx=trial_idx,
                total_trials=max_trials,
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
            )
            tuning_results.append(trial_result)

            if best_so_far is None or trial_result["test_mae"] < best_so_far["test_mae"]:
                best_so_far = trial_result
                logger.info(
                    "[Tune] New best at trial %d | MAE=%.3f R2=%.4f hparams=%s",
                    trial_idx,
                    trial_result["test_mae"],
                    trial_result["test_r2"],
                    trial_result["hparams"],
                )

            trial_bar.set_postfix(best_mae=f"{best_so_far['test_mae']:.3f}" if best_so_far else "n/a")

        best_trial = min(tuning_results, key=lambda item: item["test_mae"])
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
            "Best tuning hyperparameters selected | MAE=%.3f R2=%.4f hparams=%s | total_tune_time=%.1fs",
            best_trial["test_mae"],
            best_trial["test_r2"],
            active_hparams,
            perf_counter() - tuning_start,
        )
    else:
        # Prefer repo-level supplied hyperparameters if available and no explicit file provided.
        if not args.hparams_file and os.path.exists(repo_hparams_path):
            best_hparams_path = repo_hparams_path
        active_hparams = load_best_hyperparameters_if_available(best_hparams_path, defaults)
        logger.info("Active hyperparameters for training: %s", active_hparams)

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
):
    """Initialize trainable objects and execute the full training phase.

    Args:
        data_ctx (DataContext): Loaded data context.
        norm_ctx (NormalizationContext): Normalization and sampling context.
        hparam_ctx (HyperparameterContext): Selected hyperparameter context.
        runtime_ctx (RuntimeContext): Runtime context with selected device.
        rng (np.random.Generator): Random generator used during sampling.

    Returns:
        TrainingContext: Training artifacts and metrics histories.
    """
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
    )

    log_stage("Model Training", logger)
    logging.info("Starting training...")

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
    logging.info("Running held-out test-set evaluation...")
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
    logging.info("Test | Loss: %.4f | R2: %.4f | MAE: %.2f", test_loss, test_r2, test_mae)

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
        logging.info("Prepared %d random subject examples for plotting.", len(subject_examples))
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
    hparam_ctx: HyperparameterContext,
    train_ctx: TrainingContext,
    eval_ctx: EvaluationContext,
):
    """Persist model weights, plots, and textual/JSON run summaries.

    Args:
        run_ctx (RunContext): Run metadata context.
        runtime_ctx (RuntimeContext): Runtime context.
        data_ctx (DataContext): Data context.
        hparam_ctx (HyperparameterContext): Hyperparameter context.
        train_ctx (TrainingContext): Training stage outputs.
        eval_ctx (EvaluationContext): Evaluation stage outputs.

    Returns:
        None
    """
    log_stage("Artifacts + Reports", logger)
    model_to_save = train_ctx.model._orig_mod if hasattr(train_ctx.model, "_orig_mod") else train_ctx.model
    model_name, model_ext = os.path.splitext(MODEL_SAVE_NAME)
    model_path = os.path.join(run_ctx.run_output_dir, f"{model_name}_{run_ctx.run_tag}{model_ext}")
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
        "run_timestamp": run_end_time.isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
        "device": str(runtime_ctx.device),
        "gpu_name": runtime_ctx.gpu_name,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "learning_rate": hparam_ctx.active_lr,
        "use_huber_loss": hparam_ctx.active_use_huber,
        "huber_beta": hparam_ctx.active_huber_beta,
        "normalize_input": NORMALIZE_INPUT,
        "normalize_target": NORMALIZE_TARGET,
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

    logging.info("==== Training Complete ====")


def execute_full_workflow(args):
    """Execute all pipeline stages from setup through artifact generation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    run_ctx = initialize_run_context()
    runtime_ctx = setup_runtime_context()
    data_ctx = load_data_context(runtime_ctx)
    norm_ctx = setup_normalization_context(data_ctx, run_ctx.rng)
    hparam_ctx = select_hyperparameters(args, data_ctx, norm_ctx, runtime_ctx, run_ctx=run_ctx)
    train_ctx = run_training_stage(data_ctx, norm_ctx, hparam_ctx, runtime_ctx, run_ctx.rng)
    eval_ctx = run_evaluation_stage(data_ctx, norm_ctx, train_ctx, runtime_ctx, run_ctx.rng)
    save_artifacts_and_summary(run_ctx, runtime_ctx, data_ctx, hparam_ctx, train_ctx, eval_ctx)
