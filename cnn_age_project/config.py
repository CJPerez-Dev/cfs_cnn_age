"""Centralized static configuration for paths, training, and runtime behavior."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Immutable project configuration values used across the pipeline."""
    project_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    x_file: str = "X_T1281.fp16.npy"
    y_file: str = "y_T1281.int16.npy"
    idx_file: str = "idx_T1281.int32.npy"
    meta_file: str = "meta_T1281.csv"

    split_cache_file: str = "split_codes_T1281.uint8.npy"
    age_target_cache_file: str = "y_age_T1281.float32.npy"
    subject_code_cache_file: str = "subject_codes_T1281.int32.npy"
    subject_codebook_file: str = "subject_codebook_T1281.json"

    batch_size: int = 1024
    # With capped CNN age-weighted epochs (cnn_samples_per_epoch), more epochs are
    # typical to match prior total training exposure; tune early_stopping_* if needed.
    epochs: int = 100
    lr: float = 3e-4

    use_huber_loss: bool = True
    huber_beta: float = 1.0

    normalize_target: bool = True
    # After denormalizing (or when targets are already in years), clamp predictions
    # so reported ages are not below min_predicted_age_years (e.g. non-negative).
    clip_predicted_age_after_denorm: bool = True
    min_predicted_age_years: float = 0.0
    normalize_input: bool = True
    norm_sample_windows: int = 100_000
    norm_eps: float = 1e-6

    subject_balanced_training: bool = True
    max_windows_per_subject_per_epoch: int = 2000

    use_torch_compile: bool = True
    use_tf32: bool = True
    torch_compile_mode: str = "reduce-overhead"
    torch_compile_dynamic: bool = True
    metric_every_n_epochs: int = 1
    plot_max_points: int = 100_000
    random_seed: int = 42
    subject_example_count: int = 32
    subject_example_max_windows: int = 300

    mil_pseudo_bag_size: int = 256
    mil_pseudo_bag_min_windows: int = 256
    mil_pseudo_bag_max_windows: int = 256
    # When False, random bags use only unique windows (no upsampling); bags may
    # be shorter than mil_bag_size for subjects with few windows.
    mil_allow_replacement_when_small: bool = False
    mil_bag_batch_size: int = 16
    mil_finetune_encoder_lr: float = 1e-5
    # Number of warmup epochs where the MIL encoder is effectively frozen
    # (encoder parameter group learning rate set to 0). After this many
    # epochs, the encoder learning rate is restored and full fine-tuning
    # continues.
    mil_finetune_encoder_warmup_epochs: int = 5
    mil_finetune_head_lr: float = 5e-5
    mil_finetune_weight_decay: float = 1e-2

    log_level: str = "INFO"
    debug_chunk_log_every: int = 10

    bootstrap_enabled: bool = True
    bootstrap_iterations: int = 1000
    bootstrap_confidence: float = 0.95

    early_stopping_enabled: bool = True
    early_stopping_patience: int = 6
    early_stopping_min_epochs: int = 10
    early_stopping_min_delta_rel: float = 1e-3
    early_stopping_min_delta_abs: float = 1e-4

    # Validation set: if set (e.g. "AgeValidation_Key.csv"), split_code 3 is used for validation
    validation_key_filename: str | None = None

    # Auto 70/15/15 subject split from metadata (and optional single key file). When True,
    # AgeTraining_Key.csv and AgeTesting_Key.csv are not required; subjects are split by ratio.
    use_auto_split: bool = True
    split_ratio_train: float = 0.70
    split_ratio_val: float = 0.15
    split_ratio_test: float = 0.15
    # Optional single CSV with all subjects + age (SubjectID, VariableValue). If None and
    # use_auto_split, age is read from metadata if a column "age", "Age", or "VariableValue" exists.
    subject_key_filename: str | None = None

    # Age-stratified train/val/test when use_auto_split is True (ignored for manual key CSVs).
    # Interior: 5-year bands between tail boundaries; tails are single strata each. Sparse bands
    # are merged until each stratum has at least stratify_min_subjects_per_stratum subjects.
    use_age_stratified_split: bool = True
    stratify_age_bin_years: float = 5.0
    stratify_tail_low_max_age: float = 20.0
    # Ages >= this form one collapsed "old tail" stratum (with 5-year bins below, young tail below tail_low).
    stratify_tail_high_min_age: float = 80.0
    stratify_min_subjects_per_stratum: int = 3

    # MIL training: each epoch draw subjects with replacement, prob ∝ 1 / (subjects in age stratum).
    # Gives rare age bins equal aggregate sampling mass per epoch. Val/test MIL eval stays unweighted.
    mil_inverse_frequency_subject_sampling: bool = True
    # None = one draw per eligible train subject per epoch (same count as "all subjects", but weighted).
    mil_subject_draws_per_epoch: int | None = None

    # CNN training: sample windows with probability ∝ inverse stratum frequency (WeightedRandomSampler-style).
    # Takes precedence over subject_balanced_training when both are enabled.
    age_weighted_window_sampling: bool = True
    # Stochastic draws per CNN epoch when age_weighted_window_sampling is True (fresh weighted shuffle each epoch).
    # None = use every train window once per epoch (legacy). Use scripts/report_train_stratum_window_counts.py
    # for a data-driven n_min * B estimate (~2M for many CFS-style splits).
    cnn_samples_per_epoch: int | None = 2_000_000

    # Learning rate scheduler: "none", "plateau" (ReduceLROnPlateau on val loss), or "cosine"
    lr_scheduler: str = "plateau"
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

    # Gradient clipping (0 = disabled)
    grad_clip_norm: float = 1.0

    # CNN optimizer weight decay
    cnn_weight_decay: float = 1e-2

    # MIL early stopping (fine-tune loop)
    mil_early_stopping_enabled: bool = True
    mil_early_stopping_patience: int = 3
    mil_early_stopping_min_epochs: int = 2
    mil_early_stopping_min_delta_abs: float = 1e-4
    mil_early_stopping_min_delta_rel: float = 1e-3

    # MIL bag regressor dropout
    mil_regressor_dropout: float = 0.1

    model_save_name: str = "cnn_age_model.pt"
    report_save_name: str = "cnn_training_report.png"
    subject_report_save_name: str = "cnn_subject_examples.png"
    run_summary_txt_name: str = "cnn_run_summary.txt"
    run_summary_json_name: str = "cnn_run_summary.json"
    run_summary_history_dir: str = "run_summaries"
    best_hparams_file: str = "best_hyperparameters.json"
    tuning_results_file: str = "tuning_results.json"

    # Defaults folder for repository-provided fallbacks (user may populate)
    defaults_dir_name: str = "defaults"
    default_model_file: str = "default_model.pt"
    default_hparams_file: str = "default_hyperparameters.json"

    @property
    def input_dir(self):
        """Absolute path to the input data directory.

        Args:
            None

        Returns:
            str: Input directory path.
        """
        return os.path.join(self.project_dir, "input")

    @property
    def output_dir(self):
        """Absolute path to the output artifacts directory.

        Args:
            None

        Returns:
            str: Output directory path.
        """
        return os.path.join(self.project_dir, "output")

    @property
    def train_key_csv(self):
        """Absolute path to the training-subject age key CSV.

        Args:
            None

        Returns:
            str: Training key CSV path.
        """
        return os.path.join(self.input_dir, "AgeTraining_Key.csv")

    @property
    def test_key_csv(self):
        """Absolute path to the testing-subject age key CSV.

        Args:
            None

        Returns:
            str: Testing key CSV path.
        """
        return os.path.join(self.input_dir, "AgeTesting_Key.csv")

    @property
    def validation_key_csv(self):
        """Optional path to validation-subject age key CSV (None = no validation set)."""
        if self.validation_key_filename is None:
            return None
        return os.path.join(self.input_dir, self.validation_key_filename)


CONFIG = Config()

PROJECT_DIR = CONFIG.project_dir
INPUT_DIR = CONFIG.input_dir
OUTPUT_DIR = CONFIG.output_dir

X_FILE = CONFIG.x_file
Y_FILE = CONFIG.y_file
IDX_FILE = CONFIG.idx_file
META_FILE = CONFIG.meta_file

TRAIN_KEY_CSV = CONFIG.train_key_csv
TEST_KEY_CSV = CONFIG.test_key_csv
SPLIT_CACHE_FILE = CONFIG.split_cache_file
AGE_TARGET_CACHE_FILE = CONFIG.age_target_cache_file
SUBJECT_CODE_CACHE_FILE = CONFIG.subject_code_cache_file
SUBJECT_CODEBOOK_FILE = CONFIG.subject_codebook_file

BATCH_SIZE = CONFIG.batch_size
EPOCHS = CONFIG.epochs
LR = CONFIG.lr

USE_HUBER_LOSS = CONFIG.use_huber_loss
HUBER_BETA = CONFIG.huber_beta

NORMALIZE_TARGET = CONFIG.normalize_target
CLIP_PREDICTED_AGE_AFTER_DENORM = CONFIG.clip_predicted_age_after_denorm
MIN_PREDICTED_AGE_YEARS = CONFIG.min_predicted_age_years
NORMALIZE_INPUT = CONFIG.normalize_input
NORM_SAMPLE_WINDOWS = CONFIG.norm_sample_windows
NORM_EPS = CONFIG.norm_eps

SUBJECT_BALANCED_TRAINING = CONFIG.subject_balanced_training
MAX_WINDOWS_PER_SUBJECT_PER_EPOCH = CONFIG.max_windows_per_subject_per_epoch

USE_TORCH_COMPILE = CONFIG.use_torch_compile
USE_TF32 = CONFIG.use_tf32
TORCH_COMPILE_MODE = CONFIG.torch_compile_mode
TORCH_COMPILE_DYNAMIC = CONFIG.torch_compile_dynamic
METRIC_EVERY_N_EPOCHS = CONFIG.metric_every_n_epochs
PLOT_MAX_POINTS = CONFIG.plot_max_points
RANDOM_SEED = CONFIG.random_seed
SUBJECT_EXAMPLE_COUNT = CONFIG.subject_example_count
SUBJECT_EXAMPLE_MAX_WINDOWS = CONFIG.subject_example_max_windows

MIL_PSEUDO_BAG_SIZE = CONFIG.mil_pseudo_bag_size
MIL_PSEUDO_BAG_MIN_WINDOWS = CONFIG.mil_pseudo_bag_min_windows
MIL_PSEUDO_BAG_MAX_WINDOWS = CONFIG.mil_pseudo_bag_max_windows
MIL_ALLOW_REPLACEMENT_WHEN_SMALL = CONFIG.mil_allow_replacement_when_small
MIL_BAG_BATCH_SIZE = CONFIG.mil_bag_batch_size
MIL_FINETUNE_ENCODER_LR = CONFIG.mil_finetune_encoder_lr
MIL_FINETUNE_HEAD_LR = CONFIG.mil_finetune_head_lr
MIL_FINETUNE_WEIGHT_DECAY = CONFIG.mil_finetune_weight_decay
MIL_FINETUNE_ENCODER_WARMUP_EPOCHS = CONFIG.mil_finetune_encoder_warmup_epochs

LOG_LEVEL = CONFIG.log_level
DEBUG_CHUNK_LOG_EVERY = CONFIG.debug_chunk_log_every

BOOTSTRAP_ENABLED = CONFIG.bootstrap_enabled
BOOTSTRAP_ITERATIONS = CONFIG.bootstrap_iterations
BOOTSTRAP_CONFIDENCE = CONFIG.bootstrap_confidence

EARLY_STOPPING_ENABLED = CONFIG.early_stopping_enabled
EARLY_STOPPING_PATIENCE = CONFIG.early_stopping_patience
EARLY_STOPPING_MIN_EPOCHS = CONFIG.early_stopping_min_epochs
EARLY_STOPPING_MIN_DELTA_REL = CONFIG.early_stopping_min_delta_rel
EARLY_STOPPING_MIN_DELTA_ABS = CONFIG.early_stopping_min_delta_abs

VALIDATION_KEY_CSV = CONFIG.validation_key_csv
USE_AUTO_SPLIT = CONFIG.use_auto_split
SPLIT_RATIO_TRAIN = CONFIG.split_ratio_train
SPLIT_RATIO_VAL = CONFIG.split_ratio_val
SPLIT_RATIO_TEST = CONFIG.split_ratio_test
SUBJECT_KEY_FILENAME = CONFIG.subject_key_filename
USE_AGE_STRATIFIED_SPLIT = CONFIG.use_age_stratified_split
STRATIFY_AGE_BIN_YEARS = CONFIG.stratify_age_bin_years
STRATIFY_TAIL_LOW_MAX_AGE = CONFIG.stratify_tail_low_max_age
STRATIFY_TAIL_HIGH_MIN_AGE = CONFIG.stratify_tail_high_min_age
STRATIFY_MIN_SUBJECTS_PER_STRATUM = CONFIG.stratify_min_subjects_per_stratum
AGE_WEIGHTED_WINDOW_SAMPLING = CONFIG.age_weighted_window_sampling
CNN_SAMPLES_PER_EPOCH = CONFIG.cnn_samples_per_epoch
MIL_INVERSE_FREQUENCY_SUBJECT_SAMPLING = CONFIG.mil_inverse_frequency_subject_sampling
MIL_SUBJECT_DRAWS_PER_EPOCH = CONFIG.mil_subject_draws_per_epoch
SUBJECT_KEY_CSV = (
    os.path.join(CONFIG.input_dir, CONFIG.subject_key_filename)
    if CONFIG.subject_key_filename
    else None
)
LR_SCHEDULER = CONFIG.lr_scheduler
REDUCE_LR_PATIENCE = CONFIG.reduce_lr_patience
REDUCE_LR_FACTOR = CONFIG.reduce_lr_factor
MIN_LR = CONFIG.min_lr
GRAD_CLIP_NORM = CONFIG.grad_clip_norm
CNN_WEIGHT_DECAY = CONFIG.cnn_weight_decay
MIL_EARLY_STOPPING_ENABLED = CONFIG.mil_early_stopping_enabled
MIL_EARLY_STOPPING_PATIENCE = CONFIG.mil_early_stopping_patience
MIL_EARLY_STOPPING_MIN_EPOCHS = CONFIG.mil_early_stopping_min_epochs
MIL_EARLY_STOPPING_MIN_DELTA_ABS = CONFIG.mil_early_stopping_min_delta_abs
MIL_EARLY_STOPPING_MIN_DELTA_REL = CONFIG.mil_early_stopping_min_delta_rel
MIL_REGRESSOR_DROPOUT = CONFIG.mil_regressor_dropout

MODEL_SAVE_NAME = CONFIG.model_save_name
REPORT_SAVE_NAME = CONFIG.report_save_name
SUBJECT_REPORT_SAVE_NAME = CONFIG.subject_report_save_name
RUN_SUMMARY_TXT_NAME = CONFIG.run_summary_txt_name
RUN_SUMMARY_JSON_NAME = CONFIG.run_summary_json_name
RUN_SUMMARY_HISTORY_DIR = CONFIG.run_summary_history_dir
BEST_HPARAMS_FILE = CONFIG.best_hparams_file
TUNING_RESULTS_FILE = CONFIG.tuning_results_file

# Defaults folder and file paths (user-editable fallbacks)
DEFAULTS_DIR = os.path.join(PROJECT_DIR, CONFIG.defaults_dir_name)
DEFAULT_MODEL_PATH = os.path.join(DEFAULTS_DIR, CONFIG.default_model_file)
DEFAULT_HPARAMS_PATH = os.path.join(DEFAULTS_DIR, CONFIG.default_hparams_file)
