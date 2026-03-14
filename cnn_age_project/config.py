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
    epochs: int = 75
    lr: float = 3e-4

    use_huber_loss: bool = True
    huber_beta: float = 1.0

    normalize_target: bool = True
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
    mil_allow_replacement_when_small: bool = True
    mil_bag_batch_size: int = 16
    mil_finetune_encoder_lr: float = 1e-5
    mil_finetune_head_lr: float = 5e-5
    mil_finetune_weight_decay: float = 1e-2

    log_level: str = "INFO"
    debug_chunk_log_every: int = 10

    bootstrap_enabled: bool = True
    bootstrap_iterations: int = 1000
    bootstrap_confidence: float = 0.95

    early_stopping_enabled: bool = True
    early_stopping_patience: int = 2
    early_stopping_min_epochs: int = 3
    early_stopping_min_delta_rel: float = 1e-3
    early_stopping_min_delta_abs: float = 1e-4

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
