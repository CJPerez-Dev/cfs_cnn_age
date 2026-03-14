"""Data loading and cache-building utilities for memmap-based training."""

import csv
import json
import logging
import os
import numpy as np

from cnn_age_project.config import IDX_FILE, META_FILE, X_FILE, Y_FILE

logger = logging.getLogger(__name__)


def load_subject_age_map(csv_path):
    """Load a SubjectID -> age mapping from a key CSV file.

    Args:
        csv_path (str): Path to key CSV with ``SubjectID`` and ``VariableValue`` columns.

    Returns:
        dict[str, float]: Mapping of subject ID to age value.
    """
    subject_age = {}
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            subject_id = row.get("SubjectID", "").strip()
            if not subject_id:
                continue
            value = row.get("VariableValue", "").strip()
            if not value:
                continue
            subject_age[subject_id] = float(value)
    return subject_age


def validate_subject_wise_split_integrity(
    train_indices,
    test_indices,
    subject_codes,
    subject_codebook,
):
    """Assert train/test subject separation for generalization-safe evaluation.

    Args:
        train_indices (np.ndarray): Window indices in train split.
        test_indices (np.ndarray): Window indices in test split.
        subject_codes (np.ndarray | np.memmap): Subject code per window index.
        subject_codebook (list[str]): Code-to-subject-id mapping.

    Returns:
        tuple[set[int], set[int]]: ``(train_subject_codes, test_subject_codes)``.
    """
    train_subject_codes = set(np.asarray(subject_codes[train_indices], dtype=np.int32).tolist())
    test_subject_codes = set(np.asarray(subject_codes[test_indices], dtype=np.int32).tolist())

    train_subject_codes.discard(-1)
    test_subject_codes.discard(-1)

    overlap_codes = train_subject_codes.intersection(test_subject_codes)
    if overlap_codes:
        overlap_subjects = [subject_codebook[code] for code in sorted(overlap_codes) if 0 <= code < len(subject_codebook)]
        preview = overlap_subjects[:10]
        raise ValueError(
            "Subject-wise split integrity failed: train/test leakage detected for "
            f"{len(overlap_codes)} subjects. Example subject IDs: {preview}"
        )

    logger.info(
        "Subject-wise split integrity passed | train_subjects=%d | test_subjects=%d | overlap=0",
        len(train_subject_codes),
        len(test_subject_codes),
    )
    return train_subject_codes, test_subject_codes


def _missing_files(folder_path, required_names):
    """Return names from required_names that are missing in folder_path.

    Args:
        folder_path (str): Directory to inspect.
        required_names (list[str]): Required filenames.

    Returns:
        list[str]: Missing filenames.
    """
    return [name for name in required_names if not os.path.exists(os.path.join(folder_path, name))]


def validate_required_project_files(input_dir, output_dir):
    """Validate required input files and output write access before execution.

    Args:
        input_dir (str): Directory that should contain input CSV/memmap files.
        output_dir (str): Directory used to write caches and run artifacts.

    Returns:
        str: Resolved memmap root directory.
    """
    required_input_files = ["AgeTraining_Key.csv", "AgeTesting_Key.csv"]
    required_memmap_files = [X_FILE, Y_FILE, META_FILE]

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    missing_input = _missing_files(input_dir, required_input_files)
    if missing_input:
        details = "\n".join([f"  - {os.path.join(input_dir, name)}" for name in missing_input])
        raise FileNotFoundError(
            "Missing required files in input folder:\n"
            f"{details}\n"
            "Place the missing key CSV files into the input folder and rerun."
        )

    missing_memmap_input = _missing_files(input_dir, required_memmap_files)
    if missing_memmap_input:
        input_missing_details = "\n".join([f"  - {os.path.join(input_dir, name)}" for name in missing_memmap_input])
        raise FileNotFoundError(
            "Required memmap files are missing from input folder.\n"
            "Missing in input folder:\n"
            f"{input_missing_details}\n"
            "Place these memmap files into the input folder and rerun."
        )

    memmap_root = input_dir
    logger.info("All required memmap files found in input folder.")

    optional_idx_path = os.path.join(memmap_root, IDX_FILE)
    if not os.path.exists(optional_idx_path):
        logger.warning("Optional file missing (subject/event diagnostics may be reduced): %s", optional_idx_path)

    test_write_path = os.path.join(output_dir, ".write_test.tmp")
    try:
        with open(test_write_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_write_path)
    except Exception as exc:
        raise PermissionError(f"Output folder is not writable: {output_dir} | {exc}") from exc

    logger.info("Preflight file check passed.")
    return memmap_root


def build_or_load_targets_and_split(
    meta_path,
    n_samples,
    train_age_map,
    test_age_map,
    split_cache_path,
    age_cache_path,
    subject_code_cache_path,
    subject_codebook_path,
):
    """Create or load cached train/test split, age targets, and subject-code arrays.

    Args:
        meta_path (str): Path to metadata CSV with subject IDs per row.
        n_samples (int): Number of rows/windows in source memmaps.
        train_age_map (dict[str, float]): Training subject-age lookup.
        test_age_map (dict[str, float]): Testing subject-age lookup.
        split_cache_path (str): Path for cached split-code memmap.
        age_cache_path (str): Path for cached age-target memmap.
        subject_code_cache_path (str): Path for cached subject-code memmap.
        subject_codebook_path (str): Path for JSON list mapping code->subject ID.

    Returns:
        tuple[np.memmap, np.memmap, np.memmap, list[str]]: Split codes, age targets,
        subject codes, and subject codebook.
    """
    caches_exist = (
        os.path.exists(split_cache_path)
        and os.path.exists(age_cache_path)
        and os.path.exists(subject_code_cache_path)
        and os.path.exists(subject_codebook_path)
    )

    if caches_exist:
        split_codes = np.memmap(split_cache_path, dtype=np.uint8, mode="r")
        age_targets = np.memmap(age_cache_path, dtype=np.float32, mode="r")
        subject_codes = np.memmap(subject_code_cache_path, dtype=np.int32, mode="r")
        if len(split_codes) == n_samples and len(age_targets) == n_samples and len(subject_codes) == n_samples:
            with open(subject_codebook_path, "r", encoding="utf-8") as f:
                subject_codebook = json.load(f)
            logger.info("Loaded cached split/age/subject-code arrays.")
            return split_codes, age_targets, subject_codes, subject_codebook
        logger.warning("Cache length mismatch. Rebuilding split/age/subject caches.")

    split_codes = np.memmap(split_cache_path, dtype=np.uint8, mode="w+", shape=(n_samples,))
    age_targets = np.memmap(age_cache_path, dtype=np.float32, mode="w+", shape=(n_samples,))
    subject_codes = np.memmap(subject_code_cache_path, dtype=np.int32, mode="w+", shape=(n_samples,))

    split_codes[:] = 0
    age_targets[:] = np.nan
    subject_codes[:] = -1

    subject_to_code = {}
    codebook = []
    seen_train = 0
    seen_test = 0

    with open(meta_path, newline="", encoding="utf-8") as meta_file:
        reader = csv.DictReader(meta_file)
        for row_idx, row in enumerate(reader):
            if row_idx >= n_samples:
                break

            subject_id = row.get("subject_id", "").strip()
            if not subject_id:
                continue

            code = subject_to_code.get(subject_id)
            if code is None:
                code = len(codebook)
                subject_to_code[subject_id] = code
                codebook.append(subject_id)
            subject_codes[row_idx] = code

            if subject_id in train_age_map:
                split_codes[row_idx] = 1
                age_targets[row_idx] = np.float32(train_age_map[subject_id])
                seen_train += 1
            elif subject_id in test_age_map:
                split_codes[row_idx] = 2
                age_targets[row_idx] = np.float32(test_age_map[subject_id])
                seen_test += 1

    split_codes.flush()
    age_targets.flush()
    subject_codes.flush()
    with open(subject_codebook_path, "w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2)

    logger.info(
        "Built caches | train windows=%d test windows=%d unmatched=%d unique subjects=%d",
        seen_train,
        seen_test,
        int(n_samples - seen_train - seen_test),
        len(codebook),
    )
    return (
        np.memmap(split_cache_path, dtype=np.uint8, mode="r"),
        np.memmap(age_cache_path, dtype=np.float32, mode="r"),
        np.memmap(subject_code_cache_path, dtype=np.int32, mode="r"),
        codebook,
    )


def load_memmap_arrays(memmap_root):
    """Open source memmaps and infer sample/window dimensions.

    Args:
        memmap_root (str): Directory containing source memmap files.

    Returns:
        dict[str, Any]: Dictionary containing loaded memmaps and inferred shape metadata.
    """
    x_path = os.path.join(memmap_root, X_FILE)
    y_event_path = os.path.join(memmap_root, Y_FILE)
    meta_path = os.path.join(memmap_root, META_FILE)

    if not os.path.exists(y_event_path):
        raise FileNotFoundError(f"Missing source y memmap file: {y_event_path}")

    y_source = np.memmap(y_event_path, dtype=np.int16, mode="r")
    n_samples = len(y_source)

    x_mem = np.memmap(x_path, dtype=np.float16, mode="c")
    total_values = len(x_mem)
    window_len = total_values // n_samples
    x_mem = x_mem.reshape(n_samples, window_len)

    return {
        "x_mem": x_mem,
        "y_source": y_source,
        "n_samples": n_samples,
        "window_len": window_len,
        "meta_path": meta_path,
        "x_path": x_path,
        "y_event_path": y_event_path,
    }
