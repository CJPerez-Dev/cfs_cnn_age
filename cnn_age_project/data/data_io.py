"""Data loading and cache-building utilities for memmap-based training."""

import csv
import json
import logging
import os
import numpy as np

from cnn_age_project.config import IDX_FILE, META_FILE, USE_AUTO_SPLIT, X_FILE, Y_FILE

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
    val_indices=None,
):
    """Assert train/test (and optional validation) subject separation.

    Args:
        train_indices (np.ndarray): Window indices in train split.
        test_indices (np.ndarray): Window indices in test split.
        subject_codes (np.ndarray | np.memmap): Subject code per window index.
        subject_codebook (list[str]): Code-to-subject-id mapping.
        val_indices (np.ndarray | None): Optional validation window indices.

    Returns:
        tuple: (train_subject_codes, test_subject_codes) or with val_subject_codes if val_indices provided.
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

    if val_indices is not None and val_indices.size > 0:
        val_subject_codes = set(np.asarray(subject_codes[val_indices], dtype=np.int32).tolist())
        val_subject_codes.discard(-1)
        for name, other in [("train", train_subject_codes), ("test", test_subject_codes)]:
            overlap = val_subject_codes.intersection(other)
            if overlap:
                overlap_subjects = [subject_codebook[c] for c in sorted(overlap) if 0 <= c < len(subject_codebook)]
                raise ValueError(
                    f"Subject-wise split integrity failed: validation/{name} leakage for "
                    f"{len(overlap)} subjects. Example IDs: {overlap_subjects[:10]}"
                )
        logger.info(
            "Subject-wise split integrity passed | train_subjects=%d | test_subjects=%d | val_subjects=%d | overlap=0",
            len(train_subject_codes),
            len(test_subject_codes),
            len(val_subject_codes),
        )
        return train_subject_codes, test_subject_codes, val_subject_codes

    logger.info(
        "Subject-wise split integrity passed | train_subjects=%d | test_subjects=%d | overlap=0",
        len(train_subject_codes),
        len(test_subject_codes),
    )
    return train_subject_codes, test_subject_codes


def build_subject_age_map_from_metadata(meta_path, n_samples, age_columns=("age", "Age", "VariableValue")):
    """Build subject_id -> age from metadata CSV if an age column exists.

    Uses the first row per subject. Returns None if no recognized age column is present.

    Args:
        meta_path (str): Path to metadata CSV (e.g. meta_T1281.csv).
        n_samples (int): Max rows to read (should match memmap length).
        age_columns (tuple[str, ...]): Column names to try for age (first match wins).

    Returns:
        dict[str, float] | None: Subject -> age, or None if no age column found.
    """
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        fieldnames_lower = {k.strip().lower(): k for k in reader.fieldnames}
        age_col = None
        for cand in age_columns:
            key = cand.strip().lower()
            if key in fieldnames_lower:
                age_col = fieldnames_lower[key]
                break
        if age_col is None:
            return None
        subject_col = fieldnames_lower.get("subject_id")
        if subject_col is None:
            return None
        subject_age = {}
        for row_idx, row in enumerate(reader):
            if row_idx >= n_samples:
                break
            subject_id = row.get(subject_col, "").strip()
            if not subject_id:
                continue
            val = row.get(age_col, "").strip()
            if not val:
                continue
            if subject_id not in subject_age:
                try:
                    subject_age[subject_id] = float(val)
                except ValueError:
                    pass
        return subject_age if subject_age else None


def split_subjects_ratio(subject_ids, train_ratio, val_ratio, test_ratio, rng):
    """Split subject IDs into train/val/test by ratio (subject-level, no overlap).

    Args:
        subject_ids (list[str]): All subject IDs.
        train_ratio (float): Fraction for train (e.g. 0.7).
        val_ratio (float): Fraction for val (e.g. 0.15).
        test_ratio (float): Fraction for test (e.g. 0.15).
        rng (np.random.Generator): Random generator for reproducibility.

    Returns:
        tuple[list[str], list[str], list[str]]: (train_ids, val_ids, test_ids).
    """
    ids = list(subject_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return [], [], []
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(0, int(round(n * val_ratio)))
    n_test = max(1, n - n_train - n_val)
    if n_test <= 0:
        n_test = 1
        n_val = max(0, n - n_train - n_test)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]
    return train_ids, val_ids, test_ids


def _missing_files(folder_path, required_names):
    """Return names from required_names that are missing in folder_path.

    Args:
        folder_path (str): Directory to inspect.
        required_names (list[str]): Required filenames.

    Returns:
        list[str]: Missing filenames.
    """
    return [name for name in required_names if not os.path.exists(os.path.join(folder_path, name))]


def validate_required_project_files(input_dir, output_dir, use_auto_split=None):
    """Validate required input files and output write access before execution.

    Args:
        input_dir (str): Directory that should contain input CSV/memmap files.
        output_dir (str): Directory used to write caches and run artifacts.
        use_auto_split (bool | None): If True, key CSVs are not required. If None, uses USE_AUTO_SPLIT from config.

    Returns:
        str: Resolved memmap root directory.
    """
    if use_auto_split is None:
        use_auto_split = USE_AUTO_SPLIT
    required_input_files = [] if use_auto_split else ["AgeTraining_Key.csv", "AgeTesting_Key.csv"]
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
    val_age_map=None,
):
    """Create or load cached train/validation/test split, age targets, and subject-code arrays.

    Args:
        meta_path (str): Path to metadata CSV with subject IDs per row.
        n_samples (int): Number of rows/windows in source memmaps.
        train_age_map (dict[str, float]): Training subject-age lookup.
        test_age_map (dict[str, float]): Testing subject-age lookup.
        split_cache_path (str): Path for cached split-code memmap.
        age_cache_path (str): Path for cached age-target memmap.
        subject_code_cache_path (str): Path for cached subject-code memmap.
        subject_codebook_path (str): Path for JSON list mapping code->subject ID.
        val_age_map (dict[str, float] | None): Optional validation subject-age lookup.
            If provided, subjects in this map get split_code 3.

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
            n_val_cached = int(np.sum(split_codes == 3))
            want_val = val_age_map is not None and len(val_age_map) > 0
            if (want_val and n_val_cached == 0) or (not want_val and n_val_cached > 0):
                logger.warning(
                    "Cache validation split mismatch (want_val=%s, cached_val_windows=%d). Rebuilding.",
                    want_val,
                    n_val_cached,
                )
            else:
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
    seen_val = 0

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
            elif val_age_map is not None and subject_id in val_age_map:
                split_codes[row_idx] = 3
                age_targets[row_idx] = np.float32(val_age_map[subject_id])
                seen_val += 1

    split_codes.flush()
    age_targets.flush()
    subject_codes.flush()
    with open(subject_codebook_path, "w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2)

    logger.info(
        "Built caches | train windows=%d test windows=%d val windows=%d unmatched=%d unique subjects=%d",
        seen_train,
        seen_test,
        seen_val,
        int(n_samples - seen_train - seen_test - seen_val),
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
