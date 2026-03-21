"""Age strata for stratified subject splits and inverse-frequency window sampling.

Uses 5-year bands in a middle range, with configurable collapsed tails (one stratum
below ``tail_low_max_age`` and one at/above ``tail_high_min_age``). Sparse strata
(``min_subjects_per_stratum``) are merged with an adjacent band (by age order)
before splitting so train/val/test can all receive subjects from each stratum.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def count_middle_bins(tail_low_max_age: float, tail_high_min_age: float, bin_years: float) -> int:
    """Number of interior 5-year-style bins between tail boundaries."""
    span = float(tail_high_min_age) - float(tail_low_max_age)
    if span <= 0:
        return 1
    return max(1, int(np.ceil(span / float(bin_years))))


def age_to_stratum_id(
    age: float,
    tail_low_max_age: float,
    tail_high_min_age: float,
    bin_years: float,
) -> int:
    """Map age to an ordered stratum index (young tail, middle bins, old tail).

    Stratum 0: age < tail_low_max_age
    Strata 1..K: [tail_low + k*bin, tail_low + (k+1)*bin) capped below tail_high
    Last stratum: age >= tail_high_min_age
    """
    L = float(tail_low_max_age)
    H = float(tail_high_min_age)
    W = float(bin_years)
    if age < L:
        return 0
    if age >= H:
        n_mid = count_middle_bins(L, H, W)
        return 1 + n_mid
    n_mid = count_middle_bins(L, H, W)
    b = int((float(age) - L) // W)
    b = min(b, n_mid - 1)
    return 1 + b


def n_strata(tail_low_max_age: float, tail_high_min_age: float, bin_years: float) -> int:
    """Total number of strata for the given band configuration."""
    return 2 + count_middle_bins(tail_low_max_age, tail_high_min_age, bin_years)


def assign_initial_subject_strata(
    subject_age: dict[str, float],
    tail_low_max_age: float,
    tail_high_min_age: float,
    bin_years: float,
) -> dict[str, int]:
    """Assign each subject to a raw (pre-merge) stratum id."""
    out = {}
    for sid, age in subject_age.items():
        out[sid] = age_to_stratum_id(float(age), tail_low_max_age, tail_high_min_age, bin_years)
    return out


def merge_sparse_strata(
    subject_to_stratum: dict[str, int],
    min_subjects_per_stratum: int,
) -> dict[str, int]:
    """Merge adjacent strata (by id order) until each has at least ``min`` subjects.

    Stratum ids are contiguous 0..S-1 with younger ages at lower ids. Merges the
    smallest stratum with its right neighbor when possible, otherwise the left.
    Relabels to contiguous ids after each merge.
    """
    if min_subjects_per_stratum <= 1:
        return dict(subject_to_stratum)

    current = dict(subject_to_stratum)
    while True:
        inv: dict[int, list[str]] = defaultdict(list)
        for sid, st in current.items():
            inv[st].append(sid)
        sizes = {st: len(lst) for st, lst in inv.items() if lst}
        if not sizes:
            return current
        active = sorted(sizes.keys())
        small = [st for st in active if sizes[st] < min_subjects_per_stratum]
        if not small or len(active) <= 1:
            break

        st_merge = int(small[0])
        idx = active.index(st_merge)
        if idx + 1 < len(active):
            partner = active[idx + 1]
        elif idx > 0:
            partner = active[idx - 1]
        else:
            break

        merged_label = min(st_merge, partner)
        for sid in inv[st_merge] + inv[partner]:
            current[sid] = merged_label
        unique_sorted = sorted(set(current.values()))
        relabel = {old: i for i, old in enumerate(unique_sorted)}
        current = {sid: relabel[st] for sid, st in current.items()}

    return current


def split_subjects_stratified_train_val_test(
    subject_age: dict[str, float],
    subject_to_stratum: dict[str, int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], list[str]]:
    """Stratified 3-way subject split: within each stratum, shuffle and allocate by ratio.

    Uses largest-remainder integer allocation per stratum. Very small strata may
    deviate from exact global ratios.
    """
    r_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(r_sum - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1, got {r_sum}")

    subject_ids = list(subject_age.keys())
    strata = sorted(set(subject_to_stratum[s] for s in subject_ids))

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for st in strata:
        pool = [s for s in subject_ids if subject_to_stratum[s] == st]
        rng.shuffle(pool)
        n = len(pool)
        nt, nv, nte = _allocate_three_way_counts(n, train_ratio, val_ratio, test_ratio)
        train_ids.extend(pool[:nt])
        val_ids.extend(pool[nt : nt + nv])
        test_ids.extend(pool[nt + nv : nt + nv + nte])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    return train_ids, val_ids, test_ids


def _largest_remainder_allocation(n: int, weights: list[float]) -> list[int]:
    """Allocate ``n`` integer slots across bins with given positive weights (sum 1)."""
    exact = [n * float(w) for w in weights]
    floors = [int(np.floor(e)) for e in exact]
    rem = n - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: exact[i] - floors[i], reverse=True)
    for k in range(max(0, rem)):
        floors[order[k % len(order)]] += 1
    return floors


def _allocate_three_way_counts(n: int, rt: float, rv: float, rte: float) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) nonnegative integers summing to ``n``."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    nt, nv, nte = _largest_remainder_allocation(n, [rt, rv, rte])
    return nt, nv, nte


def build_merged_strata_for_subjects(
    subject_age: dict[str, float],
    tail_low_max_age: float,
    tail_high_min_age: float,
    bin_years: float,
    min_subjects_per_stratum: int,
) -> dict[str, int]:
    """Raw strata from age bands, then merge sparse strata for stable 3-way splits."""
    raw = assign_initial_subject_strata(subject_age, tail_low_max_age, tail_high_min_age, bin_years)
    merged = merge_sparse_strata(raw, min_subjects_per_stratum)
    return merged


def log_stratum_summary(subject_age: dict[str, float], subject_to_stratum: dict[str, int]) -> None:
    """Emit stratum sizes (subject counts) for debugging."""
    inv: dict[int, list[str]] = defaultdict(list)
    for sid, st in subject_to_stratum.items():
        inv[st].append(sid)
    parts = []
    for st in sorted(inv.keys()):
        ages = [float(subject_age[s]) for s in inv[st]]
        parts.append(
            f"st{st}:n={len(inv[st])} age[min/mean/max]={min(ages):.1f}/{float(np.mean(ages)):.1f}/{max(ages):.1f}"
        )
    logger.info("Age strata (subjects) | %s", " | ".join(parts))


def subject_stratum_from_merged_map(
    subject_id: str,
    merged_stratum: dict[str, int] | None,
    subject_age: dict[str, float],
    tail_low_max_age: float,
    tail_high_min_age: float,
    bin_years: float,
) -> int:
    """Stratum id for a subject: merged map if present, else initial band id."""
    if merged_stratum is not None and subject_id in merged_stratum:
        return int(merged_stratum[subject_id])
    age = float(subject_age[subject_id])
    return age_to_stratum_id(age, tail_low_max_age, tail_high_min_age, bin_years)


def build_mil_train_subject_inverse_frequency_probs(
    balanced_counts: np.ndarray,
    subject_codebook: list[str],
    train_age_map: dict[str, float],
    subject_stratum_merged: dict[str, int] | None,
    tail_low_max_age: float,
    tail_high_min_age: float,
    bin_years: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-train-subject sampling probabilities ∝ 1 / n_subjects_in_same_age_stratum.

    Each age stratum gets equal total probability mass; within a stratum, subjects are uniform.
    Only subjects with at least one training window and an entry in ``train_age_map`` are included.

    Returns:
        tuple: ``(eligible_subject_codes, probs)`` aligned arrays for ``np.random.Generator.choice``.
    """
    eligible: list[int] = []
    stratum_per_row: list[int] = []
    for code, n_win in enumerate(balanced_counts):
        if int(n_win) <= 0:
            continue
        if code < 0 or code >= len(subject_codebook):
            continue
        sid = subject_codebook[code]
        if sid not in train_age_map:
            continue
        st = subject_stratum_from_merged_map(
            sid,
            subject_stratum_merged,
            train_age_map,
            tail_low_max_age,
            tail_high_min_age,
            bin_years,
        )
        eligible.append(int(code))
        stratum_per_row.append(int(st))

    if not eligible:
        raise ValueError("build_mil_train_subject_inverse_frequency_probs: no eligible train subjects.")

    counts_by_stratum: dict[int, int] = {}
    for st in stratum_per_row:
        counts_by_stratum[st] = counts_by_stratum.get(st, 0) + 1

    raw = np.array([1.0 / counts_by_stratum[st] for st in stratum_per_row], dtype=np.float64)
    probs = raw / raw.sum()
    return np.asarray(eligible, dtype=np.int64), probs


def build_subject_code_stratum_lookup(
    subject_codebook: list[str],
    train_age_map: dict[str, float],
    subject_stratum_merged: dict[str, int] | None,
    tail_low_max_age: float,
    tail_high_min_age: float,
    bin_years: float,
) -> np.ndarray:
    """``lookup[code]`` = age stratum for that subject code (0 if unknown / not in train)."""
    n = len(subject_codebook)
    lookup = np.zeros(n, dtype=np.int64)
    for code, sid in enumerate(subject_codebook):
        if sid not in train_age_map:
            continue
        lookup[code] = subject_stratum_from_merged_map(
            sid,
            subject_stratum_merged,
            train_age_map,
            tail_low_max_age,
            tail_high_min_age,
            bin_years,
        )
    return lookup
