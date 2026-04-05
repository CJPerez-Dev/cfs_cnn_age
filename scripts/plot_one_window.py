#!/usr/bin/env python3
"""Plot one EEG window from input memmap and save as PNG.

Usage:
    python scripts/plot_one_window.py
    python scripts/plot_one_window.py --index 12345
    python scripts/plot_one_window.py --random
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Repository root (parent of ``scripts/``)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cnn_age_project.config import INPUT_DIR  # noqa: E402
from cnn_age_project.data.data_io import load_memmap_arrays  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Save one EEG window plot as PNG.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Window index to plot (default: 0).",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random window index (overrides --index).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --random (default: 42).",
    )
    args = parser.parse_args()

    arrays = load_memmap_arrays(INPUT_DIR)
    x_mem = arrays["x_mem"]
    x_path = arrays["x_path"]
    n = int(x_mem.shape[0])
    if n <= 0:
        raise RuntimeError(f"No windows found in {x_path}")

    if bool(args.random):
        rng = np.random.default_rng(int(args.seed))
        idx = int(rng.integers(0, n))
    else:
        idx = int(args.index)
    if idx < 0 or idx >= n:
        raise IndexError(f"index out of range: {idx} (valid: 0..{n - 1})")

    window = np.asarray(x_mem[idx], dtype=np.float32)
    t = np.arange(window.shape[0], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, window, linewidth=1.0, color="#1f77b4")
    ax.set_title(f"EEG window index {idx}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    fig.tight_layout()

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"one_window_idx{idx}.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Window shape: {window.shape} | dtype: {window.dtype} | source: {x_path}")


if __name__ == "__main__":
    main()

