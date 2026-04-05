#!/usr/bin/env python3
"""Generate presentation figures (CNN vs CNN+MIL) from run summary JSON files.

This is **not** run automatically by the training CLI; call it after a run when you
want slides/poster figures.

Examples:
  python scripts/plot_presentation_cnn_mil.py --demo --out-dir output/presentation_plots
  python scripts/plot_presentation_cnn_mil.py \\
      --cnn-json output/RUN_TS/cnn/cnn_run_summary_RUN_TS_cnn.json \\
      --mil-json output/RUN_TS/mil/cnn_run_summary_RUN_TS_mil.json \\
      --out-dir output/presentation_plots

Summaries should include ``train_maes`` and (for CNN) ``val_maes`` for side-by-side
MAE curves; older JSONs without ``val_maes`` still plot train MAE only on the CNN panel.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as script from repo root without install
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cnn_age_project.visualization.presentation_plots import (  # noqa: E402
    demo_summaries,
    load_run_summary,
    render_presentation_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Presentation plots: CNN vs CNN+MIL.")
    parser.add_argument("--cnn-json", type=str, default=None, help="Path to CNN run summary JSON.")
    parser.add_argument("--mil-json", type=str, default=None, help="Path to CNN+MIL run summary JSON.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for PNG figures.")
    parser.add_argument("--prefix", type=str, default="presentation", help="Output filename prefix.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic metrics (no JSON required) to preview figure layout.",
    )
    args = parser.parse_args()

    if args.demo:
        cnn_s, mil_s = demo_summaries()
    else:
        if not args.cnn_json or not args.mil_json:
            parser.error("--cnn-json and --mil-json are required unless --demo is set.")
        cnn_s = load_run_summary(args.cnn_json)
        mil_s = load_run_summary(args.mil_json)

    paths = render_presentation_bundle(cnn_s, mil_s, args.out_dir, prefix=args.prefix)
    print("Wrote:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
