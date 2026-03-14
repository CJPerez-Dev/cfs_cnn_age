"""Merge AgeTraining_Key.csv and AgeTesting_Key.csv into a single key file for auto-splitting.

Usage:
    python scripts/merge_key_files.py [--input-dir INPUT_DIR] [--output OUTPUT_CSV]

Defaults: input_dir = project input/, output = input_dir/AgeKey.csv
"""

import argparse
import csv
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Merge train and test key CSVs into one file for --auto-split."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing AgeTraining_Key.csv and AgeTesting_Key.csv (default: repo input/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <input-dir>/AgeKey.csv).",
    )
    args = parser.parse_args()

    if args.input_dir is None:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.input_dir = os.path.join(repo, "input")
    if args.output is None:
        args.output = os.path.join(args.input_dir, "AgeKey.csv")

    train_path = os.path.join(args.input_dir, "AgeTraining_Key.csv")
    test_path = os.path.join(args.input_dir, "AgeTesting_Key.csv")

    for p in (train_path, test_path):
        if not os.path.exists(p):
            print(f"Error: missing {p}", file=sys.stderr)
            sys.exit(1)

    subject_age = {}
    for path, label in [(train_path, "train"), (test_path, "test")]:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("SubjectID", "").strip()
                val = row.get("VariableValue", "").strip()
                if not sid or not val:
                    continue
                if sid in subject_age:
                    print(f"Warning: {sid} in both key files; keeping first value.", file=sys.stderr)
                    continue
                try:
                    subject_age[sid] = float(val)
                except ValueError:
                    continue

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SubjectID", "VariableValue"])
        for sid in sorted(subject_age.keys()):
            writer.writerow([sid, subject_age[sid]])

    print(f"Wrote {len(subject_age)} subjects to {args.output}")
    print("Enable auto-split: set config use_auto_split=True and subject_key_filename='AgeKey.csv', or run with --auto-split.")


if __name__ == "__main__":
    main()
