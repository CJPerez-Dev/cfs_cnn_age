Output folder — generated artifacts and caches

This directory holds training/evaluation outputs produced by the pipeline. It is intentionally excluded from version control (large binaries and models). The repository does track the small `output/hparams/` artifacts (tuned hyperparameter JSON files) and README/placeholders so the layout is visible in clones.

Typical contents (per-run timestamped subfolder `<YYYY-MM-DD_HH-MM-SS>`):

- `cnn_age_model_<timestamp>.pt` — saved model state_dict (do not commit large weights).
- `cnn_training_report_<timestamp>.png` — training/evaluation figures (loss, R², MAE plots).
- `cnn_subject_examples_<timestamp>.png` — sampled per-subject prediction examples.
- `cnn_run_summary_<timestamp>.txt` / `.json` — run metadata and numeric summary.

Top-level caches and helper files (may be present):

- `output/hparams/` — tracked small hyperparameter JSON files (committed by repository to reproduce runs).
- `split_codes_T1281.uint8.npy`, `y_age_T1281.float32.npy`, `subject_codes_T1281.int32.npy`, `subject_codebook_T1281.json` — cached split/age/subject mapping artifacts (usually large; ignored by git).
- `run_summaries/` — optional aggregated summaries (placeholder kept in repo).

Notes:
- Do not commit trained model weights or large .npy files to the public repo; use a controlled data storage solution or Git LFS if you need to version binaries.
