Input folder — required files and brief descriptions

Place your preprocessed window fastpack files and subject key CSVs here before running the pipeline.

Required / expected files (examples used by this project):

- `AgeTraining_Key.csv` — training subject key CSV with columns `SubjectID` and `VariableValue` (age).
- `AgeTesting_Key.csv` — testing subject key CSV with columns `SubjectID` and `VariableValue` (age).
- `X_T1281.fp16.npy` — windowed EEG signals array/memmap (fp16), shape `(n_windows, window_length)`.
- `y_T1281.int16.npy` — window-level event/target codes (int16) used for provenance and indexing.
- `meta_T1281.csv` — metadata CSV mapping each window row to source file/subject (must include `subject_id`).
- `idx_T1281.int32.npy` — original event row indices (optional provenance file).

How to obtain these files:
- These are produced by the event-extraction / fastpack workflow described in the top-level README. If you do not have them, run the upstream extraction and packing tools or request the packed files from the data owner.

Notes:
- This folder is intentionally ignored by git (to avoid committing sensitive or large data). Keep the data private and do not push raw subject-level files to public repositories.
- The pipeline requires the training/testing key CSVs to build per-window age targets and subject mappings; ensure they are present with correct column names.