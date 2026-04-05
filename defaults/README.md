Repository defaults folder

This folder is an optional place to store repository-level fallback files that
are used when you do not provide explicit paths on the command line.

Supported default files (place these in the `defaults/` folder at the repo root):

- `default_model.pt`
  - A PyTorch checkpoint containing a trained CNN model's `state_dict` or
    a full checkpoint dict with a `state_dict` key. This file is used to
    initialize the MIL instance encoder when running `--model-mode mil` or
    `--model-mode both` (unless you supply `--mil-pretrained-model`).
  - Example path: `defaults/default_model.pt`

- `default_cnn_hyperparameters.json`
  - CNN training hyperparameters (learning rate, Huber loss, window caps,
    `cnn_embedding_dim`, `cnn_dropout`, `cnn_weight_decay`, etc.).

- `default_mil_hyperparameters.json`
  - MIL hyperparameters (`mil_*` keys: attention, pooling, bag size, MIL LRs, etc.).

- `default_hyperparameters.json` (legacy, optional)
  - A **single** JSON containing both CNN and MIL keys. Still supported:
    it is merged **before** the split files, so split files override it when
    the same keys appear in both.

**Merge order (later overrides earlier):** code defaults → `default_hyperparameters.json` (if present) → `default_cnn_hyperparameters.json` → `default_mil_hyperparameters.json`. Only keys that exist in the pipeline’s full hyperparameter schema are applied.

How the pipeline uses these files

- `--mil-pretrained-model PATH` overrides `defaults/default_model.pt`.
- `--hparams-file PATH` overrides values from `defaults/` for that run (merged on top of the merged repo defaults).
- If you run `--model-mode mil` (or `both`) and no checkpoint is provided,
  the pipeline will look for `defaults/default_model.pt` and will raise an error if it is not present.
- For `--model-mode both` without `--hparams-file`, at least one of the following must exist:
  repository defaults JSON (see above), or `output/hparams/best_hyperparameters.json`,
  or pass `--cnn-hparams-file` / `--mil-hparams-file`.

---

## How tuning affects these files

Tuning **does not** automatically edit `defaults/default_*_hyperparameters.json`. It writes **new** JSON under `output/hparams/` (e.g. `best_hyperparameters.json` or `best_hyperparameters_<tune_name>.json`). You copy or merge values into `defaults/` when you want them to become the new baseline.

### `--model-mode cnn --tune`

- Searches **CNN-focused** hyperparameters (LR, dropout, weight decay, embedding dim, etc.).
- Saves **one** JSON with the **full** merged dict (CNN + MIL keys from the trial’s base defaults), same as before.
- **To update repo defaults:** copy only the CNN-related keys into `defaults/default_cnn_hyperparameters.json` (and adjust MIL file only if something shared changed).

### `--model-mode mil --tune`

- Searches **MIL-focused** hyperparameters (attention, pooling, bag size, MIL LRs, etc.).
- Saves **one** JSON with the **full** merged dict.
- **To update repo defaults:** copy MIL-related keys into `defaults/default_mil_hyperparameters.json`.

### `--model-mode both --tune`

- Runs a **paired** CNN trial and MIL trial per Optuna/grid candidate (same candidate dict).
- Selects the best configuration by **combined** validation MAE (or test if no val split).
- Saves **one** JSON with the **full** merged dict covering both stages.
- **To update repo defaults:** split the saved keys into `default_cnn_hyperparameters.json` and `default_mil_hyperparameters.json` (or keep a single legacy `default_hyperparameters.json` if you prefer one file).

### Using split defaults with `--model-mode both`

You can point each stage at its own file:

- `--hparams-file PATH` — single file for **both** CNN and MIL (unchanged).
- `--cnn-hparams-file PATH` — overrides CNN keys for the CNN run only.
- `--mil-hparams-file PATH` — overrides MIL keys for the MIL run only.

If you only set `--cnn-hparams-file` and `--mil-hparams-file`, you do not need a combined `default_hyperparameters.json`.

If you prefer not to use repository defaults, always pass `--mil-pretrained-model`
and/or `--hparams-file` on the command line.
