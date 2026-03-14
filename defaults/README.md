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

- `default_hyperparameters.json`
  - A JSON file containing hyperparameter values used when you do not pass
    `--hparams-file` on the CLI and you want repository-level defaults.
  - Example minimal contents:

```json
{
  "learning_rate": 0.0003,
  "use_huber_loss": true,
  "huber_beta": 1.0,
  "max_windows_per_subject_per_epoch": 2000,
  "mil_attention_dim": 128,
  "mil_attention_dropout": 0.1,
  "mil_pooling_type": "gated",
  "mil_bag_size": 256,
  "mil_sampling_strategy": "random",
  "mil_encoder_lr": 0.00001,
  "mil_head_lr": 0.00005,
  "mil_weight_decay": 0.01,
  "mil_regressor_hidden_dim": 64,
  "mil_allow_replacement_when_small": true
}
```

How the pipeline uses these files

- `--mil-pretrained-model PATH` overrides `defaults/default_model.pt`.
- `--hparams-file PATH` overrides `defaults/default_hyperparameters.json`.
- If you run `--model-mode mil` (or `both`) and no checkpoint is provided,
  the pipeline will look for `defaults/default_model.pt` and will raise an error if it is not present.
- If you run `--model-mode both` and do not supply `--hparams-file`, the
  pipeline will look for `defaults/default_hyperparameters.json` and will
  raise an error if it is not present (unless `--tune` is enabled).

Tuning behavior

- `--model-mode cnn --tune`: tunes CNN-focused parameters (LR, dropout, weight decay, embedding dim, etc.).
- `--model-mode mil --tune`: tunes MIL-focused parameters (attention, pooling, bag size/sampling, and MIL optimizer rates).
- `--model-mode both --tune`: runs paired CNN and MIL trial evaluations. Best configuration is chosen by **validation MAE** when a validation set exists (e.g. with auto-split), otherwise by test MAE.

If you prefer not to use repository defaults, always pass `--mil-pretrained-model`
and/or `--hparams-file` on the command line.
