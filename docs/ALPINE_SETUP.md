# Running `cfs_cnn_age` on CURC Alpine (step-by-step)

Use this path when you want **GPU tuning** on Alpine. Your **Windows `.venv` with `cu130` is not copied to the cluster** — build a **Linux** environment there and match **`cuda/12.1.1`**.

---

## 1. Get access and load Slurm

1. SSH to CURC (or use Open OnDemand Terminal).
2. On a **login node**:
   ```bash
   module load slurm/alpine
   ```
3. Confirm you can submit:
   ```bash
   squeue -u $USER
   ```

---

## 2. Put the repo and data on the cluster

1. **Clone or copy** this repository to a persistent path, e.g.:
   - `/projects/$USER/cfs_cnn_age`, or
   - PetaLibrary / RC-recommended project storage.

2. **Copy your `input/`** tree (memmaps, `meta_*.csv`, keys, etc.) into that clone so it matches what you use locally. The pipeline expects data under **`input/`** relative to the repo root (see README **Prepare Input Files**).

3. **Do not rely on `output/` from your laptop** for tuning on Alpine unless you also copy it; fresh runs will recreate caches under `output/` as needed.

---

## 3. Create a Linux Python venv (CUDA 12.1, not cu130)

On a **login node** (or a short interactive GPU session if you prefer):

```bash
cd /projects/$USER/cfs_cnn_age   # your path

module load gcc/14.2.0
module load cuda/12.1.1

# Python 3.10+ recommended (use `module avail python` if you need a specific python module)
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# PyTorch built for CUDA 12.1 (matches cuda/12.1.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Rest of project deps (see repo root)
pip install -r requirements-gpu-alpine.txt
```

**Sanity check (on a GPU node or interactive GPU job):**

```bash
module load gcc/14.2.0
module load cuda/12.1.1
source /projects/$USER/cfs_cnn_age/.venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You want `True` on a GPU node.

**AMD MI100 (`ami100`):** use `module load rocm/6.1.0` instead of CUDA, install the **ROCm** PyTorch build from [PyTorch ROCm install instructions](https://pytorch.org/get-started/locally/) — not the `cu121` line above.

---

## 4. Point your Slurm scripts at the venv and repo

Edit **`slurm_cnn_age.sh`** and **`slurm_cnn_age_array.sh`** (or your copies):

1. Set **`REPO_ROOT`** to your real repo path (or export it before `sbatch`).
2. Uncomment and set **`source .../bin/activate`** to that **`.venv`**.
3. Uncomment **`#SBATCH --account=...`** if your allocation requires it.
4. Optional: mail directives.

Keep **`module load gcc/14.2.0`** and **`module load cuda/12.1.1`** in the script body (already in the templates).

---

## 5. Single-node test (interactive, optional)

Quick check before a long batch job:

```bash
module load slurm/alpine
sinteractive --partition=aa100 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --qos=normal
```

Then on the compute node:

```bash
module load gcc/14.2.0
module load cuda/12.1.1
cd /projects/$USER/cfs_cnn_age
source .venv/bin/activate
python -m cnn_age_project.main --tune --model-mode cnn --tune-backend optuna \
  --tune-max-trials 1 --tune-epochs 1 --tune-name alpine_smoke
```

If that finishes and writes under `output/hparams/`, data + GPU + deps are OK.

---

## 6. Start tuning (batch)

### One job (single GPU, one study in memory)

```bash
module load slurm/alpine
cd /projects/$USER/cfs_cnn_age
sbatch slurm_cnn_age.sh
```

Edit the `python -m cnn_age_project.main ...` block in **`slurm_cnn_age.sh`** for your experiment (`--tune`, `--tune-max-trials`, etc.).

### Several GPUs / array — **one shared Optuna study**

Use **`slurm_cnn_age_array.sh`**: every task must share the same **`--optuna-storage`** path (e.g. under **`/scratch/alpine/$USER/...`**) and the same **`--optuna-study-name`** / **`--tune-name`**. Use **`--tune` only** on workers (no **`--tune-and-train`** on every task). After all array tasks finish, run **one** training job with **`--hparams-file`** pointing at the saved JSON.

```bash
module load slurm/alpine
sbatch slurm_cnn_age_array.sh
```

---

## 7. After tuning

- Best hyperparameters: **`output/hparams/best_hyperparameters_<tune_name>.json`**
- Full training (example):
  ```bash
  python -m cnn_age_project.main --model-mode cnn \
    --hparams-file output/hparams/best_hyperparameters_<tune_name>.json
  ```

---

## Checklist

| Step | Done |
|------|------|
| `module load slurm/alpine` before `sbatch` | |
| Repo on `/projects/...` or PetaLibrary | |
| `input/` complete on cluster | |
| New **Linux** `.venv`, **`cu121` torch**, `requirements-gpu-alpine.txt` | |
| Slurm: `REPO_ROOT` + `source .venv/bin/activate` | |
| Slurm: `gcc/14.2.0` + `cuda/12.1.1` (or ROCm on `ami100`) | |
| `--account` if required | |

---

## If something fails

- **`CUDA driver` / `no CUDA GPUs`:** you’re not on a GPU node or `CUDA_VISIBLE_DEVICES` is wrong.
- **Import / libc errors:** mixing **`cuda/12.1.1`** with a **`cu130`** torch wheel — reinstall torch with **`cu121`** as above.
- **Missing data:** validate `input/` paths and filenames vs README.
- **SQLite / Optuna errors on arrays:** keep the DB on **scratch**; ensure all tasks use the **same** storage path and study name.
