"""Evaluation, baseline comparison, and subject-level analysis utilities."""

import logging
import numpy as np
import torch

from cnn_age_project.data.dataset import iter_memmap_batches
from cnn_age_project.models.mil import MILAgeRegressor
from cnn_age_project.utils.age_predictions import clip_predicted_ages_in_years
from cnn_age_project.utils.utils import make_tqdm

logger = logging.getLogger(__name__)


def run_epoch_metrics(
    model,
    x_mem,
    y_mem,
    indices,
    batch_size,
    device,
    criterion,
    amp_enabled,
    subject_codes=None,
    n_subjects=0,
    x_mean=0.0,
    x_std=1.0,
    y_mean=0.0,
    y_std=1.0,
    normalize_target=True,
    plot_max_points=100_000,
    debug_chunk_log_every=10,
    eval_label="Test Eval",
):
    """Evaluate model over indices and return window-level plus optional subject aggregates.

    Args:
        model (torch.nn.Module): Model under evaluation.
        x_mem (np.memmap): Input windows.
        y_mem (np.memmap): Ground-truth ages.
        indices (np.ndarray): Evaluation row indices.
        batch_size (int): Batch size.
        device (torch.device): Execution device.
        criterion (torch.nn.Module): Loss criterion used for logging loss.
        amp_enabled (bool): Whether AMP is enabled.
        subject_codes (np.ndarray | np.memmap | None): Optional subject code per row.
        n_subjects (int): Total subject count for aggregation arrays.
        x_mean (float): Input normalization mean.
        x_std (float): Input normalization std.
        y_mean (float): Target normalization mean.
        y_std (float): Target normalization std.
        normalize_target (bool): Whether targets are normalized in-model.
        plot_max_points (int): Max sampled points for scatter plotting.
        debug_chunk_log_every (int): Debug logging cadence.
        eval_label (str): Label for progress bar and logs (e.g. "Val Eval" during training, "Test Eval" for final evaluation).

    Returns:
        tuple: ``(avg_loss, r2, mae, eval_targets, eval_preds, sum_true_by_subject, sum_pred_by_subject, count_by_subject)``.
    """
    model.eval()
    n_eval = indices.shape[0]
    n_batches = (n_eval + batch_size - 1) // batch_size

    logger.info("%s: %d windows, %d batches", eval_label, n_eval, n_batches)

    running_loss = 0.0
    metric_abs_error = 0.0
    metric_sse = 0.0
    metric_sum_y = 0.0
    metric_sum_y2 = 0.0
    metric_n = 0

    sampled_targets = []
    sampled_preds = []
    points_per_batch = max(1, plot_max_points // max(1, n_batches))

    sum_true_by_subject = None
    sum_pred_by_subject = None
    count_by_subject = None
    if subject_codes is not None and n_subjects > 0:
        sum_true_by_subject = np.zeros(n_subjects, dtype=np.float64)
        sum_pred_by_subject = np.zeros(n_subjects, dtype=np.float64)
        count_by_subject = np.zeros(n_subjects, dtype=np.int64)
        logger.debug("Subject-level aggregation for %d subjects.", n_subjects)

    with torch.no_grad():
        pbar = make_tqdm(
            iter_memmap_batches(x_mem, y_mem, indices, batch_size, x_mean=x_mean, x_std=x_std),
            total=n_batches,
            desc=eval_label,
            unit="batch",
            position=0,
            leave=True,
        )
        for batch_num, (x_batch, y_batch) in enumerate(pbar, start=1):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            if normalize_target:
                y_batch_model = (y_batch - y_mean) / y_std
            else:
                y_batch_model = y_batch

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch_model)

            if normalize_target:
                outputs_year = outputs * y_std + y_mean
            else:
                outputs_year = outputs
            outputs_year = clip_predicted_ages_in_years(outputs_year)
            loss_year = torch.mean((outputs_year - y_batch) ** 2)
            running_loss += float(loss_year.item())

            outputs_np = outputs_year.detach().float().cpu().numpy()
            targets_np = y_batch.detach().float().cpu().numpy()

            diff = targets_np - outputs_np
            metric_abs_error += float(np.abs(diff).sum())
            metric_sse += float(np.square(diff).sum())
            metric_sum_y += float(targets_np.sum())
            metric_sum_y2 += float(np.square(targets_np).sum())
            metric_n += targets_np.shape[0]

            sample_idx = np.arange(0, targets_np.shape[0], max(1, targets_np.shape[0] // points_per_batch))[:points_per_batch]
            sampled_targets.append(targets_np[sample_idx])
            sampled_preds.append(outputs_np[sample_idx])

            if subject_codes is not None and n_subjects > 0:
                batch_start = (batch_num - 1) * batch_size
                batch_end = min(batch_start + targets_np.shape[0], n_eval)
                batch_indices = indices[batch_start:batch_end]
                batch_subj_codes = np.asarray(subject_codes[batch_indices], dtype=np.int32)
                valid = batch_subj_codes >= 0
                if np.any(valid):
                    valid_codes = batch_subj_codes[valid]
                    valid_targets = targets_np[valid].astype(np.float64)
                    valid_outputs = outputs_np[valid].astype(np.float64)

                    count_by_subject += np.bincount(valid_codes, minlength=n_subjects)
                    sum_true_by_subject += np.bincount(valid_codes, weights=valid_targets, minlength=n_subjects)
                    sum_pred_by_subject += np.bincount(valid_codes, weights=valid_outputs, minlength=n_subjects)

            pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{metric_abs_error / max(1, metric_n):.4f}")

    avg_loss = running_loss / max(1, n_batches)
    mae = metric_abs_error / max(1, metric_n)
    y_mean_batch = metric_sum_y / max(1, metric_n)
    sst = metric_sum_y2 - (metric_n * y_mean_batch * y_mean_batch)
    r2 = (1.0 - (metric_sse / sst)) if sst > 0 else np.nan

    eval_targets = np.concatenate(sampled_targets) if len(sampled_targets) > 0 else np.array([])
    eval_preds = np.concatenate(sampled_preds) if len(sampled_preds) > 0 else np.array([])
    return avg_loss, r2, mae, eval_targets, eval_preds, sum_true_by_subject, sum_pred_by_subject, count_by_subject


def compute_constant_baseline(y_mem, train_indices, test_indices, debug_chunk_log_every=10, chunk_size=1_000_000):
    """Compute constant-mean baseline metrics on the test split.

    Args:
        y_mem (np.memmap): Target array.
        train_indices (np.ndarray): Train split indices.
        test_indices (np.ndarray): Test split indices.
        debug_chunk_log_every (int): Debug logging cadence.
        chunk_size (int): Chunk size for memmap streaming.

    Returns:
        tuple[float, float, float, float]: Baseline prediction age, loss, R², and MAE.
    """
    train_sum = 0.0
    train_count = 0

    n_train_chunks = (train_indices.shape[0] + chunk_size - 1) // chunk_size
    for chunk_idx, start in enumerate(
        make_tqdm(range(0, train_indices.shape[0], chunk_size), desc="Baseline train", unit="chunk", position=0, leave=True),
        start=1,
    ):
        end = min(start + chunk_size, train_indices.shape[0])
        chunk_indices = train_indices[start:end]
        chunk_y = np.asarray(y_mem[chunk_indices], dtype=np.float32)
        train_sum += float(chunk_y.sum())
        train_count += int(chunk_y.shape[0])

    baseline_pred = train_sum / max(1, train_count)

    test_abs_error = 0.0
    test_sse = 0.0
    test_sum_y = 0.0
    test_sum_y2 = 0.0
    test_count = 0

    n_test_chunks = (test_indices.shape[0] + chunk_size - 1) // chunk_size
    for chunk_idx, start in enumerate(
        make_tqdm(range(0, test_indices.shape[0], chunk_size), desc="Baseline test", unit="chunk", position=0, leave=True),
        start=1,
    ):
        end = min(start + chunk_size, test_indices.shape[0])
        chunk_indices = test_indices[start:end]
        chunk_y = np.asarray(y_mem[chunk_indices], dtype=np.float32)
        diff = chunk_y - baseline_pred
        test_abs_error += float(np.abs(diff).sum())
        test_sse += float(np.square(diff).sum())
        test_sum_y += float(chunk_y.sum())
        test_sum_y2 += float(np.square(chunk_y).sum())
        test_count += int(chunk_y.shape[0])

    baseline_mae = test_abs_error / max(1, test_count)
    baseline_loss = test_sse / max(1, test_count)
    y_mean = test_sum_y / max(1, test_count)
    sst = test_sum_y2 - (test_count * y_mean * y_mean)
    baseline_r2 = (1.0 - (test_sse / sst)) if sst > 0 else np.nan
    return baseline_pred, baseline_loss, baseline_r2, baseline_mae


def predict_for_indices(
    model,
    x_mem,
    indices,
    batch_size,
    device,
    amp_enabled,
    x_mean=0.0,
    x_std=1.0,
    y_mean=0.0,
    y_std=1.0,
    normalize_target=True,
):
    """Predict ages for provided indices using batch inference.

    Args:
        model (torch.nn.Module): Model used for inference.
        x_mem (np.memmap): Input windows.
        indices (np.ndarray): Indices to predict.
        batch_size (int): Batch size.
        device (torch.device): Execution device.
        amp_enabled (bool): Whether AMP is enabled.
        x_mean (float): Input normalization mean.
        x_std (float): Input normalization std.
        y_mean (float): Target denormalization mean.
        y_std (float): Target denormalization std.
        normalize_target (bool): Whether output should be denormalized.

    Returns:
        np.ndarray: Predicted ages for ``indices``.
    """
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, indices.shape[0], batch_size):
            end = min(start + batch_size, indices.shape[0])
            batch_indices = indices[start:end]
            x_batch_np = np.asarray(x_mem[batch_indices], dtype=np.float32)
            x_batch_np = (x_batch_np - x_mean) / x_std
            x_batch = torch.from_numpy(x_batch_np).unsqueeze(1).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(x_batch)
            if normalize_target:
                outputs = outputs * y_std + y_mean
            outputs = clip_predicted_ages_in_years(outputs)
            preds.append(outputs.detach().float().cpu().numpy())
    return np.concatenate(preds) if len(preds) > 0 else np.array([])


def compute_subject_level_metrics(sum_true_by_subject, sum_pred_by_subject, count_by_subject, subject_codebook):
    """Convert per-subject sums into subject-level true/predicted means.

    Args:
        sum_true_by_subject (np.ndarray): Sum of true ages per subject.
        sum_pred_by_subject (np.ndarray): Sum of predicted ages per subject.
        count_by_subject (np.ndarray): Window count per subject.
        subject_codebook (list[str]): Subject ID lookup by code.

    Returns:
        tuple[np.ndarray, np.ndarray, list[str]]: Subject true means, predicted means, and subject IDs.
    """
    valid_subject_codes = np.where(count_by_subject > 0)[0]
    if valid_subject_codes.size == 0:
        return np.array([]), np.array([]), []

    true_means = (sum_true_by_subject[valid_subject_codes] / count_by_subject[valid_subject_codes]).astype(np.float64)
    pred_means = (sum_pred_by_subject[valid_subject_codes] / count_by_subject[valid_subject_codes]).astype(np.float64)
    subject_ids = [subject_codebook[int(code)] for code in valid_subject_codes]
    return true_means, pred_means, subject_ids


def compute_r2_from_arrays(y_true, y_pred):
    """Compute R² for numpy arrays, returning NaN when undefined.

    Args:
        y_true (np.ndarray): Ground-truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: R² score, or ``np.nan`` when undefined.
    """
    if y_true.size == 0:
        return np.nan
    y_mean = float(np.mean(y_true))
    sse = float(np.sum(np.square(y_true - y_pred)))
    sst = float(np.sum(np.square(y_true - y_mean)))
    return (1.0 - sse / sst) if sst > 0 else np.nan


def bootstrap_ci_subject_metrics(y_true_subject, y_pred_subject, rng, n_boot=1000, confidence=0.95):
    """Estimate MAE and R² confidence intervals via bootstrap resampling.

    Args:
        y_true_subject (np.ndarray): Subject-level true means.
        y_pred_subject (np.ndarray): Subject-level predicted means.
        rng (np.random.Generator): Random generator.
        n_boot (int): Number of bootstrap iterations.
        confidence (float): Confidence level for percentile interval.

    Returns:
        tuple[tuple[float, float], tuple[float, float]]: MAE CI and R² CI.
    """
    n_subjects = y_true_subject.shape[0]
    if n_subjects < 2:
        return (np.nan, np.nan), (np.nan, np.nan)

    mae_samples = np.empty(n_boot, dtype=np.float64)
    r2_samples = np.empty(n_boot, dtype=np.float64)

    for boot_idx in make_tqdm(range(n_boot), desc="Bootstrap CI", unit="iter", position=0, leave=True):
        sample_idx = rng.integers(0, n_subjects, size=n_subjects)
        sample_true = y_true_subject[sample_idx]
        sample_pred = y_pred_subject[sample_idx]
        mae_samples[boot_idx] = float(np.mean(np.abs(sample_true - sample_pred)))
        r2_samples[boot_idx] = compute_r2_from_arrays(sample_true, sample_pred)

    alpha = (1.0 - confidence) / 2.0
    low_q = 100.0 * alpha
    high_q = 100.0 * (1.0 - alpha)

    mae_ci = (float(np.nanpercentile(mae_samples, low_q)), float(np.nanpercentile(mae_samples, high_q)))
    r2_ci = (float(np.nanpercentile(r2_samples, low_q)), float(np.nanpercentile(r2_samples, high_q)))
    return mae_ci, r2_ci


def build_subject_examples(
    model,
    x_mem,
    y_mem,
    test_indices,
    subject_codes,
    subject_codebook,
    batch_size,
    device,
    amp_enabled,
    x_mean,
    x_std,
    y_mean,
    y_std,
    normalize_target,
    rng,
    subject_count,
    max_windows,
):
    """Build a compact set of random subject-level prediction examples for reporting.

    Args:
        model (torch.nn.Module): Trained model.
        x_mem (np.memmap): Input windows.
        y_mem (np.memmap): Target values.
        test_indices (np.ndarray): Test split indices.
        subject_codes (np.ndarray | np.memmap): Subject code per row.
        subject_codebook (list[str]): Subject codebook.
        batch_size (int): Inference batch size.
        device (torch.device): Execution device.
        amp_enabled (bool): Whether AMP is enabled.
        x_mean (float): Input normalization mean.
        x_std (float): Input normalization std.
        y_mean (float): Target denormalization mean.
        y_std (float): Target denormalization std.
        normalize_target (bool): Whether model outputs are denormalized.
        rng (np.random.Generator): Random generator.
        subject_count (int): Number of subjects to sample.
        max_windows (int): Max windows used per sampled subject.

    Returns:
        list[dict[str, Any]]: Summary entries for sampled subjects.
    """
    if subject_codes is None or len(subject_codebook) == 0:
        return []

    test_subject_codes = np.asarray(subject_codes[test_indices], dtype=np.int32)
    valid_test_subject_codes = np.unique(test_subject_codes[test_subject_codes >= 0])
    if valid_test_subject_codes.size == 0:
        return []

    choose_n = min(subject_count, valid_test_subject_codes.size)
    selected_codes = rng.choice(valid_test_subject_codes, size=choose_n, replace=False)

    examples = []
    for subject_code in selected_codes:
        subject_id = subject_codebook[int(subject_code)]
        subject_mask = test_subject_codes == int(subject_code)
        subject_window_indices = test_indices[subject_mask]
        if subject_window_indices.size == 0:
            continue

        sample_count = min(max_windows, subject_window_indices.size)
        sample_positions = np.linspace(0, subject_window_indices.size - 1, num=sample_count, dtype=np.int64)
        sampled_indices = subject_window_indices[sample_positions]

        if isinstance(model, MILAgeRegressor):
            # MIL path: build a single bag for this subject and run MIL model once.
            bag_x = np.asarray(x_mem[sampled_indices], dtype=np.float32)
            bag_x = (bag_x - x_mean) / x_std
            # Shape: (1, n_instances, 1, window_len)
            bag_tensor = torch.from_numpy(bag_x).unsqueeze(0).unsqueeze(2).to(device, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    outputs = model(bag_tensor)
            outputs = outputs.detach().float().cpu().numpy()
            if normalize_target:
                outputs = outputs * float(y_std) + float(y_mean)
            preds = clip_predicted_ages_in_years(outputs.reshape(-1))
        else:
            # CNN path: window-wise predictions using helper.
            preds = predict_for_indices(
                model=model,
                x_mem=x_mem,
                indices=sampled_indices,
                batch_size=batch_size,
                device=device,
                amp_enabled=amp_enabled,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                normalize_target=normalize_target,
            )

        if preds.size > 0:
            logger.debug(
                "Subject example preds for %s | min=%.4f max=%.4f mean=%.4f",
                subject_id,
                float(preds.min()),
                float(preds.max()),
                float(preds.mean()),
            )
        trues = np.asarray(y_mem[sampled_indices], dtype=np.float32)

        examples.append(
            {
                "subject_id": str(subject_id),
                "pred_mean": float(preds.mean()) if preds.size > 0 else np.nan,
                "true_mean": float(trues.mean()) if trues.size > 0 else np.nan,
                "windows_used": int(sample_count),
            }
        )

    return examples
