"""Model setup, optional tuning, and epoch-level training routines."""

import importlib.util
import logging
from time import perf_counter

import numpy as np
import torch
import torch.optim as optim

from cnn_age_project.data.dataset import iter_memmap_batches, sample_balanced_train_indices
from cnn_age_project.evaluation.evaluation import run_epoch_metrics
from cnn_age_project.models.cnn_model import EEGCNN
from cnn_age_project.training.losses import get_loss_function
from cnn_age_project.utils.utils import make_tqdm

logger = logging.getLogger(__name__)


def setup_model_and_optimizers(
    window_len,
    device,
    use_torch_compile,
    torch_compile_mode,
    torch_compile_dynamic,
    use_huber_loss,
    huber_beta,
    learning_rate,
):
    """Build model/loss/optimizer/scaler and optionally apply torch.compile.

    Args:
        window_len (int): Input window length used by the model.
        device (torch.device): Execution device.
        use_torch_compile (bool): Whether to attempt ``torch.compile``.
        torch_compile_mode (str): ``torch.compile`` mode string.
        torch_compile_dynamic (bool): Dynamic-shape flag for compile.
        use_huber_loss (bool): If True, use Huber loss.
        huber_beta (float): Beta parameter for Huber loss.
        learning_rate (float): Optimizer learning rate.

    Returns:
        tuple: ``(model, criterion, optimizer, scaler, amp_enabled, triton_available, compile_applied)``.
    """
    model = EEGCNN(window_len)
    model = model.to(device)

    triton_available = importlib.util.find_spec("triton") is not None
    compile_applied = False
    if use_torch_compile and hasattr(torch, "compile") and device.type == "cuda":
        if triton_available:
            try:
                if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
                    try:
                        torch._inductor.config.max_autotune_report_choices_stats = False
                        torch._inductor.config.trace.log_autotuning_results = False
                    except Exception:
                        pass

                model = torch.compile(
                    model,
                    mode=torch_compile_mode,
                    dynamic=torch_compile_dynamic,
                )
                compile_applied = True
                logger.info(
                    "`torch.compile` enabled (Triton available | mode=%s | dynamic=%s).",
                    torch_compile_mode,
                    torch_compile_dynamic,
                )
            except Exception as exc:
                logger.warning("`torch.compile` unavailable in current environment: %s", exc)
        else:
            logger.warning("`torch.compile` skipped: Triton not available.")

    criterion = get_loss_function(use_huber_loss=use_huber_loss, huber_beta=huber_beta)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, fused=(device.type == "cuda"))
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    return model, criterion, optimizer, scaler, amp_enabled, triton_available, compile_applied


def run_tuning_trial(
    trial_idx,
    total_trials,
    hparams,
    tune_epochs,
    x_mem,
    y_mem,
    train_indices,
    test_indices,
    subject_codes,
    subject_codebook,
    x_mean,
    x_std,
    y_mean,
    y_std,
    balanced_sorted_indices,
    balanced_offsets,
    balanced_counts,
    device,
    amp_enabled,
    rng,
    window_len,
    batch_size,
    normalize_target,
    subject_balanced_training,
    max_windows_per_subject_per_epoch,
    debug_chunk_log_every,
    plot_max_points,
):
    """Train a short trial with candidate hyperparameters and return held-out metrics.

    Args:
        trial_idx (int): 1-based trial index.
        total_trials (int): Total trial count.
        hparams (dict[str, Any]): Candidate hyperparameters for this trial.
        tune_epochs (int): Number of epochs for this trial.
        x_mem (np.memmap): Input memmap.
        y_mem (np.memmap): Target memmap.
        train_indices (np.ndarray): Train split indices.
        test_indices (np.ndarray): Test split indices.
        subject_codes (np.ndarray | np.memmap): Subject code per row.
        subject_codebook (list[str]): Code -> subject-id mapping.
        x_mean (float): Input normalization mean.
        x_std (float): Input normalization std.
        y_mean (float): Target normalization mean.
        y_std (float): Target normalization std.
        balanced_sorted_indices (np.ndarray | None): Subject-sorted train indices.
        balanced_offsets (np.ndarray | None): Per-subject offsets into sorted indices.
        balanced_counts (np.ndarray | None): Per-subject sample counts.
        device (torch.device): Execution device.
        amp_enabled (bool): Whether AMP autocast/scaler is enabled.
        rng (np.random.Generator): Random generator for sampling.
        window_len (int): Input window length.
        batch_size (int): Training/eval batch size.
        normalize_target (bool): Whether targets are normalized during optimization.
        subject_balanced_training (bool): Enables subject-balanced sampling.
        max_windows_per_subject_per_epoch (int): Sampling cap per subject.
        debug_chunk_log_every (int): Debug logging cadence.
        plot_max_points (int): Max plotted points for eval metric helper.

    Returns:
        dict[str, float | dict]: Trial metrics and hyperparameters.
    """
    trial_start = perf_counter()
    logger.info("[Tune %d/%d] Starting trial with hparams=%s", trial_idx, total_trials, hparams)

    model = EEGCNN(window_len).to(device)
    criterion = get_loss_function(
        use_huber_loss=hparams.get("use_huber_loss", True),
        huber_beta=float(hparams.get("huber_beta", 1.0)),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(hparams.get("learning_rate", 3e-4)),
        fused=(device.type == "cuda"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    max_per_subject = int(hparams.get("max_windows_per_subject_per_epoch", max_windows_per_subject_per_epoch))

    epoch_bar = make_tqdm(
        range(tune_epochs),
        desc=f"Tune Trial {trial_idx}/{total_trials}",
        unit="epoch",
        position=1,
        leave=False,
    )
    for epoch_idx in epoch_bar:
        model.train()
        if subject_balanced_training and balanced_sorted_indices is not None:
            epoch_train_indices = sample_balanced_train_indices(
                sorted_indices=balanced_sorted_indices,
                offsets=balanced_offsets,
                counts=balanced_counts,
                max_per_subject=max_per_subject,
                rng=rng,
            )
            if epoch_train_indices.size == 0:
                epoch_train_indices = train_indices
        else:
            epoch_train_indices = train_indices

        n_train = epoch_train_indices.shape[0]
        n_batches = (n_train + batch_size - 1) // batch_size
        batch_bar = make_tqdm(
            iter_memmap_batches(
                x_mem,
                y_mem,
                epoch_train_indices,
                batch_size,
                x_mean=x_mean,
                x_std=x_std,
            ),
            total=n_batches,
            desc=f"Tune T{trial_idx} E{epoch_idx + 1}/{tune_epochs}",
            unit="batch",
            position=2,
            leave=False,
        )

        running_train_loss = 0.0
        for batch_num, (x_batch, y_batch) in enumerate(batch_bar, start=1):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            if normalize_target:
                y_batch_model = (y_batch - y_mean) / y_std
            else:
                y_batch_model = y_batch

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch_model)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += float(loss.item())
            if (batch_num % debug_chunk_log_every) == 0:
                logger.debug(
                    "[Tune %d/%d] epoch %d/%d batch %d/%d | train_loss=%.4f",
                    trial_idx,
                    total_trials,
                    epoch_idx + 1,
                    tune_epochs,
                    batch_num,
                    n_batches,
                    running_train_loss / batch_num,
                )

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_bar.set_postfix(avg_loss=f"{(running_train_loss / max(1, n_batches)):.4f}")

    test_loss, test_r2, test_mae, _, _, _, _, _ = run_epoch_metrics(
        model=model,
        x_mem=x_mem,
        y_mem=y_mem,
        indices=test_indices,
        batch_size=batch_size,
        device=device,
        criterion=criterion,
        amp_enabled=amp_enabled,
        subject_codes=subject_codes,
        n_subjects=len(subject_codebook),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        normalize_target=normalize_target,
        plot_max_points=plot_max_points,
        debug_chunk_log_every=debug_chunk_log_every,
    )
    trial_seconds = perf_counter() - trial_start
    logger.info(
        "[Tune %d/%d] Completed in %.1fs | test_mae=%.3f test_r2=%.4f",
        trial_idx,
        total_trials,
        trial_seconds,
        test_mae,
        test_r2,
    )

    return {
        "hparams": hparams,
        "test_loss": float(test_loss),
        "test_r2": float(test_r2) if np.isfinite(test_r2) else np.nan,
        "test_mae": float(test_mae),
        "trial_seconds": float(trial_seconds),
    }


def train_model(
    model,
    criterion,
    optimizer,
    scaler,
    device,
    x_mem,
    y_mem,
    train_indices,
    balanced_sorted_indices,
    balanced_offsets,
    balanced_counts,
    rng,
    epochs,
    batch_size,
    x_mean,
    x_std,
    y_mean,
    y_std,
    normalize_target,
    subject_balanced_training,
    active_max_windows_per_subject,
    plot_max_points,
    metric_every_n_epochs,
    debug_chunk_log_every,
    early_stopping_enabled,
    early_stopping_patience,
    early_stopping_min_epochs,
    early_stopping_min_delta_abs,
    early_stopping_min_delta_rel,
    amp_enabled,
):
    """Run full training loop with optional balanced sampling and early stopping.

    Args:
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.Module): Loss criterion.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scaler (torch.amp.GradScaler): AMP gradient scaler.
        device (torch.device): Execution device.
        x_mem (np.memmap): Input memmap.
        y_mem (np.memmap): Target memmap.
        train_indices (np.ndarray): Training indices.
        balanced_sorted_indices (np.ndarray | None): Subject-sorted indices.
        balanced_offsets (np.ndarray | None): Per-subject offsets.
        balanced_counts (np.ndarray | None): Per-subject counts.
        rng (np.random.Generator): RNG for balanced sampling.
        epochs (int): Maximum training epochs.
        batch_size (int): Mini-batch size.
        x_mean (float): Input normalization mean.
        x_std (float): Input normalization std.
        y_mean (float): Target normalization mean.
        y_std (float): Target normalization std.
        normalize_target (bool): Whether targets are normalized.
        subject_balanced_training (bool): Enables balanced per-subject sampling.
        active_max_windows_per_subject (int): Subject sampling cap per epoch.
        plot_max_points (int): Max points collected for visualization.
        metric_every_n_epochs (int): Epoch interval for metric computation.
        debug_chunk_log_every (int): Debug logging cadence.
        early_stopping_enabled (bool): Enables early stopping.
        early_stopping_patience (int): Patience epochs before stopping.
        early_stopping_min_epochs (int): Minimum epochs before early stopping allowed.
        early_stopping_min_delta_abs (float): Absolute improvement threshold.
        early_stopping_min_delta_rel (float): Relative improvement threshold.
        amp_enabled (bool): Whether AMP is enabled.

    Returns:
        dict[str, Any]: Trained model plus loss/metric histories and best-epoch metadata.
    """
    train_losses = []
    r2_scores = []
    maes = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = None
    final_targets_plot = []
    final_preds_plot = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        metric_abs_error = 0.0
        metric_sse = 0.0
        metric_sum_y = 0.0
        metric_sum_y2 = 0.0
        metric_n = 0

        epoch_targets_plot = []
        epoch_preds_plot = []

        if subject_balanced_training and balanced_sorted_indices is not None:
            epoch_train_indices = sample_balanced_train_indices(
                sorted_indices=balanced_sorted_indices,
                offsets=balanced_offsets,
                counts=balanced_counts,
                max_per_subject=active_max_windows_per_subject,
                rng=rng,
            )
            if epoch_train_indices.size == 0:
                raise ValueError("Balanced sampling produced zero training windows.")
        else:
            epoch_train_indices = train_indices

        n_train = epoch_train_indices.shape[0]
        n_batches = (n_train + batch_size - 1) // batch_size
        points_per_batch = max(1, plot_max_points // max(1, n_batches))
        pbar = make_tqdm(
            iter_memmap_batches(x_mem, y_mem, epoch_train_indices, batch_size, x_mean=x_mean, x_std=x_std),
            total=n_batches,
            desc=f"Epoch {epoch + 1}/{epochs}",
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

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch_model)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if normalize_target:
                outputs_year = outputs * y_std + y_mean
            else:
                outputs_year = outputs

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

            if points_per_batch > 0:
                sample_idx = np.arange(0, targets_np.shape[0], max(1, targets_np.shape[0] // points_per_batch))[:points_per_batch]
                epoch_targets_plot.append(targets_np[sample_idx])
                epoch_preds_plot.append(outputs_np[sample_idx])

            if (batch_num % debug_chunk_log_every) == 0:
                logger.debug(
                    "Train epoch %d batch %d/%d | running_loss=%.4f running_mae=%.4f",
                    epoch + 1,
                    batch_num,
                    n_batches,
                    running_loss / batch_num,
                    metric_abs_error / max(1, metric_n),
                )

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / n_batches
        train_losses.append(avg_loss)

        compute_metrics = ((epoch + 1) % metric_every_n_epochs == 0) or (epoch + 1 == epochs)
        if compute_metrics:
            mae = metric_abs_error / max(1, metric_n)
            y_mean_metric = metric_sum_y / max(1, metric_n)
            sst = metric_sum_y2 - (metric_n * y_mean_metric * y_mean_metric)
            if sst > 0:
                r2 = 1.0 - (metric_sse / sst)
            else:
                r2 = np.nan
        else:
            r2 = np.nan
            mae = np.nan

        r2_scores.append(r2)
        maes.append(mae)

        logger.info("Epoch %d | Loss: %.4f | R2: %.4f | MAE: %.2f", epoch + 1, avg_loss, r2, mae)

        if len(epoch_targets_plot) > 0 and len(epoch_preds_plot) > 0:
            final_targets_plot = epoch_targets_plot
            final_preds_plot = epoch_preds_plot

        improvement_threshold = max(
            early_stopping_min_delta_abs,
            abs(best_loss) * early_stopping_min_delta_rel if np.isfinite(best_loss) else 0.0,
        )
        improved = (best_loss - avg_loss) > improvement_threshold

        if improved or not np.isfinite(best_loss):
            best_loss = avg_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_enabled
            and (epoch + 1) >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            logger.info(
                "Early stopping triggered at epoch %d (best epoch: %d, best loss: %.4f).",
                epoch + 1,
                best_epoch,
                best_loss,
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored best model weights from epoch %d.", best_epoch)

    return {
        "model": model,
        "train_losses": train_losses,
        "r2_scores": r2_scores,
        "maes": maes,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "final_targets_plot": final_targets_plot,
        "final_preds_plot": final_preds_plot,
    }
