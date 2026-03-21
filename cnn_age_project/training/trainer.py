"""Model setup, optional tuning, and epoch-level training routines."""

import importlib.util
import logging
from time import perf_counter

import numpy as np
import torch
import torch.optim as optim

from cnn_age_project.data.age_strata import build_mil_train_subject_inverse_frequency_probs
from cnn_age_project.data.dataset import (
    iter_subject_pseudo_bag_batches,
    iter_memmap_batches,
    sample_balanced_train_indices,
    sample_epoch_subject_pseudo_bags,
    sample_subject_pseudo_bag_indices,
    sample_weighted_train_epoch_indices,
)
from cnn_age_project.evaluation.evaluation import run_epoch_metrics
from cnn_age_project.utils.age_predictions import clip_predicted_ages_in_years
from cnn_age_project.models.cnn_model import EEGCNN
from cnn_age_project.models.mil import MILAgeRegressor, MILInstanceEncoder, summarize_parameter_counts
from cnn_age_project.training.losses import get_loss_function
from cnn_age_project.utils.utils import make_tqdm
from cnn_age_project.config import (
    MIL_INVERSE_FREQUENCY_SUBJECT_SAMPLING,
    MIL_SUBJECT_DRAWS_PER_EPOCH,
    STRATIFY_AGE_BIN_YEARS,
    STRATIFY_TAIL_HIGH_MIN_AGE,
    STRATIFY_TAIL_LOW_MAX_AGE,
)

logger = logging.getLogger(__name__)


def prepare_mil_instance_encoder(
    window_len,
    device,
    freeze_encoder=True,
    base_cnn_state_dict=None,
    cnn_embedding_dim: int = 128,
    cnn_dropout: float = 0.0,
):
    """Prepare MIL Step 1 encoder by stripping CNN head and freezing features.

    Args:
        window_len (int): Input window length used by the CNN architecture.
        device (torch.device): Device where encoder will be placed.
        freeze_encoder (bool): If True, disable encoder gradients.
        base_cnn_state_dict (dict[str, torch.Tensor] | None): Optional pretrained
            EEGCNN weights. If provided, they are loaded before encoder extraction.

    Returns:
        MILInstanceEncoder: MIL-ready instance encoder module.
    """
    logger.info(
        "MIL encoder: freeze=%s pretrained=%s",
        freeze_encoder,
        base_cnn_state_dict is not None,
    )
    base_cnn = EEGCNN(window_len, embedding_dim=cnn_embedding_dim, dropout=cnn_dropout)
    if base_cnn_state_dict is not None:
        missing, unexpected = base_cnn.load_state_dict(base_cnn_state_dict, strict=False)
        if missing:
            logger.debug("MIL encoder load missing keys: %s", missing)
        if unexpected:
            logger.debug("MIL encoder load unexpected keys: %s", unexpected)
    encoder = MILInstanceEncoder.from_eegcnn(base_cnn, freeze_encoder=freeze_encoder).to(device)
    total_params, trainable_params = summarize_parameter_counts(encoder)
    logger.info(
        "MIL encoder ready | params=%d trainable=%d",
        total_params,
        trainable_params,
    )
    return encoder


def build_mil_gated_attention_model(
    window_len,
    device,
    freeze_encoder=True,
    base_cnn_state_dict=None,
    feature_dim=128,
    cnn_embedding_dim: int = 128,
    cnn_dropout: float = 0.0,
    attention_dim=128,
    regressor_hidden_dim=64,
    pooling_type="gated",
    attention_dropout=0.0,
    regressor_dropout=0.0,
):
    """Construct Step 2 MIL model with gated attention on top of encoder.

    Args:
        window_len (int): Input window length.
        device (torch.device): Device where model is allocated.
        freeze_encoder (bool): Freeze CNN encoder parameters if True.
        base_cnn_state_dict (dict[str, torch.Tensor] | None): Optional pretrained
            base CNN state dict loaded prior to encoder extraction.
        feature_dim (int): Encoder embedding size.
        attention_dim (int): Gated attention hidden size.
        regressor_hidden_dim (int): Bag-level regressor hidden size.
        pooling_type (str): Bag pooling strategy, ``gated`` or ``mean``.
        attention_dropout (float): Dropout within gated attention features.
        regressor_dropout (float): Dropout in bag regressor after hidden ReLU.

    Returns:
        MILAgeRegressor: MIL age regressor with gated attention.
    """
    encoder = prepare_mil_instance_encoder(
        window_len=window_len,
        device=device,
        freeze_encoder=freeze_encoder,
        base_cnn_state_dict=base_cnn_state_dict,
        cnn_embedding_dim=cnn_embedding_dim,
        cnn_dropout=cnn_dropout,
    )

    mil_model = MILAgeRegressor(
        instance_encoder=encoder,
        feature_dim=feature_dim,
        attention_dim=attention_dim,
        regressor_hidden_dim=regressor_hidden_dim,
        pooling_type=pooling_type,
        attention_dropout=attention_dropout,
        regressor_dropout=regressor_dropout,
    ).to(device)

    total_params, trainable_params = summarize_parameter_counts(mil_model)
    logger.info(
        "Step 2 MIL model ready | total_params=%d | trainable_params=%d | "
        "feature_dim=%d attention_dim=%d regressor_hidden_dim=%d pooling_type=%s attention_dropout=%.3f",
        total_params,
        trainable_params,
        feature_dim,
        attention_dim,
        regressor_hidden_dim,
        pooling_type,
        float(attention_dropout),
    )
    return mil_model


def sample_mil_epoch_pseudo_bags(
    balanced_sorted_indices,
    balanced_offsets,
    balanced_counts,
    rng,
    pseudo_bag_min_windows=256,
    pseudo_bag_max_windows=500,
    allow_replacement_when_small=False,
    sampling_strategy="random",
):
    """Sample one random pseudo-bag per subject for MIL epoch training.

    Args:
        balanced_sorted_indices (np.ndarray): Subject-sorted train indices.
        balanced_offsets (np.ndarray): Per-subject offset boundaries.
        balanced_counts (np.ndarray): Per-subject train-window counts.
        rng (np.random.Generator): Random generator for reproducible sampling.
        pseudo_bag_min_windows (int): Lower bound for sampled bag size.
        pseudo_bag_max_windows (int): Upper bound for sampled bag size.
        allow_replacement_when_small (bool): If True, upsample small subjects with
            replacement; if False, use all unique windows (bag may be shorter).
        sampling_strategy (str): ``random`` or ``sequential`` bag sampling.

    Returns:
        list[tuple[int, np.ndarray]]: Subject pseudo-bags for the epoch.
    """
    pseudo_bags = sample_epoch_subject_pseudo_bags(
        sorted_indices=balanced_sorted_indices,
        offsets=balanced_offsets,
        counts=balanced_counts,
        rng=rng,
        min_windows=pseudo_bag_min_windows,
        max_windows=pseudo_bag_max_windows,
        allow_replacement_when_small=allow_replacement_when_small,
        sampling_strategy=sampling_strategy,
    )

    bag_sizes = np.asarray([bag_indices.size for _, bag_indices in pseudo_bags], dtype=np.int32)
    if bag_sizes.size == 0:
        logger.warning("MIL pseudo-bag sampler produced zero bags for this epoch.")
        return pseudo_bags

    logger.info(
        "MIL pseudo-bags sampled | n_bags=%d | bag_size[min/median/max]=%d/%d/%d | range=[%d,%d]",
        bag_sizes.size,
        int(bag_sizes.min()),
        int(np.median(bag_sizes)),
        int(bag_sizes.max()),
        pseudo_bag_min_windows,
        pseudo_bag_max_windows,
    )
    return pseudo_bags


def sample_mil_epoch_weighted_subject_pseudo_bags(
    balanced_sorted_indices,
    balanced_offsets,
    balanced_counts,
    rng,
    pseudo_bag_min_windows,
    pseudo_bag_max_windows,
    allow_replacement_when_small,
    sampling_strategy,
    eligible_subject_codes,
    subject_sample_probs,
    num_draws,
):
    """Sample ``num_draws`` pseudo-bags by drawing subjects with replacement (inverse-frequency weights).

    Used for MIL **training** only so rare age strata appear as often in expectation as common ones.
    """
    n_eligible = int(eligible_subject_codes.shape[0])
    if n_eligible == 0:
        return []
    draws = int(num_draws)
    if draws <= 0:
        return []

    pick = rng.choice(n_eligible, size=draws, replace=True, p=subject_sample_probs)
    bags = []
    for j in pick:
        subject_code = int(eligible_subject_codes[j])
        bag_indices = sample_subject_pseudo_bag_indices(
            sorted_indices=balanced_sorted_indices,
            offsets=balanced_offsets,
            counts=balanced_counts,
            subject_code=subject_code,
            rng=rng,
            min_windows=pseudo_bag_min_windows,
            max_windows=pseudo_bag_max_windows,
            allow_replacement_when_small=allow_replacement_when_small,
            sampling_strategy=sampling_strategy,
        )
        if bag_indices.size > 0:
            bags.append((subject_code, bag_indices))

    rng.shuffle(bags)
    bag_sizes = np.asarray([b.size for _, b in bags], dtype=np.int32)
    if bag_sizes.size == 0:
        logger.warning("MIL weighted pseudo-bag sampler produced zero bags for this epoch.")
        return bags

    n_unique_subj = len({c for c, _ in bags})
    logger.info(
        "MIL weighted pseudo-bags | n_bags=%d | unique_subjects=%d | draws=%d | "
        "bag_size[min/median/max]=%d/%d/%d | range=[%d,%d]",
        bag_sizes.size,
        n_unique_subj,
        draws,
        int(bag_sizes.min()),
        int(np.median(bag_sizes)),
        int(bag_sizes.max()),
        pseudo_bag_min_windows,
        pseudo_bag_max_windows,
    )
    return bags


def build_mil_pseudo_bag_batch_iterator(
    x_mem,
    y_mem,
    pseudo_bags,
    batch_size,
    x_mean=0.0,
    x_std=1.0,
):
    """Create iterator that yields MIL pseudo-bag mini-batches.

    Args:
        x_mem (np.memmap): Input windows.
        y_mem (np.memmap): Window-level age targets.
        pseudo_bags (list[tuple[int, np.ndarray]]): Sampled bags for this epoch.
        batch_size (int): Number of subject bags per mini-batch.
        x_mean (float): Input normalization mean.
        x_std (float): Input normalization std.

    Returns:
        Iterator: Yields ``(x_bags, y_bags, subject_codes, bag_mask)`` tuples.
    """
    n_bags = len(pseudo_bags)
    n_batches = (n_bags + batch_size - 1) // max(1, batch_size)
    logger.debug("MIL pseudo-bags: n_bags=%d batch_size=%d n_batches=%d", n_bags, batch_size, n_batches)
    return iter_subject_pseudo_bag_batches(
        x_mem=x_mem,
        y_mem=y_mem,
        pseudo_bags=pseudo_bags,
        batch_size=batch_size,
        x_mean=x_mean,
        x_std=x_std,
    )


def configure_mil_finetune_optimizer(
    mil_model,
    device,
    encoder_learning_rate=1e-5,
    mil_head_learning_rate=5e-5,
    weight_decay=1e-2,
):
    """Configure Step 3 optimizer with differential learning rates.

    Args:
        mil_model (MILAgeRegressor): MIL model to optimize.
        device (torch.device): Device used to determine fused optimizer support.
        encoder_learning_rate (float): Learning rate for CNN encoder params.
        mil_head_learning_rate (float): Learning rate for attention/regressor params.
        weight_decay (float): AdamW weight decay.

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """
    mil_model.set_encoder_trainable(True)

    encoder_params = [p for p in mil_model.instance_encoder.parameters() if p.requires_grad]
    encoder_param_ids = {id(p) for p in encoder_params}
    mil_head_params = [
        p
        for p in mil_model.parameters()
        if p.requires_grad and id(p) not in encoder_param_ids
    ]

    n_encoder_params = int(sum(p.numel() for p in encoder_params))
    n_head_params = int(sum(p.numel() for p in mil_head_params))
    if (n_encoder_params + n_head_params) == 0:
        raise ValueError("No trainable parameters detected for MIL fine-tuning.")

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": float(encoder_learning_rate)})
    if mil_head_params:
        param_groups.append({"params": mil_head_params, "lr": float(mil_head_learning_rate)})

    optimizer = optim.AdamW(
        param_groups,
        lr=float(mil_head_learning_rate),
        weight_decay=float(weight_decay),
        fused=(device.type == "cuda"),
    )

    logger.info(
        "MIL Step 3 optimizer configured | encoder_lr=%.2e head_lr=%.2e "
        "encoder_trainable_params=%d head_trainable_params=%d weight_decay=%.4g",
        encoder_learning_rate,
        mil_head_learning_rate,
        n_encoder_params,
        n_head_params,
        weight_decay,
    )
    return optimizer


def run_mil_finetune_training(
    mil_model,
    criterion,
    optimizer,
    scaler,
    device,
    x_mem,
    y_mem,
    balanced_sorted_indices,
    balanced_offsets,
    balanced_counts,
    rng,
    epochs,
    bag_batch_size,
    x_mean,
    x_std,
    y_mean,
    y_std,
    normalize_target,
    pseudo_bag_min_windows,
    pseudo_bag_max_windows,
    allow_replacement_when_small,
    sampling_strategy,
    debug_chunk_log_every,
    amp_enabled,
    early_stopping_enabled=True,
    early_stopping_patience=3,
    early_stopping_min_epochs=2,
    early_stopping_min_delta_abs=1e-4,
    early_stopping_min_delta_rel=1e-3,
    grad_clip_norm=0.0,
    encoder_warmup_epochs: int = 0,
    encoder_learning_rate: float | None = None,
    subject_codebook: list[str] | None = None,
    train_age_map: dict[str, float] | None = None,
    subject_stratum_merged: dict[str, int] | None = None,
    mil_inverse_frequency_subject_sampling: bool | None = None,
    mil_subject_draws_per_epoch: int | None = None,
    stratify_tail_low_max_age: float | None = None,
    stratify_tail_high_min_age: float | None = None,
    stratify_age_bin_years: float | None = None,
):
    """Run Step 3 full-model fine-tuning over stochastic pseudo-bags.

    Args:
        mil_model (MILAgeRegressor): MIL model (encoder + attention + regressor).
        criterion (torch.nn.Module): Loss criterion.
        optimizer (torch.optim.Optimizer): Optimizer with parameter groups.
        scaler (torch.amp.GradScaler): Gradient scaler for AMP.
        device (torch.device): Execution device.
        x_mem (np.memmap): Input windows.
        y_mem (np.memmap): Age targets.
        balanced_sorted_indices (np.ndarray): Subject-sorted train indices.
        balanced_offsets (np.ndarray): Per-subject offsets.
        balanced_counts (np.ndarray): Per-subject counts.
        rng (np.random.Generator): RNG for pseudo-bag sampling.
        epochs (int): Number of fine-tuning epochs.
        bag_batch_size (int): Number of bags per optimizer step.
        x_mean (float): Input normalization mean.
        x_std (float): Input normalization std.
        y_mean (float): Target normalization mean.
        y_std (float): Target normalization std.
        normalize_target (bool): If True, train in normalized age space.
        pseudo_bag_min_windows (int): Minimum pseudo-bag size.
        pseudo_bag_max_windows (int): Maximum pseudo-bag size.
        allow_replacement_when_small (bool): Whether small subjects are sampled
            with replacement to meet bag size.
        sampling_strategy (str): ``random`` or ``sequential`` pseudo-bag sampling.
        debug_chunk_log_every (int): Debug logging cadence.
        amp_enabled (bool): Whether AMP autocast/scaler are enabled.
        early_stopping_enabled (bool): Enable early stopping.
        early_stopping_patience (int): Patience epochs.
        early_stopping_min_epochs (int): Minimum epochs before early stop allowed.
        early_stopping_min_delta_abs (float): Absolute improvement threshold.
        early_stopping_min_delta_rel (float): Relative improvement threshold.
        grad_clip_norm (float): Max gradient norm (0 = disabled).
        subject_codebook (list[str] | None): For inverse-frequency MIL sampling.
        train_age_map (dict | None): Train subject -> age.
        subject_stratum_merged (dict | None): Optional merged stratum ids (same as split/CNN).
        mil_inverse_frequency_subject_sampling (bool | None): If True, weighted subject draws;
            if None, use ``MIL_INVERSE_FREQUENCY_SUBJECT_SAMPLING`` from config.
        mil_subject_draws_per_epoch (int | None): With replacement draws per epoch; None =
            one per eligible subject (same cardinality as unweighted epoch).
        stratify_tail_low_max_age (float | None): Age stratum boundary (None = config).
        stratify_tail_high_min_age (float | None): Old tail boundary (None = config).
        stratify_age_bin_years (float | None): Interior bin width (None = config).

    Returns:
        dict[str, Any]: Fine-tuning history and best-epoch metadata.
    """
    train_losses = []
    maes = []
    r2_scores = []
    best_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    use_mil_if = (
        MIL_INVERSE_FREQUENCY_SUBJECT_SAMPLING
        if mil_inverse_frequency_subject_sampling is None
        else bool(mil_inverse_frequency_subject_sampling)
    )
    draws_cfg = MIL_SUBJECT_DRAWS_PER_EPOCH if mil_subject_draws_per_epoch is None else mil_subject_draws_per_epoch
    tail_lo = STRATIFY_TAIL_LOW_MAX_AGE if stratify_tail_low_max_age is None else float(stratify_tail_low_max_age)
    tail_hi = STRATIFY_TAIL_HIGH_MIN_AGE if stratify_tail_high_min_age is None else float(stratify_tail_high_min_age)
    bin_y = STRATIFY_AGE_BIN_YEARS if stratify_age_bin_years is None else float(stratify_age_bin_years)

    mil_eligible_codes = None
    mil_subject_probs = None
    mil_num_draws = None
    if use_mil_if:
        if subject_codebook is None or train_age_map is None:
            logger.warning(
                "MIL inverse-frequency subject sampling disabled: missing subject_codebook or train_age_map."
            )
            use_mil_if = False
        else:
            mil_eligible_codes, mil_subject_probs = build_mil_train_subject_inverse_frequency_probs(
                balanced_counts=np.asarray(balanced_counts),
                subject_codebook=subject_codebook,
                train_age_map=train_age_map,
                subject_stratum_merged=subject_stratum_merged,
                tail_low_max_age=tail_lo,
                tail_high_min_age=tail_hi,
                bin_years=bin_y,
            )
            mil_num_draws = int(draws_cfg) if draws_cfg is not None else int(mil_eligible_codes.shape[0])
            logger.info(
                "MIL training | inverse-frequency subject sampling | eligible_subjects=%d | draws/epoch=%d "
                "| strata use tail_lo=%.1f tail_hi=%.1f bin_y=%.1f",
                int(mil_eligible_codes.shape[0]),
                mil_num_draws,
                tail_lo,
                tail_hi,
                bin_y,
            )

    # Cache original encoder learning rate (if provided) so we can
    # restore it after the warmup "frozen" phase.
    encoder_group_lr = float(encoder_learning_rate) if encoder_learning_rate is not None else None

    for epoch in range(epochs):
        mil_model.train()

        if use_mil_if and mil_eligible_codes is not None and mil_subject_probs is not None and mil_num_draws is not None:
            pseudo_bags = sample_mil_epoch_weighted_subject_pseudo_bags(
                balanced_sorted_indices=balanced_sorted_indices,
                balanced_offsets=balanced_offsets,
                balanced_counts=balanced_counts,
                rng=rng,
                pseudo_bag_min_windows=pseudo_bag_min_windows,
                pseudo_bag_max_windows=pseudo_bag_max_windows,
                allow_replacement_when_small=allow_replacement_when_small,
                sampling_strategy=sampling_strategy,
                eligible_subject_codes=mil_eligible_codes,
                subject_sample_probs=mil_subject_probs,
                num_draws=mil_num_draws,
            )
        else:
            pseudo_bags = sample_mil_epoch_pseudo_bags(
                balanced_sorted_indices=balanced_sorted_indices,
                balanced_offsets=balanced_offsets,
                balanced_counts=balanced_counts,
                rng=rng,
                pseudo_bag_min_windows=pseudo_bag_min_windows,
                pseudo_bag_max_windows=pseudo_bag_max_windows,
                allow_replacement_when_small=allow_replacement_when_small,
                sampling_strategy=sampling_strategy,
            )
        if len(pseudo_bags) == 0:
            raise ValueError("MIL fine-tuning received zero pseudo-bags. Check subject grouping inputs.")

        n_batches = (len(pseudo_bags) + bag_batch_size - 1) // bag_batch_size
        batch_iter = build_mil_pseudo_bag_batch_iterator(
            x_mem=x_mem,
            y_mem=y_mem,
            pseudo_bags=pseudo_bags,
            batch_size=bag_batch_size,
            x_mean=x_mean,
            x_std=x_std,
        )
        pbar = make_tqdm(
            batch_iter,
            total=n_batches,
            desc=f"MIL FineTune {epoch + 1}/{epochs}",
            unit="batch",
            position=0,
            leave=True,
        )

        running_loss = 0.0
        metric_abs_error = 0.0
        metric_sse = 0.0
        metric_sum_y = 0.0
        metric_sum_y2 = 0.0
        metric_n = 0

        # During warmup epochs, freeze encoder updates by setting the
        # encoder parameter group's learning rate to zero. After warmup
        # is finished, restore the configured encoder learning rate.
        if encoder_group_lr is not None and encoder_warmup_epochs > 0:
            # By convention, the first param group corresponds to encoder
            # params when configured via `configure_mil_finetune_optimizer`.
            if epoch < encoder_warmup_epochs:
                optimizer.param_groups[0]["lr"] = 0.0
            else:
                optimizer.param_groups[0]["lr"] = encoder_group_lr

        for batch_num, (x_bags, y_bags, _, bag_mask) in enumerate(pbar, start=1):
            x_bags = x_bags.to(device, non_blocking=True)
            y_bags = y_bags.to(device, non_blocking=True)
            bag_mask = bag_mask.to(device, non_blocking=True)

            if normalize_target:
                y_model = (y_bags - y_mean) / y_std
            else:
                y_model = y_bags

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = mil_model(x_bags, bag_mask=bag_mask)
                loss = criterion(outputs, y_model)

            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mil_model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            outputs_year = outputs * y_std + y_mean if normalize_target else outputs
            outputs_year = clip_predicted_ages_in_years(outputs_year)
            loss_year = torch.mean((outputs_year - y_bags) ** 2)
            running_loss += float(loss_year.item())

            outputs_np = outputs_year.detach().float().cpu().numpy()
            targets_np = y_bags.detach().float().cpu().numpy()
            diff = targets_np - outputs_np

            metric_abs_error += float(np.abs(diff).sum())
            metric_sse += float(np.square(diff).sum())
            metric_sum_y += float(targets_np.sum())
            metric_sum_y2 += float(np.square(targets_np).sum())
            metric_n += targets_np.shape[0]

            pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{metric_abs_error / max(1, metric_n):.4f}")

        avg_loss = running_loss / max(1, n_batches)
        mae = metric_abs_error / max(1, metric_n)
        y_bar = metric_sum_y / max(1, metric_n)
        sst = metric_sum_y2 - (metric_n * y_bar * y_bar)
        r2 = (1.0 - (metric_sse / sst)) if sst > 0 else np.nan

        train_losses.append(avg_loss)
        maes.append(mae)
        r2_scores.append(r2)

        logger.info(
            "MIL epoch %d/%d | loss=%.4f mae=%.3f r2=%.4f",
            epoch + 1,
            epochs,
            avg_loss,
            mae,
            r2,
        )

        improvement_threshold = max(
            early_stopping_min_delta_abs,
            abs(best_loss) * early_stopping_min_delta_rel if np.isfinite(best_loss) else 0.0,
        )
        improved = (best_loss - avg_loss) > improvement_threshold

        if improved or not np.isfinite(best_loss):
            best_loss = avg_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in mil_model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_enabled
            and (epoch + 1) >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            logger.info(
                "MIL early stopping at epoch %d (best epoch: %d, best loss: %.4f).",
                epoch + 1,
                best_epoch,
                best_loss,
            )
            break

    if best_state_dict is not None:
        mil_model.load_state_dict(best_state_dict)
        logger.info("MIL restored best weights (epoch %d).", best_epoch)

    return {
        "model": mil_model,
        "train_losses": train_losses,
        "maes": maes,
        "r2_scores": r2_scores,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
    }


def evaluate_mil_on_subject_bags(
    mil_model,
    criterion,
    x_mem,
    y_mem,
    eval_indices,
    subject_codes,
    n_subjects,
    batch_size,
    device,
    x_mean,
    x_std,
    y_mean,
    y_std,
    normalize_target,
    rng,
    pseudo_bag_min_windows,
    pseudo_bag_max_windows,
    allow_replacement_when_small,
    sampling_strategy,
):
    """Evaluate MIL model using one pseudo-bag per subject from eval split.

    Uses **unweighted** subject coverage (every eval subject with windows once) so
    metrics reflect the natural age mix of the split—not inverse-frequency training.

    Returns:
        dict[str, Any]: loss/r2/mae plus arrays and subject aggregations.
    """
    from cnn_age_project.data.dataset import build_subject_group_index  # local import avoids cycles

    mil_model.eval()

    sorted_idx, offsets, counts = build_subject_group_index(
        train_indices=eval_indices,
        subject_codes=subject_codes,
        n_subjects=n_subjects,
    )

    pseudo_bags = sample_mil_epoch_pseudo_bags(
        balanced_sorted_indices=sorted_idx,
        balanced_offsets=offsets,
        balanced_counts=counts,
        rng=rng,
        pseudo_bag_min_windows=pseudo_bag_min_windows,
        pseudo_bag_max_windows=pseudo_bag_max_windows,
        allow_replacement_when_small=allow_replacement_when_small,
        sampling_strategy=sampling_strategy,
    )
    if len(pseudo_bags) == 0:
        raise ValueError("MIL evaluation produced zero pseudo-bags on eval split.")

    n_batches = (len(pseudo_bags) + batch_size - 1) // batch_size
    batch_iter = build_mil_pseudo_bag_batch_iterator(
        x_mem=x_mem,
        y_mem=y_mem,
        pseudo_bags=pseudo_bags,
        batch_size=batch_size,
        x_mean=x_mean,
        x_std=x_std,
    )

    running_loss = 0.0
    pred_chunks = []
    true_chunks = []
    sum_true_by_subject = np.zeros((n_subjects,), dtype=np.float64)
    sum_pred_by_subject = np.zeros((n_subjects,), dtype=np.float64)
    count_by_subject = np.zeros((n_subjects,), dtype=np.int64)

    with torch.no_grad():
        for x_bags, y_bags, bag_subject_codes, bag_mask in batch_iter:
            x_bags = x_bags.to(device, non_blocking=True)
            y_bags = y_bags.to(device, non_blocking=True)
            bag_mask = bag_mask.to(device, non_blocking=True)

            y_model = (y_bags - y_mean) / y_std if normalize_target else y_bags
            outputs = mil_model(x_bags, bag_mask=bag_mask)
            loss = criterion(outputs, y_model)

            outputs_year = outputs * y_std + y_mean if normalize_target else outputs
            outputs_year = clip_predicted_ages_in_years(outputs_year)
            mse_year = torch.mean((outputs_year - y_bags) ** 2)
            running_loss += float(mse_year.item())

            pred_np = outputs_year.detach().float().cpu().numpy()
            true_np = y_bags.detach().float().cpu().numpy()
            pred_chunks.append(pred_np)
            true_chunks.append(true_np)

            for subj_code, true_value, pred_value in zip(bag_subject_codes, true_np, pred_np):
                subj_idx = int(subj_code)
                if subj_idx < 0 or subj_idx >= n_subjects:
                    continue
                sum_true_by_subject[subj_idx] += float(true_value)
                sum_pred_by_subject[subj_idx] += float(pred_value)
                count_by_subject[subj_idx] += 1

    final_preds = np.concatenate(pred_chunks) if pred_chunks else np.array([], dtype=np.float32)
    final_targets = np.concatenate(true_chunks) if true_chunks else np.array([], dtype=np.float32)
    if final_targets.size == 0:
        raise ValueError("MIL evaluation produced no predictions.")

    diff = final_targets - final_preds
    mae = float(np.mean(np.abs(diff)))
    sse = float(np.sum(np.square(diff)))
    centered = final_targets - float(np.mean(final_targets))
    sst = float(np.sum(np.square(centered)))
    r2 = (1.0 - (sse / sst)) if sst > 0 else np.nan
    avg_loss = running_loss / max(1, n_batches)

    return {
        "test_loss": float(avg_loss),
        "test_r2": float(r2) if np.isfinite(r2) else np.nan,
        "test_mae": float(mae),
        "final_targets": final_targets,
        "final_preds": final_preds,
        "sum_true_by_subject": sum_true_by_subject,
        "sum_pred_by_subject": sum_pred_by_subject,
        "count_by_subject": count_by_subject,
    }


def setup_model_and_optimizers(
    window_len,
    device,
    use_torch_compile,
    torch_compile_mode,
    torch_compile_dynamic,
    use_huber_loss,
    huber_beta,
    learning_rate,
    cnn_embedding_dim: int = 128,
    cnn_dropout: float = 0.0,
    cnn_weight_decay: float = 0.0,
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
        cnn_weight_decay (float): Weight decay for AdamW (0 = disabled).

    Returns:
        tuple: ``(model, criterion, optimizer, scaler, amp_enabled, triton_available, compile_applied)``.
    """
    model = EEGCNN(window_len, embedding_dim=cnn_embedding_dim, dropout=cnn_dropout)
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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(cnn_weight_decay),
        fused=(device.type == "cuda"),
    )
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
    grad_clip_norm=0.0,
    val_indices=None,
    train_window_sample_weights=None,
    use_age_weighted_window_sampling=False,
):
    """Train a short trial with candidate hyperparameters and return held-out metrics.

    Tuning **minimizes** ``selection_mae``: validation window MAE when ``val_indices`` is set,
    otherwise test window MAE. ``test_mae`` is always the test split for post-hoc reporting.

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
    logger.info("Tune trial %d/%d | lr=%.0e dropout=%.2f emb=%d", trial_idx, total_trials,
                float(hparams.get("learning_rate", 3e-4)), float(hparams.get("cnn_dropout", 0)),
                int(hparams.get("cnn_embedding_dim", 128)))

    model = EEGCNN(
        window_len,
        embedding_dim=int(hparams.get("cnn_embedding_dim", 128)),
        dropout=float(hparams.get("cnn_dropout", 0.0)),
    ).to(device)
    criterion = get_loss_function(
        use_huber_loss=hparams.get("use_huber_loss", True),
        huber_beta=float(hparams.get("huber_beta", 1.0)),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(hparams.get("learning_rate", 3e-4)),
        weight_decay=float(hparams.get("cnn_weight_decay", 0.0)),
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
        if (
            use_age_weighted_window_sampling
            and train_window_sample_weights is not None
            and train_window_sample_weights.shape[0] == train_indices.shape[0]
        ):
            epoch_train_indices = sample_weighted_train_epoch_indices(
                train_indices,
                train_window_sample_weights,
                num_samples=int(train_indices.shape[0]),
                rng=rng,
                replace=True,
            )
        elif subject_balanced_training and balanced_sorted_indices is not None:
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
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += float(loss.item())
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

    selection_mae = float(test_mae)
    val_mae = None
    val_r2 = None
    if val_indices is not None and val_indices.size > 0:
        val_loss, val_r2, val_mae, _, _, _, _, _ = run_epoch_metrics(
            model=model,
            x_mem=x_mem,
            y_mem=y_mem,
            indices=val_indices,
            batch_size=batch_size,
            device=device,
            criterion=criterion,
            amp_enabled=amp_enabled,
            subject_codes=None,
            n_subjects=0,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            normalize_target=normalize_target,
            plot_max_points=0,
            debug_chunk_log_every=debug_chunk_log_every,
            eval_label="Val Eval",
        )
        selection_mae = float(val_mae)
        logger.info("Tune %d/%d done | val_mae=%.3f (selection) test_mae=%.3f", trial_idx, total_trials, selection_mae, test_mae)
    else:
        logger.info("Tune %d/%d done | test_mae=%.3f (%.1fs)", trial_idx, total_trials, test_mae, perf_counter() - trial_start)

    trial_seconds = perf_counter() - trial_start
    out = {
        "hparams": hparams,
        "test_loss": float(test_loss),
        "test_r2": float(test_r2) if np.isfinite(test_r2) else np.nan,
        "test_mae": float(test_mae),
        "selection_mae": selection_mae,
        "trial_seconds": float(trial_seconds),
    }
    if val_mae is not None:
        out["val_mae"] = float(val_mae)
        out["val_r2"] = float(val_r2) if np.isfinite(val_r2) else np.nan
    return out


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
    val_indices=None,
    lr_scheduler_type="none",
    reduce_lr_patience=3,
    reduce_lr_factor=0.5,
    min_lr=1e-6,
    grad_clip_norm=0.0,
    train_window_sample_weights=None,
    use_age_weighted_window_sampling=False,
):
    """Run full training loop with optional balanced sampling, validation, LR scheduler, and early stopping.

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

    scheduler = None
    if lr_scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr
        )
        logger.debug("LR scheduler: plateau patience=%d factor=%.2f", reduce_lr_patience, reduce_lr_factor)
    elif lr_scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        logger.debug("LR scheduler: cosine eta_min=%.0e", min_lr)

    use_val_for_monitor = val_indices is not None and val_indices.size > 0

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

        if (
            use_age_weighted_window_sampling
            and train_window_sample_weights is not None
            and train_window_sample_weights.shape[0] == train_indices.shape[0]
        ):
            epoch_train_indices = sample_weighted_train_epoch_indices(
                train_indices,
                train_window_sample_weights,
                num_samples=int(train_indices.shape[0]),
                rng=rng,
                replace=True,
            )
        elif subject_balanced_training and balanced_sorted_indices is not None:
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
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

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

            if points_per_batch > 0:
                sample_idx = np.arange(0, targets_np.shape[0], max(1, targets_np.shape[0] // points_per_batch))[:points_per_batch]
                epoch_targets_plot.append(targets_np[sample_idx])
                epoch_preds_plot.append(outputs_np[sample_idx])

            pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{metric_abs_error / max(1, metric_n):.4f}")

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

        val_loss = None
        if use_val_for_monitor:
            val_result = run_epoch_metrics(
                model,
                x_mem,
                y_mem,
                val_indices,
                batch_size,
                device,
                criterion,
                amp_enabled,
                subject_codes=None,
                n_subjects=0,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                normalize_target=normalize_target,
                plot_max_points=0,
                debug_chunk_log_every=debug_chunk_log_every,
                eval_label="Val Eval",
            )
            val_loss = val_result[0]
            val_mae = val_result[2]
            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_mae=%.2f",
                epoch + 1,
                epochs,
                avg_loss,
                val_loss,
                val_mae,
            )
        else:
            logger.info("Epoch %d/%d | loss=%.4f mae=%.2f", epoch + 1, epochs, avg_loss, mae)

        if len(epoch_targets_plot) > 0 and len(epoch_preds_plot) > 0:
            final_targets_plot = epoch_targets_plot
            final_preds_plot = epoch_preds_plot

        monitor_loss = val_loss if (use_val_for_monitor and val_loss is not None) else avg_loss
        improvement_threshold = max(
            early_stopping_min_delta_abs,
            abs(best_loss) * early_stopping_min_delta_rel if np.isfinite(best_loss) else 0.0,
        )
        improved = (best_loss - monitor_loss) > improvement_threshold

        if improved or not np.isfinite(best_loss):
            best_loss = monitor_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if lr_scheduler_type == "plateau":
                scheduler.step(monitor_loss)
            else:
                scheduler.step()

        if (
            early_stopping_enabled
            and (epoch + 1) >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            logger.info("Early stopping at epoch %d (best: epoch %d).", epoch + 1, best_epoch)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored best weights (epoch %d).", best_epoch)

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
