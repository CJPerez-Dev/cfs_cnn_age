"""MIL-specific model components built from the existing EEG CNN backbone.

Step 1 of the MIL migration freezes the pretrained CNN feature extractor and
removes the original age-regression head. This module centralizes that logic
so later MIL heads can rely on a stable, reusable encoder interface.
"""

import copy
import logging

import torch
import torch.nn as nn

from cnn_age_project.models.cnn_model import EEGCNN

logger = logging.getLogger(__name__)


def summarize_parameter_counts(module: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts for a module.

    Args:
        module (torch.nn.Module): Module to summarize.

    Returns:
        tuple[int, int]: ``(total_params, trainable_params)``.
    """
    total_params = sum(parameter.numel() for parameter in module.parameters())
    trainable_params = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total_params, trainable_params


def freeze_module_parameters(module: nn.Module) -> None:
    """Freeze all parameters in ``module`` by disabling gradient updates.

    Args:
        module (torch.nn.Module): Module to freeze.

    Returns:
        None
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module_parameters(module: nn.Module) -> None:
    """Unfreeze all parameters in ``module`` by enabling gradient updates.

    Args:
        module (torch.nn.Module): Module to unfreeze.

    Returns:
        None
    """
    for parameter in module.parameters():
        parameter.requires_grad = True


class MILInstanceEncoder(nn.Module):
    """Frozen feature encoder for MIL built from ``EEGCNN.features``.

    This encoder keeps only the convolution/pooling path from the baseline CNN,
    discarding the original regression head. The output is a compact embedding
    per instance with shape ``(batch, 128)`` for the current architecture.
    """

    def __init__(self, feature_extractor: nn.Module, freeze_encoder: bool = True):
        """Create MIL instance encoder from an existing feature extractor.

        Args:
            feature_extractor (torch.nn.Module): CNN conv/pool stack.
            freeze_encoder (bool): If True, disables gradient updates.

        Returns:
            None
        """
        super().__init__()
        self.feature_extractor = feature_extractor

        if freeze_encoder:
            freeze_module_parameters(self.feature_extractor)

    @classmethod
    def from_eegcnn(cls, cnn_model: EEGCNN, freeze_encoder: bool = True) -> "MILInstanceEncoder":
        """Build a MIL encoder by cloning ``cnn_model.features``.

        Args:
            cnn_model (EEGCNN): Source CNN model.
            freeze_encoder (bool): If True, freeze encoder parameters.

        Returns:
            MILInstanceEncoder: MIL-ready instance encoder.
        """
        feature_extractor = copy.deepcopy(cnn_model.features)
        encoder = cls(feature_extractor=feature_extractor, freeze_encoder=freeze_encoder)
        total_params, trainable_params = summarize_parameter_counts(encoder)
        logger.info(
            "MIL encoder built from EEGCNN.features | total_params=%d | trainable_params=%d | frozen=%s",
            total_params,
            trainable_params,
            freeze_encoder,
        )
        return encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw EEG windows into fixed-length instance embeddings.

        Args:
            x (torch.Tensor): Input tensor shaped ``(batch, 1, window_len)``.

        Returns:
            torch.Tensor: Encoded features shaped ``(batch, feature_dim)``.
        """
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)


class GatedAttentionHead(nn.Module):
    """Gated attention module for adaptive window weighting in MIL.

    The head computes two parallel non-linear projections over instance
    embeddings: ``tanh`` to detect informative patterns and ``sigmoid`` to
    gate/suppress unreliable patterns. Their elementwise product is converted
    to per-window logits and normalized with a Softmax over bag instances.
    """

    def __init__(self, input_dim: int, attention_dim: int, dropout: float = 0.0):
        """Initialize gated attention projections.

        Args:
            input_dim (int): Instance embedding dimension.
            attention_dim (int): Hidden attention dimension.

        Returns:
            None
        """
        super().__init__()
        self.feature_projection = nn.Linear(input_dim, attention_dim)
        self.gate_projection = nn.Linear(input_dim, attention_dim)
        self.score_projection = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(
        self,
        instance_embeddings: torch.Tensor,
        bag_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bag embedding and normalized attention weights.

        Args:
            instance_embeddings (torch.Tensor): Tensor shaped
                ``(batch, n_instances, input_dim)``.
            bag_mask (torch.Tensor | None): Optional boolean mask shaped
                ``(batch, n_instances)`` where True means valid instance.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                ``(bag_embedding, attention_weights, attention_logits)`` where
                bag embedding is shaped ``(batch, input_dim)`` and weights/logits
                are shaped ``(batch, n_instances)``.
        """
        if instance_embeddings.ndim != 3:
            raise ValueError(
                "GatedAttentionHead expects shape (batch, n_instances, input_dim). "
                f"Received {tuple(instance_embeddings.shape)}."
            )

        tanh_features = torch.tanh(self.feature_projection(instance_embeddings))
        sigmoid_gate = torch.sigmoid(self.gate_projection(instance_embeddings))
        gated_features = tanh_features * sigmoid_gate

        gated_features = self.dropout(gated_features)
        attention_logits = self.score_projection(gated_features).squeeze(-1)

        if bag_mask is not None:
            if bag_mask.shape != attention_logits.shape:
                raise ValueError(
                    "bag_mask must match attention logit shape. "
                    f"Expected {tuple(attention_logits.shape)}, got {tuple(bag_mask.shape)}."
                )
            valid_counts = bag_mask.sum(dim=1)
            if torch.any(valid_counts == 0):
                raise ValueError("Each bag must contain at least one valid instance in bag_mask.")
            attention_logits = attention_logits.masked_fill(~bag_mask, float("-inf"))

        attention_weights = torch.softmax(attention_logits, dim=1)
        bag_embedding = torch.sum(attention_weights.unsqueeze(-1) * instance_embeddings, dim=1)
        return bag_embedding, attention_weights, attention_logits


class MILAgeRegressor(nn.Module):
    """MIL age regressor using a frozen instance encoder and gated attention.

    Input is a bag of EEG windows per subject. The model encodes each window,
    computes attention weights across windows, aggregates them into a bag
    embedding, and predicts one age per bag.
    """

    def __init__(
        self,
        instance_encoder: MILInstanceEncoder,
        feature_dim: int = 128,
        attention_dim: int = 128,
        regressor_hidden_dim: int = 64,
        pooling_type: str = "gated",
        attention_dropout: float = 0.0,
        regressor_dropout: float = 0.0,
    ):
        """Initialize MIL regressor modules.

        Args:
            instance_encoder (MILInstanceEncoder): Window-level encoder.
            feature_dim (int): Encoder output dimension.
            attention_dim (int): Hidden size inside gated attention.
            regressor_hidden_dim (int): Hidden size of bag-level regressor.
            regressor_dropout (float): Dropout probability in bag regressor after hidden ReLU.

        Returns:
            None
        """
        super().__init__()
        pooling = str(pooling_type).strip().lower()
        if pooling not in {"gated", "mean"}:
            raise ValueError("pooling_type must be either 'gated' or 'mean'.")

        self.instance_encoder = instance_encoder
        self.pooling_type = pooling
        self.attention_head = GatedAttentionHead(
            input_dim=feature_dim,
            attention_dim=attention_dim,
            dropout=float(attention_dropout),
        )
        rd = float(regressor_dropout)
        reg_drop = nn.Dropout(rd) if rd > 0.0 else nn.Identity()
        self.bag_regressor = nn.Sequential(
            nn.Linear(feature_dim, regressor_hidden_dim),
            nn.ReLU(),
            reg_drop,
            nn.Linear(regressor_hidden_dim, 1),
        )

        enc_total, enc_trainable = summarize_parameter_counts(self.instance_encoder)
        logger.info(
            "MILAgeRegressor initialized | feature_dim=%d attention_dim=%d "
            "regressor_hidden_dim=%d pooling_type=%s attention_dropout=%.3f regressor_dropout=%.3f "
            "encoder_total_params=%d encoder_trainable_params=%d",
            feature_dim,
            attention_dim,
            regressor_hidden_dim,
            self.pooling_type,
            float(attention_dropout),
            float(regressor_dropout),
            enc_total,
            enc_trainable,
        )

    def encode_bag_instances(self, bag_windows: torch.Tensor) -> torch.Tensor:
        """Encode all windows in a bag tensor into per-instance embeddings.

        Args:
            bag_windows (torch.Tensor): Tensor shaped
                ``(batch, n_instances, 1, window_len)``.

        Returns:
            torch.Tensor: Encoded embeddings shaped
                ``(batch, n_instances, feature_dim)``.
        """
        if bag_windows.ndim != 4:
            raise ValueError(
                "MILAgeRegressor expects bag windows with shape "
                f"(batch, n_instances, 1, window_len). Received {tuple(bag_windows.shape)}."
            )

        batch_size, n_instances, channels, window_len = bag_windows.shape
        flat_windows = bag_windows.reshape(batch_size * n_instances, channels, window_len)

        encoder_is_frozen = not any(p.requires_grad for p in self.instance_encoder.parameters())
        if encoder_is_frozen:
            with torch.no_grad():
                instance_embeddings = self.instance_encoder(flat_windows)
        else:
            instance_embeddings = self.instance_encoder(flat_windows)

        return instance_embeddings.reshape(batch_size, n_instances, -1)

    def forward(
        self,
        bag_windows: torch.Tensor,
        bag_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """Predict one age per bag and optionally return attention details.

        Args:
            bag_windows (torch.Tensor): Input shaped
                ``(batch, n_instances, 1, window_len)``.
            bag_mask (torch.Tensor | None): Optional boolean mask over valid
                windows shaped ``(batch, n_instances)``.
            return_attention (bool): If True, also return attention weights/logits.

        Returns:
            torch.Tensor | tuple: Bag-level predictions shaped ``(batch,)`` or
                ``(predictions, attention_weights, attention_logits, bag_embedding)``.
        """
        instance_embeddings = self.encode_bag_instances(bag_windows)

        if self.pooling_type == "mean":
            if bag_mask is None:
                bag_embedding = torch.mean(instance_embeddings, dim=1)
                attention_weights = torch.full(
                    (instance_embeddings.shape[0], instance_embeddings.shape[1]),
                    1.0 / max(1, instance_embeddings.shape[1]),
                    device=instance_embeddings.device,
                    dtype=instance_embeddings.dtype,
                )
            else:
                mask = bag_mask.to(instance_embeddings.dtype).unsqueeze(-1)
                valid = torch.clamp(mask.sum(dim=1), min=1.0)
                bag_embedding = (instance_embeddings * mask).sum(dim=1) / valid
                attention_weights = mask.squeeze(-1) / valid
            attention_logits = attention_weights
        else:
            bag_embedding, attention_weights, attention_logits = self.attention_head(
                instance_embeddings,
                bag_mask=bag_mask,
            )
        predictions = self.bag_regressor(bag_embedding).squeeze(-1)

        if return_attention:
            return predictions, attention_weights, attention_logits, bag_embedding
        return predictions

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Toggle whether the instance encoder participates in gradient updates.

        Args:
            trainable (bool): If True, unfreeze encoder; if False, freeze encoder.

        Returns:
            None
        """
        if trainable:
            unfreeze_module_parameters(self.instance_encoder)
        else:
            freeze_module_parameters(self.instance_encoder)

        enc_total, enc_trainable = summarize_parameter_counts(self.instance_encoder)
        logger.info(
            "MIL encoder trainability updated | trainable=%s | total_params=%d | trainable_params=%d",
            trainable,
            enc_total,
            enc_trainable,
        )
