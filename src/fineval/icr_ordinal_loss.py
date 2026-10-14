"""Distance-aware losses for ordered JF-ICR class distributions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ordered_cdf_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return normalized squared CDF distance to ordered one-hot targets.

    For K ordered classes there are K-1 meaningful cumulative thresholds.
    This is the discrete Cramér/EMD-squared form and remains differentiable
    through the ordinary K-way softmax head.
    """
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [batch, at least two classes]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("labels must contain integer class indices")
    if bool(torch.any(labels < 0)) or bool(torch.any(labels >= logits.shape[1])):
        raise ValueError("label index is outside the logit class range")
    probabilities = torch.softmax(logits, dim=-1)
    predicted_cdf = probabilities.cumsum(dim=-1)[:, :-1]
    thresholds = torch.arange(
        logits.shape[1] - 1, device=labels.device,
    ).unsqueeze(0)
    target_cdf = (labels.unsqueeze(1) <= thresholds).to(logits.dtype)
    return F.mse_loss(predicted_cdf, target_cdf)


def distance_aware_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, *, distance_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, ordinary CE, and normalized ordered-CDF components."""
    if distance_weight < 0:
        raise ValueError("distance_weight must be non-negative")
    cross_entropy = F.cross_entropy(logits, labels)
    cdf = ordered_cdf_loss(logits, labels)
    return cross_entropy + distance_weight * cdf, cross_entropy, cdf
