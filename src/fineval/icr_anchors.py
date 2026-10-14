"""Anchor-conditioned ordinal utilities for Japanese ICR."""

from __future__ import annotations

import numpy as np


LABELS = ("-2", "-1", "0", "+1", "+2")


def cumulative_targets(label_indices: np.ndarray) -> np.ndarray:
    """Return four independent targets indicating whether y exceeds k."""
    indices = np.asarray(label_indices, dtype=int)
    if indices.ndim != 1 or np.any((indices < 0) | (indices > 4)):
        raise ValueError("label indices must be a one-dimensional array in [0, 4]")
    return (indices[:, None] > np.arange(4)[None, :]).astype(float)


def balanced_boundary_pos_weights(
    targets: np.ndarray, sample_weights: np.ndarray,
) -> np.ndarray:
    """Compute source-weighted neg/pos ratios for independent BCE heads."""
    matrix = np.asarray(targets, dtype=float)
    weights = np.asarray(sample_weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or weights.shape != (len(matrix),):
        raise ValueError("targets must be (n, 4) and sample weights must be (n,)")
    positive = (matrix * weights[:, None]).sum(axis=0)
    negative = ((1.0 - matrix) * weights[:, None]).sum(axis=0)
    if np.any(positive <= 0) or np.any(negative <= 0):
        raise ValueError("every boundary must contain positive and negative examples")
    return negative / positive


def undo_positive_weighted_bce_logits(
    logits: np.ndarray, positive_weights: np.ndarray,
) -> np.ndarray:
    """Recover prevalence-scale logits from cost-sensitive BCE logits.

    For a positive BCE multiplier ``w``, the population optimum is shifted by
    ``log(w)``. Subtracting that fixed offset is the analytic inverse; it does
    not fit a threshold on evaluation labels.
    """
    matrix = np.asarray(logits, dtype=float)
    weights = np.asarray(positive_weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("expected logits with shape (n, 4)")
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("positive_weights must be four finite positive values")
    return matrix - np.log(weights)[None, :]


def _pava_increasing(values: np.ndarray) -> np.ndarray:
    """Least-squares projection of one vector onto nondecreasing values."""
    means: list[float] = []
    weights: list[int] = []
    for raw in values:
        means.append(float(raw))
        weights.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            mean = (
                means[-2] * weights[-2] + means[-1] * weights[-1]
            ) / weight
            means[-2:] = [mean]
            weights[-2:] = [weight]
    return np.asarray([
        mean for mean, weight in zip(means, weights, strict=True)
        for _ in range(weight)
    ])


def project_cumulative_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Project P(y > k) onto a valid nonincreasing cumulative sequence."""
    matrix = np.asarray(probabilities, dtype=float)
    squeeze = matrix.ndim == 1
    if squeeze:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("expected cumulative probabilities with shape (n, 4)")
    if not np.isfinite(matrix).all():
        raise ValueError("cumulative probabilities must be finite")
    result = np.stack([
        -_pava_increasing(-np.clip(row, 0.0, 1.0)) for row in matrix
    ])
    return result[0] if squeeze else result


def decode_cumulative_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Decode projected cumulative probabilities by rounded expected rank."""
    projected = project_cumulative_probabilities(probabilities)
    matrix = projected[None, :] if projected.ndim == 1 else projected
    indices = np.clip(np.rint(matrix.sum(axis=1)), 0, 4).astype(int)
    return np.asarray([LABELS[index] for index in indices])


def boundary_view_text(record: dict, boundary: dict, include_anchors: bool) -> str:
    """Build a label-free target or target-plus-anchors comparison view."""
    fields = [
        "[BOUNDARY] " + str(boundary["meaning"]),
        "[TARGET QUESTION] " + str(record["question"]),
        "[TARGET RESPONSE] " + str(record["response"]),
    ]
    events = record.get("top_event_candidates", [])
    if events:
        fields.append("[TARGET EVENTS] " + " [EVENT] ".join(map(str, events)))
    if include_anchors:
        for role, key in (("BELOW", "below_anchor"), ("ABOVE", "above_anchor")):
            anchor = boundary[key]
            fields.extend((
                f"[{role} QUESTION] " + str(anchor["question"]),
                f"[{role} RESPONSE] " + str(anchor["response"]),
            ))
            if anchor.get("top_events"):
                fields.append(
                    f"[{role} EVENTS] "
                    + " [EVENT] ".join(map(str, anchor["top_events"]))
                )
    return " ".join(fields)
