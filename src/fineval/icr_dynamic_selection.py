"""Local-competence routing for frozen JF-ICR experts."""

from __future__ import annotations

from collections import Counter

import numpy as np


def source_balanced_weights(similarities: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Downweight repeated neighbours from one source while retaining similarity."""
    counts = Counter(sources.tolist())
    return np.asarray([
        max(float(similarity), 0.0) / counts[source]
        for similarity, source in zip(similarities, sources, strict=True)
    ])


def dynamic_select(
    similarities: np.ndarray,
    train_gold: np.ndarray,
    train_sources: np.ndarray,
    train_predictions: np.ndarray,
    target_predictions: np.ndarray,
    *,
    neighbours: int,
    margin: float,
    conditional_on_prediction: bool,
    prior_strength: float = 2.0,
) -> tuple[np.ndarray, list[dict]]:
    """Choose a frozen expert using only labelled neighbours from training.

    Expert column zero is the incumbent.  A challenger is used only when its
    smoothed local competence exceeds the incumbent by ``margin``.
    """
    if similarities.shape != (len(target_predictions), len(train_gold)):
        raise ValueError("similarity matrix shape mismatch")
    if train_predictions.shape[0] != len(train_gold):
        raise ValueError("training prediction row mismatch")
    if train_predictions.shape[1] != target_predictions.shape[1]:
        raise ValueError("expert column mismatch")
    if neighbours < 1:
        raise ValueError("neighbours must be positive")

    source_counts = Counter(train_sources.tolist())
    row_source_weights = np.asarray([1.0 / source_counts[x] for x in train_sources])
    global_competence = np.average(
        train_predictions == train_gold[:, None], axis=0,
        weights=row_source_weights,
    )
    selected = np.empty(len(target_predictions), dtype=train_predictions.dtype)
    audits = []
    for row_index, row_similarities in enumerate(similarities):
        order = np.argsort(-row_similarities, kind="stable")[:neighbours]
        positive = row_similarities[order] > 0
        order = order[positive]
        if len(order) == 0:
            selected[row_index] = target_predictions[row_index, 0]
            audits.append({
                "selected_expert": 0, "incumbent_competence": None,
                "selected_competence": None, "neighbours_used": 0,
            })
            continue
        weights = source_balanced_weights(
            row_similarities[order], train_sources[order],
        )
        competences = np.empty(train_predictions.shape[1], dtype=float)
        local_support = np.empty(train_predictions.shape[1], dtype=float)
        for expert in range(train_predictions.shape[1]):
            eligible = np.ones(len(order), dtype=bool)
            if conditional_on_prediction:
                eligible = (
                    train_predictions[order, expert]
                    == target_predictions[row_index, expert]
                )
            local_weight = float(weights[eligible].sum())
            local_correct = float(np.sum(
                weights[eligible]
                * (train_predictions[order[eligible], expert] == train_gold[order[eligible]])
            ))
            competences[expert] = (
                local_correct + prior_strength * global_competence[expert]
            ) / (local_weight + prior_strength)
            local_support[expert] = local_weight

        best_competence = float(np.max(competences))
        # Prefer the incumbent on exact ties, then the lowest stable expert id.
        best_expert = int(np.flatnonzero(competences == best_competence)[0])
        if competences[best_expert] < competences[0] + margin:
            best_expert = 0
        selected[row_index] = target_predictions[row_index, best_expert]
        audits.append({
            "selected_expert": best_expert,
            "incumbent_competence": float(competences[0]),
            "selected_competence": float(competences[best_expert]),
            "neighbours_used": int(len(order)),
            "selected_local_support": float(local_support[best_expert]),
        })
    return selected, audits
