import torch
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.icr_multitask import (
    normalized_span_bounds, ordinal_pairwise_loss, window_span_tokens,
)
from evaluate_icr_multitask_promise import pilot_decision


def test_ordered_logits_have_lower_pairwise_loss_than_reversed_logits():
    labels = torch.tensor([0, 2, 4])
    ordered = torch.tensor([
        [8.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 8.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 8.0],
    ])
    assert ordinal_pairwise_loss(ordered, labels) < ordinal_pairwise_loss(
        ordered.flip(0), labels,
    )


def test_pairwise_loss_is_differentiable_zero_without_unequal_pairs():
    logits = torch.randn(3, 5, requires_grad=True)
    loss = ordinal_pairwise_loss(logits, torch.tensor([2, 2, 2]))
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_span_window_preserves_full_span_and_balances_context():
    tokens, labels = window_span_tokens(
        [1, 2, 3, 4], [10, 11], [5, 6, 7, 8], budget=6,
    )
    assert tokens == [3, 4, 10, 11, 5, 6]
    assert labels == [0, 0, 1, 1, 0, 0]


def test_span_window_spends_unused_context_on_other_side():
    tokens, labels = window_span_tokens(
        [], [10, 11], [5, 6, 7, 8], budget=5,
    )
    assert tokens == [10, 11, 5, 6, 7]
    assert labels == [1, 1, 0, 0, 0]


def test_normalized_span_bounds_map_annotation_whitespace_to_source():
    text = "企画 Hondaは高い品質を実現する。 続き"
    span = "Honda は高い品質を実現する。"
    bounds = normalized_span_bounds(text, span)
    assert bounds is not None
    assert text[slice(*bounds)] == "Hondaは高い品質を実現する。"


def test_pilot_gate_rejects_accuracy_regression_even_with_rare_rescue():
    gold = np.array([2, 1, 1, 1])
    incumbent = np.array([1, 1, 1, 1])
    candidate = np.array([2, 0, 0, 1])
    result = pilot_decision(candidate, incumbent, gold)
    assert result["candidate_rare_boundary_recall"] == 1.0
    assert not result["survives_preregistered_futility_gate"]


def test_pilot_gate_accepts_noninferior_low_harm_candidate():
    gold = np.array([2, 1, 0, 1])
    incumbent = np.array([1, 1, 1, 1])
    candidate = np.array([2, 1, 1, 1])
    assert pilot_decision(candidate, incumbent, gold)[
        "survives_preregistered_futility_gate"
    ]
