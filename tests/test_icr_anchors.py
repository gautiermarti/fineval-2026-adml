import numpy as np

from fineval.icr_anchors import (
    balanced_boundary_pos_weights, boundary_view_text, cumulative_targets,
    decode_cumulative_probabilities,
    project_cumulative_probabilities,
    undo_positive_weighted_bce_logits,
)


def test_cumulative_projection_is_bounded_and_nonincreasing():
    projected = project_cumulative_probabilities([0.2, 0.9, -1.0, 2.0])
    assert np.all((0 <= projected) & (projected <= 1))
    assert np.all(projected[:-1] >= projected[1:])


def test_cumulative_decode_uses_expected_ordinal_rank():
    predictions = decode_cumulative_probabilities(np.asarray([
        [0.99, 0.95, 0.90, 0.80],
        [0.80, 0.60, 0.20, 0.05],
        [0.10, 0.05, 0.01, 0.00],
    ]))
    assert predictions.tolist() == ["+2", "0", "-2"]


def test_independent_boundary_targets_and_balancing():
    targets = cumulative_targets(np.asarray([0, 1, 2, 3, 4]))
    assert targets.tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]
    weights = balanced_boundary_pos_weights(targets, np.ones(5))
    assert np.allclose(weights, [1 / 4, 2 / 3, 1.5, 4.0])


def test_boundary_view_does_not_serialize_anchor_gold_labels():
    record = {
        "question": "投資しますか。",
        "response": "検討します。",
        "top_event_candidates": ["検討します"],
    }
    boundary = {
        "meaning": "forward action",
        "below_anchor": {
            "label": "-1", "question": "下", "response": "回答しない",
            "top_events": ["回答しない"],
        },
        "above_anchor": {
            "label": "+2", "question": "上", "response": "実施する",
            "top_events": ["実施する"],
        },
    }
    text = boundary_view_text(record, boundary, include_anchors=True)
    assert "-1" not in text
    assert "+2" not in text
    assert "[BELOW RESPONSE]" in text
    assert "[ABOVE RESPONSE]" in text


def test_positive_weight_logit_inverse_is_analytic():
    true_logits = np.asarray([[0.2, -0.5, 1.0, -1.5]])
    weights = np.asarray([0.1, 0.5, 2.0, 10.0])
    cost_sensitive = true_logits + np.log(weights)[None, :]
    recovered = undo_positive_weighted_bce_logits(cost_sensitive, weights)
    np.testing.assert_allclose(recovered, true_logits)
