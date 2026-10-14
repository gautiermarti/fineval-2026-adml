import numpy as np

from fineval.icr_dynamic_selection import dynamic_select, source_balanced_weights


def test_source_balancing_splits_repeated_source_mass():
    values = source_balanced_weights(
        np.asarray([0.9, 0.7, 0.8]), np.asarray(["a", "a", "b"]),
    )
    assert np.allclose(values, [0.45, 0.35, 0.8])


def test_dynamic_selection_switches_only_above_margin():
    similarities = np.asarray([[1.0, 0.9, 0.8, 0.7]])
    gold = np.asarray([1, 1, 0, 0])
    sources = np.asarray(["a", "b", "c", "d"])
    train_predictions = np.asarray([
        [0, 1], [0, 1], [1, 0], [1, 0],
    ])
    target_predictions = np.asarray([[0, 1]])
    selected, audit = dynamic_select(
        similarities, gold, sources, train_predictions, target_predictions,
        neighbours=4, margin=0.05, conditional_on_prediction=False,
        prior_strength=0.0,
    )
    assert selected.tolist() == [1]
    assert audit[0]["selected_expert"] == 1

    conservative, _ = dynamic_select(
        similarities, gold, sources, train_predictions, target_predictions,
        neighbours=4, margin=1.01, conditional_on_prediction=False,
        prior_strength=0.0,
    )
    assert conservative.tolist() == [0]
