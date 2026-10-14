from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_transductive_label_shift import (  # noqa: E402
    FIVE_WAY,
    THREE_BIN,
    aggregate_probabilities,
    estimate_em_prior,
    estimate_soft_bbse_prior,
    reweight_probabilities,
)
from train_icr_japanese_cv import LABELS  # noqa: E402


def test_three_bin_aggregation_preserves_probability_mass():
    probabilities = np.asarray([
        [0.1, 0.2, 0.3, 0.15, 0.25],
        [0.0, 0.1, 0.2, 0.3, 0.4],
    ])
    aggregated = aggregate_probabilities(probabilities, THREE_BIN)
    assert np.allclose(aggregated, [[0.3, 0.3, 0.4], [0.1, 0.2, 0.7]])
    assert np.allclose(aggregated.sum(axis=1), 1.0)


def test_soft_bbse_recovers_prior_with_perfect_soft_confusion():
    source_probabilities = np.eye(5)
    source_labels = np.asarray(LABELS)
    target_counts = np.asarray([1, 1, 2, 3, 3])
    target_probabilities = np.repeat(np.eye(5), target_counts, axis=0)
    source_prior, target_prior, confusion = estimate_soft_bbse_prior(
        source_probabilities, source_labels, target_probabilities,
        FIVE_WAY, ridge=0.0,
    )
    assert np.allclose(source_prior, np.full(5, 0.2))
    assert np.allclose(confusion, np.eye(5))
    assert np.allclose(target_prior, target_counts / target_counts.sum(), atol=1e-6)


def test_reweighting_is_finite_normalized_and_moves_mass():
    probabilities = np.asarray([[0.1, 0.2, 0.3, 0.2, 0.2]])
    source_prior = np.asarray([0.3, 0.3, 0.4])
    target_prior = np.asarray([0.1, 0.2, 0.7])
    adjusted = reweight_probabilities(
        probabilities, source_prior, target_prior, THREE_BIN,
    )
    before = aggregate_probabilities(probabilities, THREE_BIN)
    after = aggregate_probabilities(adjusted, THREE_BIN)
    assert np.all(np.isfinite(adjusted))
    assert np.allclose(adjusted.sum(axis=1), 1.0)
    assert after[0, 2] > before[0, 2]
    assert after[0, 0] < before[0, 0]


def test_shrunk_em_stays_at_source_prior_when_target_matches_source():
    source_probabilities = np.eye(5)
    source_labels = np.asarray(LABELS)
    target_probabilities = np.eye(5)
    source_prior, target_prior = estimate_em_prior(
        source_probabilities, source_labels, target_probabilities,
        FIVE_WAY, shrink=0.5,
    )
    assert np.allclose(target_prior, source_prior)
