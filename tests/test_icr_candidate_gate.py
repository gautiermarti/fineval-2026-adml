import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import evaluate_profile, source_bootstrap_delta
from train_icr_japanese_cv import LABELS


def test_source_bootstrap_identical_profiles_have_zero_gain():
    gold = np.asarray(["+1", "0", "+1", "0"])
    prediction = np.asarray(["+1", "+1", "+1", "0"])
    sources = np.asarray(["a", "a", "b", "b"])

    result = source_bootstrap_delta(
        prediction, prediction, gold, sources, draws=1_000, seed=17,
    )

    assert result["bootstrap_p05_delta"] == 0.0
    assert result["bootstrap_median_delta"] == 0.0
    assert result["bootstrap_p95_delta"] == 0.0
    assert result["bootstrap_probability_positive"] == 0.0


def test_profile_clearing_every_source_and_fold_passes_gate():
    # Twenty rows per source/fold. The incumbent misses one row in every fold;
    # the candidate predicts +1 correctly everywhere.
    gold = np.asarray(["+1"] * 100)
    incumbent = np.asarray((["0"] + ["+1"] * 19) * 5)
    folds = np.repeat(np.arange(5), 20)
    sources = np.repeat(np.asarray(["a", "b", "c", "d", "e"]), 20)
    rules = np.asarray(["+1"] * 100)
    probabilities = np.zeros((100, len(LABELS)), dtype=float)
    probabilities[:, LABELS.index("+1")] = 1.0

    result, prediction = evaluate_profile(
        "synthetic", probabilities, gold, folds, sources, rules, incumbent,
        ordinal=True, draws=2_000,
    )

    assert np.array_equal(prediction, gold)
    assert result["all_accuracy"] == 1.0
    assert result["folds_improved"] == 5
    assert result["folds_worsened"] == 0
    assert result["bootstrap_probability_positive"] == 1.0
    assert np.isclose(result["bootstrap_p05_delta"], 0.05)
    assert result["promotion_eligible"] is True
