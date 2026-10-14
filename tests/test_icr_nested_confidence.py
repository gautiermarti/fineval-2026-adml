import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_nested_confidence import confidence_values, gated_prediction


def test_confidence_gate_only_falls_back_on_unruled_nonmajority_calls():
    probabilities = np.asarray([
        [0.05, 0.05, 0.40, 0.30, 0.20],
        [0.05, 0.05, 0.40, 0.30, 0.20],
    ])
    rules = np.asarray(["+1", "-2"])
    confidence = confidence_values(probabilities, "max_probability")
    assert gated_prediction(probabilities, rules, confidence, 0.5).tolist() == ["+1", "-2"]


def test_top2_margin_confidence():
    probabilities = np.asarray([[0.05, 0.10, 0.15, 0.50, 0.20]])
    assert np.allclose(confidence_values(probabilities, "top2_margin"), [0.30])
