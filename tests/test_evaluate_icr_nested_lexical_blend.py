import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_nested_lexical_blend import choose_weight


def test_choose_weight_ties_toward_neural_incumbent():
    neural = np.asarray([[0, 0, 0, 1, 0], [0, 0, 1, 0, 0]], dtype=float)
    lexical = neural.copy()
    gold = np.asarray(["+1", "0"])
    rules = np.asarray(["+1", "+1"])
    weight, accuracy = choose_weight(
        neural, lexical, gold, rules, np.asarray([True, True]), np.asarray([0.0, 0.5, 1.0]),
    )
    assert accuracy == 1.0
    assert weight == 0.0
