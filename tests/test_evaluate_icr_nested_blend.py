import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_nested_blend import choose_weight


def test_choose_weight_uses_smallest_weight_on_tie(monkeypatch):
    monkeypatch.setattr(
        "evaluate_icr_nested_blend.apply_rules",
        lambda matrix, rules: np.where(matrix[:, 0] >= 0.5, "+1", "0"),
    )
    general = np.asarray([[0.8], [0.8]])
    adapted = np.asarray([[0.9], [0.9]])
    gold = np.asarray(["+1", "+1"])
    rules = np.asarray(["+1", "+1"])
    weight, accuracy = choose_weight(
        general, adapted, gold, rules, np.asarray([True, True]),
        np.asarray([0.0, 0.5, 1.0]),
    )
    assert weight == 0.0
    assert accuracy == 1.0
