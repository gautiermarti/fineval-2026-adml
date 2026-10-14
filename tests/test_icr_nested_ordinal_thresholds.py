import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_nested_ordinal_thresholds import decode, fit_thresholds


def test_dynamic_thresholds_separate_ordered_classes():
    scores = np.array([-.9, -.8, -.1, .0, .7, .8])
    gold = np.array([-1, -1, 0, 0, 1, 1])
    bounds = fit_thresholds(scores, gold)
    assert np.array_equal(decode(scores, bounds), gold)
