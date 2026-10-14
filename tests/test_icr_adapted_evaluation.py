import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import source_bootstrap_delta


def test_source_bootstrap_is_deterministic_and_detects_strict_gain():
    gold = np.asarray(["+1", "+1", "0", "0"])
    incumbent = np.asarray(["0", "+1", "+1", "0"])
    candidate = gold.copy()
    sources = np.asarray(["a", "a", "b", "b"])
    first = source_bootstrap_delta(candidate, incumbent, gold, sources, draws=1000, seed=7)
    second = source_bootstrap_delta(candidate, incumbent, gold, sources, draws=1000, seed=7)
    assert first == second
    assert first["bootstrap_probability_positive"] == 1.0
    assert first["bootstrap_p05_delta"] > 0
