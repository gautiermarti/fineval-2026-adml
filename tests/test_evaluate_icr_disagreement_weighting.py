import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.evaluate_icr_disagreement_weighting import pilot_gate


def test_pilot_gate_requires_every_preregistered_condition():
    passing = {
        "weighted_seed7": {"correct": 34},
        "primary_replace_seed7": {"correct": 35, "rescues": 3, "harms": 2},
        "secondary_general2_plus_weighted": {"correct": 34},
    }
    assert pilot_gate(passing)["pilot_passed"] is True
    passing["primary_replace_seed7"]["harms"] = 3
    assert pilot_gate(passing)["pilot_passed"] is False

