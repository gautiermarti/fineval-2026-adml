import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_human_rationale import pilot_gate


def test_pilot_requires_control_after_first_stage_passes():
    replacement = {"accuracy": 35 / 56, "harms": 2}
    gate = pilot_gate(replacement, 35 / 56, 34 / 56, None)
    assert gate["first_stage_passed"]
    assert gate["same_code_control_required"]
    assert gate["survives_preregistered_futility_gate"] is False


def test_pilot_passes_noninferior_control():
    replacement = {"accuracy": 34 / 56, "harms": 2}
    gate = pilot_gate(replacement, 34 / 56, 34 / 56, 33 / 56)
    assert gate["survives_preregistered_futility_gate"]


def test_pilot_rejects_extra_harms_or_worse_standalone():
    too_many_harms = pilot_gate(
        {"accuracy": 35 / 56, "harms": 3}, 35 / 56, 34 / 56, 34 / 56,
    )
    assert not too_many_harms["first_stage_passed"]
    weak_standalone = pilot_gate(
        {"accuracy": 34 / 56, "harms": 1}, 33 / 56, 34 / 56, 32 / 56,
    )
    assert not weak_standalone["first_stage_passed"]
