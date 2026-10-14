import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_task1_all_released_labels import (
    BASELINE_SCORE, attribution_gate, promotion_gate,
)


def test_all_released_label_gate_requires_multitype_safe_gain():
    baseline = {"person": 0.8, "organization": 0.8, "location": 0.8}
    pilot = {"person": 0.8, "organization": 0.82, "location": 0.795}
    gate = promotion_gate(BASELINE_SCORE + 0.01, baseline, pilot, 3, True)
    assert gate["passed"]


def test_all_released_label_gate_rejects_one_type_regression():
    baseline = {"person": 0.8, "organization": 0.8, "location": 0.8}
    pilot = {"person": 0.8, "organization": 0.84, "location": 0.78}
    gate = promotion_gate(BASELINE_SCORE + 0.01, baseline, pilot, 3, True)
    assert not gate["no_type_regresses_more_than_0.01"]
    assert not gate["passed"]


def test_all_released_label_gate_rejects_missing_provenance():
    baseline = {"person": 0.8, "organization": 0.8, "location": 0.8}
    pilot = {"person": 0.81, "organization": 0.82, "location": 0.8}
    gate = promotion_gate(BASELINE_SCORE + 0.01, baseline, pilot, 3, False)
    assert not gate["passed"]


def test_attribution_gate_requires_same_code_multitype_gain():
    control = {"person": 0.8, "organization": 0.8, "location": 0.8}
    pilot = {"person": 0.8, "organization": 0.82, "location": 0.795}
    assert attribution_gate(0.86, 0.85, pilot, control)["passed"]


def test_attribution_gate_rejects_macro_regression():
    control = {"person": 0.8, "organization": 0.8, "location": 0.8}
    pilot = {"person": 0.81, "organization": 0.82, "location": 0.8}
    assert not attribution_gate(0.84, 0.85, pilot, control)["passed"]
