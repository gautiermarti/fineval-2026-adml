import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_task1_public_label_ensemble import BASELINE_SCORE, ensemble_gate


def test_external_member_ensemble_gate_accepts_supported_safe_gain():
    baseline = {"person": 0.8, "organization": 0.8, "location": 0.8}
    candidate = {"person": 0.81, "organization": 0.82, "location": 0.795}
    gate = ensemble_gate(
        BASELINE_SCORE + 0.01, baseline, candidate, 3, 0.81, True,
    )
    assert gate["passed"]


def test_external_member_ensemble_gate_rejects_single_row_gain():
    baseline = {"person": 0.8, "organization": 0.8, "location": 0.8}
    candidate = {"person": 0.81, "organization": 0.82, "location": 0.8}
    gate = ensemble_gate(
        BASELINE_SCORE + 0.01, baseline, candidate, 1, 0.9, True,
    )
    assert not gate["passed"]


def test_external_member_ensemble_gate_rejects_missing_provenance():
    baseline = {"person": 0.8, "organization": 0.8, "location": 0.8}
    candidate = {"person": 0.81, "organization": 0.82, "location": 0.8}
    gate = ensemble_gate(
        BASELINE_SCORE + 0.01, baseline, candidate, 3, 0.9, False,
    )
    assert not gate["passed"]
