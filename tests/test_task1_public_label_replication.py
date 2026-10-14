import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_task1_public_label_replication import BASELINE_SCORE, replication_gate


def test_replication_gate_accepts_multitype_supported_gain():
    baseline = {"person": 0.8, "organization": 0.8, "location": 0.8}
    pilot = {"person": 0.81, "organization": 0.82, "location": 0.795}
    gate = replication_gate(
        BASELINE_SCORE + 0.01, baseline, pilot, 3, 0.81, True,
    )
    assert gate["passed"]


def test_replication_gate_rejects_weak_bootstrap_support():
    baseline = {"person": 0.8, "organization": 0.8, "location": 0.8}
    pilot = {"person": 0.81, "organization": 0.82, "location": 0.8}
    gate = replication_gate(
        BASELINE_SCORE + 0.01, baseline, pilot, 3, 0.7999, True,
    )
    assert not gate["passed"]


def test_replication_gate_rejects_large_type_regression():
    baseline = {"person": 0.8, "organization": 0.8, "location": 0.8}
    pilot = {"person": 0.81, "organization": 0.82, "location": 0.78}
    gate = replication_gate(
        BASELINE_SCORE + 0.01, baseline, pilot, 3, 0.9, True,
    )
    assert not gate["passed"]
