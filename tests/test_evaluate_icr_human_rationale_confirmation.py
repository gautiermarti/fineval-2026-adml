import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_human_rationale_confirmation import initial_confirmation_gate


def test_initial_confirmation_gate_passes_noninferior_low_harm_candidate():
    candidate = {"accuracy": 0.62, "incumbent_accuracy": 0.60, "harms": 2}
    assert initial_confirmation_gate(candidate, 0.61)["passed"]


def test_initial_confirmation_gate_rejects_generic_seed_explanation():
    candidate = {"accuracy": 0.62, "incumbent_accuracy": 0.60, "harms": 1}
    assert not initial_confirmation_gate(candidate, 0.63)["passed"]


def test_initial_confirmation_gate_rejects_excess_harms():
    candidate = {"accuracy": 0.65, "incumbent_accuracy": 0.60, "harms": 3}
    assert not initial_confirmation_gate(candidate, 0.61)["passed"]
