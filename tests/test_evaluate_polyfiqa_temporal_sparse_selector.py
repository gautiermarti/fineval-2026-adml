from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_polyfiqa_temporal_sparse_selector import (  # noqa: E402
    apply_signature_gates, fit_signature_gates, promotion_decision,
)


def _row(company: str, delta: float) -> dict:
    return {
        "row_index": hash((company, delta)), "task_id": company,
        "internal_id": f"{company}_20200101", "tier": "easy",
        "family": "cash", "candidates": {"baseline": "b", "edit:x": "x"},
        "scores": {"baseline": 0.5, "edit:x": 0.5 + delta},
    }


def test_signature_gate_requires_repetition_in_every_training_company() -> None:
    rows = [
        _row(company, 0.01 + index * 1e-6)
        for company in ("HON", "JNJ", "MSFT") for index in range(3)
    ]
    gates = fit_signature_gates(rows)
    assert gates[("easy", "cash", "edit:x")]["admitted"]
    selected = apply_signature_gates([_row("UVV", 0.02)], gates)
    assert selected[0]["selected"] == "edit:x"
    assert not fit_signature_gates(rows[:-1])[("easy", "cash", "edit:x")]["admitted"]


def test_promotion_gate_rejects_too_few_changes_or_bootstrap_uncertainty() -> None:
    base = {
        "delta": 0.01, "changed_rows": 4, "improved_rows": 3,
        "harmed_rows": 1,
        "folds": [{"delta": 0.01}, {"delta": 0.01}, {"delta": 0.0}, {"delta": -0.001}],
        "company_cluster_bootstrap": {"p_gain": 0.95, "p05": 0.0},
    }
    assert promotion_decision(base)
    assert not promotion_decision({**base, "changed_rows": 3})
    assert not promotion_decision({
        **base, "company_cluster_bootstrap": {"p_gain": 0.89, "p05": 0.0},
    })
