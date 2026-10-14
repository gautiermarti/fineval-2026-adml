from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_polyfiqa_cash_evidence_recall_confirmation import (  # noqa: E402
    choose, promotion_decision,
)


def test_choose_requires_exact_recall_operation() -> None:
    assert choose({"baseline": "b", "evidence:extractive": "x"}) == "baseline"
    assert choose({"baseline": "b", "evidence:recall": "r"}) == "evidence:recall"


def test_promotion_requires_nonharmful_confirmation_and_all_companies() -> None:
    discovery = {
        "by_company": {
            company: {"eligible_edits": 1, "delta": 0.01}
            for company in ("HON", "JNJ", "MSFT", "UVV")
        }
    }
    confirmation = {
        "eligible_edits": 2, "harmed_edits": 0,
        "improved_edits": 1, "eligible_mean_delta": 0.01,
    }
    assert promotion_decision(discovery, confirmation, {"delta": 0.01})
    assert not promotion_decision(
        discovery, {**confirmation, "harmed_edits": 1}, {"delta": 0.01},
    )
    missing = {"by_company": dict(discovery["by_company"])}
    missing["by_company"].pop("UVV")
    assert not promotion_decision(missing, confirmation, {"delta": 0.01})
