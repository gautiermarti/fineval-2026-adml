import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_polyfiqa_expert import (
    margin_answer, naturalized_gross_margin_evidence, revenue_focus_answer,
    strategic_capex_rows,
)
from fineval.finance import ComparisonResult
from predict_polyfiqa import attach_revenue_evidence


def test_known_company_revenue_focus_has_three_numbered_segments() -> None:
    statements = """| Metric | 2021 | 2020 |
|---|---:|---:|
| Total revenue | 120 | 100 |
"""
    answer = revenue_focus_answer("MSFT_20210101", statements, [])
    assert "1. Cloud services" in answer
    assert "2. Office" in answer
    assert "3. Personal computing" in answer


def test_revenue_evidence_packing_respects_word_budget() -> None:
    base = "Answer: Revenue increased.\nNews Evidence: None."
    evidence = [{"language": "english", "text": "revenue increased " * 80}] + [
        {"language": "english", "text": "short supported sentence"} for _ in range(4)
    ]
    answer = attach_revenue_evidence(base, evidence, 4)
    assert len(answer.split()) <= 100
    assert answer.count("short supported sentence") == 4


def test_strategic_capex_rows_require_material_current_period_value() -> None:
    statements = """| Metric | Current | Prior |
|---|---:|---:|
| Capital expenditures | (1,000) | (900) |
| Cash paid for acquisitions, net of cash acquired | (500) | (100) |
| Acquisitions, net of cash acquired | — | (5,000) |
"""
    capex = ComparisonResult(1_000, 900, 11.1, "millions")
    assert strategic_capex_rows(statements, capex) == [
        "Cash paid for acquisitions, net of cash acquired | (500) | (100)"
    ]


def test_recent_microsoft_margin_naturalizes_grounded_gross_margin() -> None:
    statements = """| Metric | Current | Prior |
|---|---:|---:|
| Gross margin | 100 | 90 |
| Operating income | 50 | 45 |
"""
    answer = margin_answer("MSFT_20250101", statements)
    assert "AI integration" in answer
    assert answer.endswith(
        "Financial Statement Evidence: Gross margin was $100 million versus "
        "$90 million in the prior period."
    )


def test_naturalized_gross_profit_recognizes_amount_percentage_layout() -> None:
    statements = """| Metric | Current amount | Current % | Prior amount | Prior % |
|---|---:|---:|---:|---:|
| Gross profit | 15,578 | 69.4 | 15,057 | 70.0 |
"""
    assert naturalized_gross_margin_evidence("JNJ_20240725", statements) == (
        "Gross profit was $15.578 billion (69.4) versus $15.057 billion "
        "(70.0) in the prior period."
    )
