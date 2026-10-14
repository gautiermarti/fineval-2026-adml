import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.icr_silver import RUBRIC_VERSION
from summarize_icr_decisive_event_pilot import summarize


RESPONSE = "現時点では実施しませんが、環境が整えば今後検討します。"


def judgment():
    return {
        "proposition": "施策を実施する",
        "future_action": "yes",
        "direction": "refuse",
        "strength": "qualified",
        "conditionality": "conditional",
        "temporal_scope": "time_bounded",
        "responsiveness": "direct",
        "decisive_span": "現時点では実施しません",
        "counterevidence_span": "環境が整えば今後検討します",
        "label": "-1",
        "confidence": 0.9,
        "rationale": "現在は拒否するが再考余地がある。",
    }


def records():
    return [
        {
            "id": 1,
            "gold_label": "-1",
            "teacher_id": "qwen",
            "model": "qwen",
            "model_revision": "commit",
            "rubric": rubric,
            "rubric_version": RUBRIC_VERSION,
            "judgment": judgment(),
            "valid": True,
            "validation_reason": "valid",
        }
        for rubric in ("event_factors", "ordinal_boundary")
    ]


def test_summary_recomputes_two_rubric_invariant_coverage():
    queue = {1: {"id": 1, "gold_label": "-1", "response": RESPONSE}}
    result = summarize(queue, records())
    teacher = result["teachers"]["qwen"]
    assert teacher["ids_with_two_valid_rubrics"] == [1]
    assert teacher["ids_with_invariant_span_factors"] == [1]
    assert result["pilot_futility"]["passed"]


def test_summary_marks_zero_two_rubric_coverage_as_futile():
    queue = {1: {"id": 1, "gold_label": "-1", "response": RESPONSE}}
    rows = records()
    rows[1]["judgment"]["direction"] = "neutral"
    rows[1]["valid"] = False
    rows[1]["validation_reason"] = "factor_label_inconsistency"
    result = summarize(queue, rows)
    assert not result["pilot_futility"]["passed"]
    assert result["teachers"]["qwen"]["ids_with_two_valid_rubrics"] == []
