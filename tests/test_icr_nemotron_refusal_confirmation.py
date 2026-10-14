from scripts.evaluate_icr_nemotron_japanese_refusal_confirmation import (
    grounded_refusal_override,
    normalized_text,
    promotion_gate,
)


def judgment(**updates):
    value = {
        "label": "-1",
        "direction": "reject",
        "confidence": 0.80,
        "decisive_span": "予定 は ない",
    }
    value.update(updates)
    return value


def test_grounded_refusal_accepts_nfkc_whitespace_match() -> None:
    result = grounded_refusal_override(
        "+1", judgment(), "現時点では予定はない。", valid=True,
    )
    assert result == ("-1", True, "grounded_refusal")
    assert normalized_text("Ａ B") == "AB"


def test_grounded_refusal_is_fail_closed() -> None:
    cases = [
        (judgment(label="0"), "not_refusal_label"),
        (judgment(direction="unclear"), "direction_not_reject"),
        (judgment(confidence=0.7999), "confidence_below_0.80"),
        (judgment(decisive_span="予"), "decisive_span_too_short"),
        (judgment(decisive_span="別の文"), "decisive_span_not_grounded"),
    ]
    for value, reason in cases:
        assert grounded_refusal_override(
            "+1", value, "現時点では予定はない。", valid=True,
        ) == ("+1", False, reason)
    assert grounded_refusal_override(
        "+1", judgment(), "現時点では予定はない。", valid=False,
    ) == ("+1", False, "invalid_judgment")


def test_promotion_gate_requires_every_frozen_condition() -> None:
    metrics = {
        "pilot_passed": True,
        "valid_judgments": 253,
        "incumbent_correct": 175,
        "candidate_correct": 179,
        "rescues": 6,
        "harms": 2,
        "changed_beneficial_fraction": 0.75,
        "refusal_correct": 7,
        "folds_improved": 3,
        "folds_worsened": 1,
        "bootstrap_probability_positive": 0.90,
        "bootstrap_p05_delta": 0.0,
    }
    assert promotion_gate(metrics)["passed"] is True
    metrics["harms"] = 3
    result = promotion_gate(metrics)
    assert result["at_most_two_harms"] is False
    assert result["passed"] is False
