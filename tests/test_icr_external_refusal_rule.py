from scripts.evaluate_icr_external_refusal_rule import refusal_rule, gate


def test_withholding_has_precedence_and_is_weak_refusal() -> None:
    label, family, span = refusal_rule(
        "計画はありませんが、具体的な数字への回答は差し控えます。"
    )
    assert (label, family) == ("-1", "withholding")
    assert "回答は差し控え" in span


def test_action_rejection_uses_frozen_temporal_mapping() -> None:
    assert refusal_rule("実施する予定はありません。")[0:2] == (
        "-2", "action_rejection",
    )
    assert refusal_rule("現時点では実施する予定はありません。")[0:2] == (
        "-1", "temporary_action_rejection",
    )
    assert refusal_rule("影響は現時点では限定的です。") == (None, None, None)


def test_rule_ignores_generic_undecided_and_disclosure_language() -> None:
    assert refusal_rule("具体的な人数は未定ですが、採用を強化します。") == (
        None, None, None,
    )
    assert refusal_rule("資料で開示している通り進捗しています。") == (
        None, None, None,
    )


def test_gate_is_conjunctive() -> None:
    metrics = {
        "incumbent_correct": 175,
        "candidate_correct": 179,
        "rescues": 5,
        "harms": 1,
        "changed_beneficial_fraction": 5 / 6,
        "refusal_correct": 5,
        "folds_improved": 3,
        "folds_worsened": 1,
        "bootstrap_probability_positive": 0.90,
        "bootstrap_p05_delta": 0.0,
    }
    assert gate(metrics)["passed"] is True
    metrics["refusal_correct"] = 3
    assert gate(metrics)["passed"] is False
