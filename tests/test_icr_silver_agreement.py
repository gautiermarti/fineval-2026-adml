from fineval.icr_silver import (
    LABELS,
    audit_teacher_independence,
    canonical_model_family,
    decide_agreement,
    decide_counterfactual_verification,
    decide_decisive_event_agreement,
    validate_counterfactual_lattice,
    validate_judgment,
)


RESPONSE = "現時点では実施しませんが、環境が整えば今後検討します。"


def judgment(label: str = "-1") -> dict:
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
        "label": label,
        "confidence": 0.91,
        "rationale": "現在は拒否するが再考余地がある。",
    }


def record(teacher: str, rubric: str, label: str = "-1") -> dict:
    return {"teacher_id": teacher, "rubric": rubric, "judgment": judgment(label)}


def test_judgment_requires_grounded_evidence_and_consistent_factors() -> None:
    valid, reason = validate_judgment(judgment(), RESPONSE)
    assert valid and reason == "valid"
    bad = judgment()
    bad["decisive_span"] = "回答に存在しない引用"
    valid, reason = validate_judgment(bad, RESPONSE)
    assert not valid and reason == "ungrounded_decisive_span"
    bad = judgment()
    bad["responsiveness"] = "invented"
    assert validate_judgment(bad, RESPONSE) == (
        False, "invalid_enum:responsiveness",
    )
    bad = judgment()
    bad["confidence"] = float("nan")
    assert validate_judgment(bad, RESPONSE) == (False, "invalid_confidence")
    bad = judgment()
    bad["extra"] = "not allowed"
    assert validate_judgment(bad, RESPONSE) == (
        False, "unexpected_fields:extra",
    )


def test_accepts_exact_two_teacher_two_rubric_agreement() -> None:
    rows = [
        record("qwen", "factor_first"),
        record("qwen", "boundary_contrast"),
        record("gemma", "factor_first"),
        record("gemma", "boundary_contrast"),
    ]
    result = decide_agreement(rows, RESPONSE)
    assert result.accepted
    assert result.label == "-1"
    assert result.judgment_count == 4


def test_rejects_between_teacher_disagreement() -> None:
    rows = [
        record("qwen", "factor_first"),
        record("qwen", "boundary_contrast"),
        record("gemma", "factor_first", "0"),
        record("gemma", "boundary_contrast", "0"),
    ]
    for row in rows[2:]:
        row["judgment"].update({
            "direction": "neutral",
            "strength": "hedged",
        })
    result = decide_agreement(rows, RESPONSE)
    assert not result.accepted
    assert result.reason == "between_teacher_disagreement"


def test_gold_anchored_event_requires_span_and_factor_agreement() -> None:
    rows = [
        record("qwen", "event_factors"),
        record("qwen", "ordinal_boundary"),
        record("gemma", "event_factors"),
        record("gemma", "ordinal_boundary"),
    ]
    result = decide_decisive_event_agreement(rows, RESPONSE, "-1")
    assert result.accepted
    assert result.reason == "exact_span_factor_agreement"
    assert result.decisive_span == "現時点では実施しません"
    rows[-1]["judgment"]["decisive_span"] = "今後検討します"
    result = decide_decisive_event_agreement(rows, RESPONSE, "-1")
    assert not result.accepted
    assert result.reason == "within_teacher_span_disagreement"


def test_gold_anchored_event_rejects_changed_fixed_label() -> None:
    rows = [
        record("qwen", "event_factors"),
        record("qwen", "ordinal_boundary"),
        record("gemma", "event_factors"),
        record("gemma", "ordinal_boundary"),
    ]
    rows[-1]["judgment"].update({
        "label": "0", "direction": "neutral", "strength": "hedged",
    })
    result = decide_decisive_event_agreement(rows, RESPONSE, "-1")
    assert not result.accepted
    assert result.reason == "insufficient_teacher_rubric_coverage"


def test_teacher_independence_requires_distinct_model_families() -> None:
    rows = [
        {"teacher_id": "teacher-a", "model": "Qwen/Qwen3.6-35B-A3B-FP8", "model_revision": "qwen-commit"},
        {"teacher_id": "teacher-a", "model": "Qwen/Qwen3.6-35B-A3B-FP8", "model_revision": "qwen-commit"},
        {"teacher_id": "teacher-b", "model": "gemma3:12b", "model_revision": "gemma-digest"},
        {"teacher_id": "teacher-b", "model": "gemma3:12b", "model_revision": "gemma-digest"},
    ]
    audit = audit_teacher_independence(rows)
    assert audit["passed"]
    assert audit["model_families"] == ["gemma", "qwen"]
    assert canonical_model_family("Qwen/Qwen3.6-35B-A3B-FP8") == "qwen"
    assert canonical_model_family("nvidia/NVIDIA-Nemotron-Nano-9B") == "nemotron"


def test_teacher_independence_rejects_same_model_aliases_and_family() -> None:
    same_model = [
        {"teacher_id": "alias-a", "model": "gemma3:12b", "model_revision": "same"},
        {"teacher_id": "alias-b", "model": "gemma3:12b", "model_revision": "same"},
    ]
    audit = audit_teacher_independence(same_model)
    assert not audit["passed"]
    assert audit["same_model_aliases"] == {
        "gemma3:12b": ["alias-a", "alias-b"],
    }

    same_family = [
        {"teacher_id": "qwen-a", "model": "Qwen/Qwen3.6-35B", "model_revision": "qwen36"},
        {"teacher_id": "qwen-b", "model": "Qwen/Qwen2.5-32B", "model_revision": "qwen25"},
    ]
    audit = audit_teacher_independence(same_family)
    assert not audit["passed"]
    assert audit["model_families"] == ["qwen"]


def test_teacher_independence_rejects_revision_drift() -> None:
    rows = [
        {"teacher_id": "qwen", "model": "Qwen/Qwen3.6-35B", "model_revision": "commit-a"},
        {"teacher_id": "qwen", "model": "Qwen/Qwen3.6-35B", "model_revision": "commit-b"},
        {"teacher_id": "gemma", "model": "gemma3:12b", "model_revision": "digest"},
    ]
    audit = audit_teacher_independence(rows)
    assert not audit["passed"]
    assert audit["teacher_revision_drift"] == {
        "qwen": ["commit-a", "commit-b"],
    }


def counterfactual_lattice(source_span: str = "今後検討します") -> dict:
    specifications = {
        "-2": ("今後も実施する予定はありません", "refuse", "strong", "none", "permanent"),
        "-1": ("現時点では実施しませんが、今後検討します", "refuse", "qualified", "conditional", "time_bounded"),
        "0": ("今後については引き続き検討中です", "neutral", "hedged", "unknown", "unspecified"),
        "+1": ("条件が整えば今後実施したいと考えています", "commit", "qualified", "conditional", "unspecified"),
        "+2": ("今後実施します", "commit", "strong", "none", "unspecified"),
    }
    return {
        "source_span": source_span,
        "variants": [
            {
                "label": label,
                "replacement_span": values[0],
                "future_action": "yes",
                "direction": values[1],
                "strength": values[2],
                "conditionality": values[3],
                "temporal_scope": values[4],
                "rationale": "test",
            }
            for label, values in specifications.items()
        ],
    }


def test_counterfactual_lattice_changes_only_grounded_modality_span() -> None:
    response = "売上目標は100億円です。今後検討します。"
    valid, reason, variants = validate_counterfactual_lattice(
        counterfactual_lattice(), response,
    )
    assert valid and reason == "valid"
    assert [row["target_label"] for row in variants] == list(LABELS)
    assert all("100億円" in row["response"] for row in variants)
    assert all(row["response"].startswith("売上目標は100億円です。") for row in variants)


def test_counterfactual_lattice_rejects_editing_financial_values() -> None:
    response = "ROE12%を達成します。"
    lattice = counterfactual_lattice("ROE12%を達成します")
    valid, reason, variants = validate_counterfactual_lattice(lattice, response)
    assert not valid
    assert reason == "source_span_contains_protected_token"
    assert variants == []


def test_counterfactual_verifier_must_be_independent_and_recover_target() -> None:
    rows = [
        record("qwen", "factor_first"),
        record("qwen", "boundary_contrast"),
        record("gemma", "factor_first"),
        record("gemma", "boundary_contrast"),
    ]
    accepted = decide_counterfactual_verification(rows, RESPONSE, "-1", "generator")
    assert accepted.accepted
    assert accepted.reason == "independent_target_recovery"
    wrong = decide_counterfactual_verification(rows, RESPONSE, "-2", "generator")
    assert not wrong.accepted
    assert wrong.reason == "verified_label_misses_target"
    self_verified = decide_counterfactual_verification(rows, RESPONSE, "-1", "qwen")
    assert not self_verified.accepted
    assert self_verified.reason == "generator_cannot_self_verify"
