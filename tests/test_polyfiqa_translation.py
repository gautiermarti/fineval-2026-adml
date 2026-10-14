from fineval.polyfiqa_translation import (
    extract_translatable_evidence,
    mask_immutables,
    replace_evidence_translations,
    validate_translation,
    verification_is_acceptable,
)


def test_mask_restore_rejects_changed_or_new_values() -> None:
    source = "売上高は350億ドルでAzureが59%増加した。"
    masked = mask_immutables(source)
    assert "350" not in masked.masked and "59" not in masked.masked
    result = {
        "translation": masked.masked.replace("売上高は", "Revenue was ").replace(
            "で", ", and "
        ).replace("が", " ").replace("増加した。", " increased."),
        "source_claim_count": 2,
        "translation_claim_count": 2,
        "notes": "",
    }
    valid, reason, restored = validate_translation(result, masked)
    assert valid and reason == "valid"
    assert restored is not None and "350億ドル" in restored and "59%" in restored
    result["translation"] += " 99"
    valid, reason, _ = validate_translation(result, masked)
    assert not valid and reason == "unprotected_digit_introduced"


def test_extract_and_replace_translation_respects_offsets_and_budget() -> None:
    answer = 'Answer: Revenue rose.\nNews Evidence: Japanese news: "売上高は増加した。"'
    segments = extract_translatable_evidence(answer)
    assert len(segments) == 1 and segments[0]["language"] == "japanese"
    record = segments[0] | {"translation": "Revenue increased."}
    translated, reason = replace_evidence_translations(answer, [record])
    assert reason == "translated"
    assert translated is not None and "verified English translation" in translated
    assert "Revenue increased." in translated


def test_semantic_verifier_requires_grounded_quotes_and_all_checks() -> None:
    source = "売上高は増加した。"
    translation = "Revenue increased."
    verdict = {
        "faithful": True,
        "all_material_claims_preserved": True,
        "no_unsupported_claims": True,
        "polarity_preserved": True,
        "temporal_scope_preserved": True,
        "source_support_quote": "売上高は増加した",
        "translation_support_quote": "Revenue increased",
        "confidence": 0.93,
    }
    valid, reason = verification_is_acceptable(verdict, source, translation)
    assert valid and reason == "verified"
    verdict["polarity_preserved"] = False
    valid, reason = verification_is_acceptable(verdict, source, translation)
    assert not valid and reason == "semantic_verifier_rejected"

