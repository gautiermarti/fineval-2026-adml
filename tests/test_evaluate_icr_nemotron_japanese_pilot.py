from scripts.evaluate_icr_nemotron_japanese_pilot import (
    EXPECTED_IDS,
    request_payload,
    score_records,
    validate_prediction,
)
from scripts.pilot_icr_judges import STRUCTURED_SCHEMA


def prediction(label):
    return {
        "proposition": "p",
        "responsiveness": "direct",
        "direction": "accept",
        "strength": "qualified",
        "conditionality": "ordinary",
        "decisive_span": "s",
        "counterevidence_span": "",
        "label": label,
        "confidence": 0.7,
        "rationale": "r",
    }


def test_payload_freezes_reasoning_sampling_and_schema():
    payload = request_payload("prompt")
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["seed"] == 2026
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["response_format"]["json_schema"]["schema"] == STRUCTURED_SCHEMA


def test_manual_schema_validation_rejects_missing_and_bad_label():
    assert validate_prediction(prediction("+1")) == (True, "valid")
    bad = prediction("+1")
    del bad["rationale"]
    assert not validate_prediction(bad)[0]
    bad = prediction("invalid")
    assert validate_prediction(bad) == (False, "bad_enum:label")
    bad = prediction("+1")
    bad["extra"] = "not allowed"
    assert validate_prediction(bad) == (False, "unexpected:extra")
    bad = prediction("+1")
    bad["confidence"] = float("nan")
    assert validate_prediction(bad) == (False, "non_finite:confidence")


def test_frozen_gate_passes_only_with_required_total_and_negative_recall():
    # Frozen gold distribution in ID order.
    gold = (
        ["-2"] * 2 + ["-1"] * 11 + ["0"] * 3 + ["+1"] * 3 + ["+2"] * 3
    )
    # The actual IDs are interleaved by label, so build a synthetic record cover
    # by assigning enough correct values after recovering each gold below.
    actual_gold = {
        25: "+1", 61: "+1", 86: "-1", 87: "-2", 90: "-1", 91: "-1",
        95: "-1", 97: "-1", 107: "-1", 108: "-1", 109: "-1", 110: "-1",
        117: "-1", 118: "-1", 126: "0", 129: "-2", 157: "+1",
        176: "+2", 189: "+2", 196: "0", 209: "+2", 243: "0",
    }
    assert sorted(actual_gold.values()) == sorted(gold)
    correct_ids = set(EXPECTED_IDS[:13])
    records = []
    for identifier in EXPECTED_IDS:
        target = actual_gold[identifier]
        guess = target if identifier in correct_ids else ("0" if target != "0" else "+1")
        records.append({
            "id": identifier,
            "gold": target,
            "prediction": prediction(guess),
            "valid": True,
        })
    metrics = score_records(records)
    assert metrics["correct"] == 13
    assert metrics["advance_gate"]["passed"]
    records[0]["valid"] = False
    assert not score_records(records)["advance_gate"]["passed"]
