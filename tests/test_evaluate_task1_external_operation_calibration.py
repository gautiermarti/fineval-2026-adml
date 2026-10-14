from scripts.evaluate_task1_external_operation_calibration import (
    offset_operations,
    promotion_gate,
)


def test_offset_operations_does_not_mutate_source():
    source = [{"row_index": 2, "candidate": ["x", "ΟΡΓΑΝΙΣΜΟΣ"]}]
    shifted = offset_operations(source, 320)
    assert source[0]["row_index"] == 2
    assert shifted[0]["row_index"] == 322
    assert shifted[0]["candidate"] == source[0]["candidate"]


def test_promotion_gate_requires_every_safety_condition():
    metrics = {
        "delta": 0.006,
        "rows_improved": 2,
        "rows_harmed": 0,
        "type_deltas": {"ΠΡΟΣΩΠΟ": 0.0, "ΟΡΓΑΝΙΣΜΟΣ": 0.01, "ΤΟΠΟΘΕΣΙΑ": 0.0},
        "bootstrap_probability_gain": 0.95,
    }
    assert promotion_gate(metrics, True)["passed"]
    metrics["rows_harmed"] = 1
    assert not promotion_gate(metrics, True)["passed"]
    metrics["rows_harmed"] = 0
    assert not promotion_gate(metrics, False)["passed"]
