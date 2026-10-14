from fineval.task1_annotation_ops import (
    apply_operation,
    apply_ranked_operations,
    candidate_operations,
)


def occurrence(entity, kind, start, end):
    return {
        "entity": entity, "surface_text": entity, "type": kind,
        "start": start, "end": end,
    }


def test_full_organization_replaces_fragments_but_offers_tiny_cross_variant():
    candidate = occurrence("South Gas Company of Iraq", "ΟΡΓΑΝΙΣΜΟΣ", 0, 25) | {
        "row_index": 0, "score": 0.91,
    }
    incumbent = [
        occurrence("South Gas Company of", "ΟΡΓΑΝΙΣΜΟΣ", 0, 20),
        occurrence("Ir", "ΠΡΟΣΩΠΟ", 21, 23),
        occurrence("aq", "ΤΟΠΟΘΕΣΙΑ", 23, 25),
    ]
    items = {(row["entity"], row["type"]) for row in incumbent}
    operations = candidate_operations(candidate, incumbent, items)
    assert [row["operation"] for row in operations] == [
        "replace_same_type", "replace_same_type_drop_tiny_cross",
    ]
    repaired = apply_operation(items, operations[1])
    assert repaired == {("South Gas Company of Iraq", "ΟΡΓΑΝΙΣΜΟΣ")}


def test_real_nested_location_is_retained():
    candidate = occurrence("Bank of Greece", "ΟΡΓΑΝΙΣΜΟΣ", 0, 14) | {
        "row_index": 0, "score": 0.8,
    }
    incumbent = [occurrence("Greece", "ΤΟΠΟΘΕΣΙΑ", 8, 14)]
    items = {("Greece", "ΤΟΠΟΘΕΣΙΑ")}
    operations = candidate_operations(candidate, incumbent, items)
    assert len(operations) == 1
    assert apply_operation(items, operations[0]) == {
        ("Bank of Greece", "ΟΡΓΑΝΙΣΜΟΣ"), ("Greece", "ΤΟΠΟΘΕΣΙΑ"),
    }


def test_ranked_decoder_uses_one_variant_and_blocks_same_type_conflict():
    operations = [
        {
            "row_index": 0, "candidate": ("ABC Holdings", "ΟΡΓΑΝΙΣΜΟΣ"),
            "candidate_start": 0, "candidate_end": 12,
            "operation": "replace_same_type", "remove": [("ABC", "ΟΡΓΑΝΙΣΜΟΣ")],
        },
        {
            "row_index": 0, "candidate": ("ABC Holdings", "ΟΡΓΑΝΙΣΜΟΣ"),
            "candidate_start": 0, "candidate_end": 12,
            "operation": "replace_same_type_drop_tiny_cross",
            "remove": [("ABC", "ΟΡΓΑΝΙΣΜΟΣ"), ("H", "ΠΡΟΣΩΠΟ")],
        },
        {
            "row_index": 0, "candidate": ("ABC Hold", "ΟΡΓΑΝΙΣΜΟΣ"),
            "candidate_start": 0, "candidate_end": 8,
            "operation": "replace_same_type", "remove": [("ABC", "ΟΡΓΑΝΙΣΜΟΣ")],
        },
    ]
    selected, accepted = apply_ranked_operations(
        [{("ABC", "ΟΡΓΑΝΙΣΜΟΣ"), ("H", "ΠΡΟΣΩΠΟ")}],
        operations, [0.9, 0.8, 0.7], 0.5,
    )
    assert len(accepted) == 1
    assert selected[0] == {
        ("ABC Holdings", "ΟΡΓΑΝΙΣΜΟΣ"), ("H", "ΠΡΟΣΩΠΟ"),
    }
