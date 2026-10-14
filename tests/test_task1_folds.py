from fineval.task1_folds import (
    assign_entity_disjoint_folds,
    audit_entity_disjoint_folds,
    normalize_entity_surface,
)


def test_normalization_links_legal_suffix_and_accents():
    assert normalize_entity_surface("Όμιλος ACME Α.Ε.") == "acme"
    assert normalize_entity_surface("ΟΜΙΛΟΥ ACME AE") == "acme"


def test_entity_disjoint_assignment_is_deterministic_and_balanced():
    rows = [
        [("ACME A.E.", "ΟΡΓΑΝΙΣΜΟΣ")],
        [("ACME AE", "ΟΡΓΑΝΙΣΜΟΣ")],
        [("Jane Doe", "ΠΡΟΣΩΠΟ")],
        [("Jane Doe", "ΠΡΟΣΩΠΟ"), ("Athens", "ΤΟΠΟΘΕΣΙΑ")],
    ] + [[(f"ORG {index}", "ΟΡΓΑΝΙΣΜΟΣ")] for index in range(8)]
    first = assign_entity_disjoint_folds(rows, n_splits=3, seed=7)
    second = assign_entity_disjoint_folds(rows, n_splits=3, seed=7)
    assert first == second
    assert first[0] == first[1]
    assert first[2] == first[3]
    audit = audit_entity_disjoint_folds(rows, first)
    assert audit["leaking_entity_keys"] == []
    assert max(audit["rows_by_fold"].values()) - min(audit["rows_by_fold"].values()) <= 2


def test_generic_locations_do_not_force_same_fold():
    rows = [
        [("ONE", "ΟΡΓΑΝΙΣΜΟΣ"), ("Ελλάδα", "ΤΟΠΟΘΕΣΙΑ")],
        [("TWO", "ΟΡΓΑΝΙΣΜΟΣ"), ("Ελλάδα", "ΤΟΠΟΘΕΣΙΑ")],
        [("THREE", "ΟΡΓΑΝΙΣΜΟΣ")],
        [("FOUR", "ΟΡΓΑΝΙΣΜΟΣ")],
    ]
    assignments = assign_entity_disjoint_folds(rows, n_splits=2)
    audit = audit_entity_disjoint_folds(rows, assignments)
    assert audit["linked_entity_keys"] == 4
    assert audit["leaking_entity_keys"] == []
