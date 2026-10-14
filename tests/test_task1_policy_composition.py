from fineval.task1_policy_composition import compose_policy_deltas


def test_composition_preserves_independent_additions_and_replacements():
    base = {("A", "ORG"), ("B", "ORG")}
    operation = {("AB", "ORG")}
    structural = {("A", "ORG"), ("B", "ORG"), ("C", "LOC")}
    union = base | {("D", "PER")}
    assert compose_policy_deltas(base, [operation, structural, union]) == {
        ("AB", "ORG"), ("C", "LOC"), ("D", "PER"),
    }


def test_later_policy_can_remove_item_from_common_baseline():
    base = {("broken", "ORG"), ("keep", "ORG")}
    first = base | {("new", "ORG")}
    repair = {("keep", "ORG"), ("fixed", "ORG")}
    assert compose_policy_deltas(base, [first, repair]) == {
        ("keep", "ORG"), ("new", "ORG"), ("fixed", "ORG"),
    }
