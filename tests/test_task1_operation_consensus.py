from fineval.task1_operation_consensus import (
    apply_consensus_operations,
    grouped_consensus_operations,
    majority_operations,
)


def op(row, candidate, remove=(), probability=0.7):
    return {
        "row_index": row,
        "candidate": list(candidate),
        "operation": "replace_same_type" if remove else "add_disjoint",
        "remove": [list(item) for item in remove],
        "probability": probability,
    }


def test_majority_ignores_scores_and_rejects_single_view_operation():
    common_a = op(2, ("Alpha", "ORG"), probability=0.9)
    common_b = op(2, ("Alpha", "ORG"), probability=0.6)
    singleton = op(4, ("Beta", "LOC"), probability=0.99)
    selected = majority_operations([[common_a, singleton], [common_b], []], 2)
    assert len(selected) == 1
    assert selected[0]["candidate"] == ["Alpha", "ORG"]
    assert selected[0]["support"] == 2
    assert selected[0]["mean_probability"] == 0.75


def test_apply_consensus_removes_fragments_and_adds_whole_entity():
    baselines = [{("A", "ORG"), ("B", "ORG")}, set()]
    operations = [op(
        0, ("A B", "ORG"), remove=(("A", "ORG"), ("B", "ORG")),
    )]
    assert apply_consensus_operations(baselines, operations) == [
        {("A B", "ORG")}, set(),
    ]


def test_grouped_consensus_requires_independent_seed_support():
    cross_seed = op(2, ("Alpha", "ORG"))
    one_seed = op(4, ("Beta", "LOC"))
    views = [
        [cross_seed, one_seed], [cross_seed, one_seed], [one_seed], [cross_seed], [],
    ]
    selected = grouped_consensus_operations(
        views, groups=["seed7", "seed7", "seed7", "seed19", "seed19"],
        minimum_support=3, minimum_groups=2,
    )
    assert [row["candidate"] for row in selected] == [["Alpha", "ORG"]]
    assert selected[0]["supporting_groups"] == ["seed19", "seed7"]
