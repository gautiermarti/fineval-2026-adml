"""Consensus utilities for conservative Task 1 annotation operations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def operation_signature(operation: dict) -> tuple:
    """Return the semantic identity of an accepted annotation operation.

    Scores are deliberately excluded: the deployment views have different
    confidence scales, while candidate/add/remove semantics are comparable.
    """
    return (
        int(operation["row_index"]),
        tuple(operation["candidate"]),
        str(operation["operation"]),
        tuple(sorted(tuple(item) for item in operation.get("remove", []))),
    )


def majority_operations(
    views: Iterable[Iterable[dict]], minimum_support: int,
) -> list[dict]:
    """Keep operations accepted by at least ``minimum_support`` views."""
    votes: dict[tuple, set[int]] = defaultdict(set)
    representatives: dict[tuple, dict] = {}
    probabilities: dict[tuple, list[float]] = defaultdict(list)
    for view_index, operations in enumerate(views):
        seen = set()
        for operation in operations:
            signature = operation_signature(operation)
            if signature in seen:
                continue
            seen.add(signature)
            votes[signature].add(view_index)
            representatives.setdefault(signature, operation)
            probabilities[signature].append(float(operation.get("probability", 0.0)))

    selected = []
    for signature, supporting_views in votes.items():
        if len(supporting_views) < minimum_support:
            continue
        representative = representatives[signature]
        selected.append({
            "row_index": int(representative["row_index"]),
            "candidate": list(representative["candidate"]),
            "operation": str(representative["operation"]),
            "remove": [list(item) for item in signature[3]],
            "support": len(supporting_views),
            "supporting_views": sorted(supporting_views),
            "mean_probability": sum(probabilities[signature]) / len(probabilities[signature]),
        })
    return sorted(
        selected,
        key=lambda row: (
            int(row["row_index"]), -int(row["support"]),
            -float(row["mean_probability"]), tuple(row["candidate"]),
        ),
    )


def grouped_consensus_operations(
    views: Iterable[Iterable[dict]], groups: Iterable[str],
    minimum_support: int, minimum_groups: int,
) -> list[dict]:
    """Require both view-level support and independent proposer-group support.

    A group normally identifies a proposer seed.  This permits an operation to
    survive when exact candidate semantics agree across seeds even if one
    seed's mean and vote deployment views disagree with each other.
    """

    view_list = [list(view) for view in views]
    group_list = list(groups)
    if len(view_list) != len(group_list):
        raise ValueError("every operation view must have one group")
    selected = majority_operations(view_list, minimum_support=minimum_support)
    output = []
    for operation in selected:
        supporting_groups = sorted({
            group_list[index] for index in operation["supporting_views"]
        })
        if len(supporting_groups) < minimum_groups:
            continue
        output.append(operation | {
            "group_support": len(supporting_groups),
            "supporting_groups": supporting_groups,
        })
    return output


def apply_consensus_operations(
    baselines: Iterable[set[tuple[str, str]]], operations: Iterable[dict],
) -> list[set[tuple[str, str]]]:
    """Apply semantic operations deterministically to copies of baselines."""
    output = [set(items) for items in baselines]
    for operation in operations:
        row_index = int(operation["row_index"])
        output[row_index].difference_update(
            tuple(item) for item in operation.get("remove", [])
        )
        output[row_index].add(tuple(operation["candidate"]))
    return output
