"""Structured add/replace operations for nested financial NER candidates."""

from __future__ import annotations

from collections import defaultdict


def overlaps(left, right) -> bool:
    return left["start"] < right["end"] and left["end"] > right["start"]


def candidate_operations(candidate, incumbent_occurrences, incumbent_items):
    """Return distinct operations induced by one proposal.

    Same-type overlaps are boundary alternatives and therefore replaced.
    Cross-type spans are retained as legitimate nesting, except that a second
    explicit variant may remove contained one/two-character fragments.  The
    latter captures tokenizer fractures such as ``Ir``/``aq`` without deleting
    real nested locations.
    """
    item = (candidate["entity"], candidate["type"])
    if item in incumbent_items:
        return []
    overlapping = [row for row in incumbent_occurrences if overlaps(candidate, row)]
    same = {
        (row["entity"], row["type"])
        for row in overlapping if row["type"] == candidate["type"]
    }
    tiny_cross = {
        (row["entity"], row["type"])
        for row in overlapping
        if row["type"] != candidate["type"]
        and row["start"] >= candidate["start"]
        and row["end"] <= candidate["end"]
        and len(row.get("surface_text", row["entity"]).strip()) <= 2
    }
    if same:
        kind = "replace_same_type"
    elif overlapping:
        kind = "add_nested_cross_type"
    else:
        kind = "add_disjoint"

    common = {
        "row_index": int(candidate["row_index"]),
        "candidate": item,
        "candidate_start": int(candidate["start"]),
        "candidate_end": int(candidate["end"]),
        "candidate_score": float(candidate.get("score", candidate.get("gliner_score", 0.0))),
        "same_type_overlap_count": sum(
            row["type"] == candidate["type"] for row in overlapping
        ),
        "cross_type_overlap_count": sum(
            row["type"] != candidate["type"] for row in overlapping
        ),
        "candidate_contains_count": sum(
            row["start"] >= candidate["start"] and row["end"] <= candidate["end"]
            for row in overlapping
        ),
        "candidate_contained_by_count": sum(
            row["start"] <= candidate["start"] and row["end"] >= candidate["end"]
            for row in overlapping
        ),
    }
    operations = [common | {"operation": kind, "remove": sorted(same)}]
    if tiny_cross:
        operations.append(common | {
            "operation": kind + "_drop_tiny_cross",
            "remove": sorted(same | tiny_cross),
        })
    return operations


def apply_operation(items, operation):
    output = set(items)
    output.difference_update(map(tuple, operation["remove"]))
    output.add(tuple(operation["candidate"]))
    return output


def apply_ranked_operations(baselines, operations, probabilities, threshold):
    """Apply at most one operation per proposal with same-type conflict safety."""
    selected = [set(row) for row in baselines]
    accepted = []
    candidate_spans = defaultdict(list)
    seen_candidates = set()
    ranked = sorted(
        zip(probabilities, operations),
        key=lambda pair: (-float(pair[0]), int(pair[1]["row_index"])),
    )
    for probability, operation in ranked:
        if float(probability) < threshold:
            break
        row_index = int(operation["row_index"])
        candidate = tuple(operation["candidate"])
        proposal_key = (
            row_index, int(operation["candidate_start"]),
            int(operation["candidate_end"]), candidate,
        )
        if proposal_key in seen_candidates:
            continue
        kind = candidate[1]
        left, right = int(operation["candidate_start"]), int(operation["candidate_end"])
        if any(
            previous_kind == kind and previous_left < right and previous_right > left
            for previous_left, previous_right, previous_kind in candidate_spans[row_index]
        ):
            continue
        before = selected[row_index]
        after = apply_operation(before, operation)
        if after == before:
            continue
        selected[row_index] = after
        seen_candidates.add(proposal_key)
        candidate_spans[row_index].append((left, right, kind))
        accepted.append({
            "probability": float(probability),
            "row_index": row_index,
            "candidate": list(candidate),
            "operation": operation["operation"],
            "remove": operation["remove"],
        })
    return selected, accepted
