"""Deterministic composition of independently gated Task 1 policies."""

from __future__ import annotations


Entity = tuple[str, str]


def apply_policy_delta(
    baseline: set[Entity], current: set[Entity], policy: set[Entity],
) -> set[Entity]:
    """Apply a policy's removals/additions relative to the common incumbent."""
    removed = baseline - policy
    added = policy - baseline
    return (set(current) - removed) | added


def compose_policy_deltas(
    baseline: set[Entity], policies: list[set[Entity]],
) -> set[Entity]:
    """Apply independently defined deltas in an explicit stable order."""
    current = set(baseline)
    for policy in policies:
        current = apply_policy_delta(baseline, current, policy)
    return current
