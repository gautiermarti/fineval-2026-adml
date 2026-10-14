#!/usr/bin/env python3
"""Paired comparison of input-only planner predictions on common records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def load(path: Path) -> dict[tuple, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        (
            row["held_company"], row["task_id"], row["tier"], row["family"],
        ): row
        for row in rows
    }


def as_set(items) -> set[tuple]:
    return {tuple(item) for item in items}


def set_f1(target, prediction) -> float:
    target, prediction = as_set(target), as_set(prediction)
    if not target and not prediction:
        return 1.0
    if not target or not prediction:
        return 0.0
    return 2 * len(target & prediction) / (len(target) + len(prediction))


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[int((len(values) - 1) * quantile)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    baseline, candidate = load(args.baseline), load(args.candidate)
    common = sorted(set(baseline) & set(candidate))
    if not common:
        raise ValueError("planner runs contain no common evaluation records")
    methods = sorted(
        {"threshold", "two_stage", "topk", "nested"}
        & set.intersection(
            *(set(row) for row in [baseline[common[0]], candidate[common[0]]])
        )
    )
    rng = random.Random(7)
    comparisons = {}
    for method in methods:
        old = [set_f1(baseline[key]["target"], baseline[key][method]) for key in common]
        new = [set_f1(candidate[key]["target"], candidate[key][method]) for key in common]
        nonempty = [index for index, key in enumerate(common) if baseline[key]["target"]]
        differences = [right - left for left, right in zip(old, new)]
        bootstraps = [
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
            for _ in range(args.bootstrap_samples)
        ]
        comparisons[method] = {
            "baseline_macro_set_f1": sum(old) / len(old),
            "candidate_macro_set_f1": sum(new) / len(new),
            "delta": sum(differences) / len(differences),
            "baseline_nonempty_macro_set_f1": (
                sum(old[index] for index in nonempty) / len(nonempty)
            ),
            "candidate_nonempty_macro_set_f1": (
                sum(new[index] for index in nonempty) / len(nonempty)
            ),
            "bootstrap_p05": percentile(bootstraps, 0.05),
            "bootstrap_p50": percentile(bootstraps, 0.50),
            "bootstrap_p95": percentile(bootstraps, 0.95),
            "bootstrap_probability_gain": (
                sum(value > 0 for value in bootstraps) / len(bootstraps)
            ),
        }
    summary = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "baseline_records": len(baseline),
        "candidate_records": len(candidate),
        "common_records": len(common),
        "paired_reference_targets_identical": all(
            baseline[key]["target"] == candidate[key]["target"] for key in common
        ),
        "comparisons": comparisons,
        "selection_claim": "paired diagnostic; no promotion from this script alone",
        "public_test_answers_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
