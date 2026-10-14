#!/usr/bin/env python3
"""Evaluate label-free majority consensus across GLiNER deployment views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.task1_operation_consensus import (  # noqa: E402
    apply_consensus_operations,
    grouped_consensus_operations,
    majority_operations,
)
from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import ENTITY_TYPES  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_incumbent(paths: list[Path], rows: int) -> list[set[tuple[str, str]]]:
    output = [set() for _ in range(rows)]
    for path in paths:
        records = load_jsonl(path)
        if len(records) != rows:
            raise ValueError(f"{path}: expected {rows} rows, got {len(records)}")
        for fallback, record in enumerate(records):
            index = int(record.get("row_index", fallback))
            output[index].update(parse_answer(record.get("answer", "")))
    return output


def type_f1(gold, predicted) -> dict[str, float]:
    result = {}
    for kind in ENTITY_TYPES:
        hits = guesses = targets = 0
        for target, guess in zip(gold, predicted, strict=True):
            target_type = {item for item in target if item[1] == kind}
            guess_type = {item for item in guess if item[1] == kind}
            hits += len(target_type & guess_type)
            guesses += len(guess_type)
            targets += len(target_type)
        precision = hits / guesses if guesses else 0.0
        recall = hits / targets if targets else 0.0
        result[kind] = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-operation-view", type=Path, action="append", required=True)
    parser.add_argument(
        "--view-group", action="append",
        help="Optional proposer group for each view (for example, a seed).",
    )
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--minimum-groups", type=int, default=1)
    parser.add_argument("--incumbent", type=Path, nargs="+", default=[
        ROOT / "artifacts/task1_textual/validation_flat_ensemble_3model_release_encoded.jsonl",
        ROOT / "artifacts/task1_textual/validation_nested_release_encoded.jsonl",
    ])
    parser.add_argument("--output-dir", type=Path, default=
                        ROOT / "artifacts/task1_operation_view_consensus_v1")
    args = parser.parse_args()

    if args.minimum_support < 1 or args.minimum_support > len(args.accepted_operation_view):
        raise ValueError("minimum support must be within the number of views")
    if args.view_group is not None and len(args.view_group) != len(args.accepted_operation_view):
        raise ValueError("--view-group must be supplied once per operation view")
    frame = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    rows = len(frame)
    gold = [set(parse_answer(answer)) for answer in frame.answer]
    baseline = load_incumbent(args.incumbent, rows)
    views = [load_jsonl(path) for path in args.accepted_operation_view]
    if args.view_group is None:
        if args.minimum_groups != 1:
            raise ValueError("minimum groups requires view groups")
        selected_operations = majority_operations(views, args.minimum_support)
    else:
        selected_operations = grouped_consensus_operations(
            views, args.view_group, args.minimum_support, args.minimum_groups,
        )
    predictions = apply_consensus_operations(baseline, selected_operations)

    baseline_scores = np.asarray([
        score(list(target), list(guess))[2]
        for target, guess in zip(gold, baseline, strict=True)
    ])
    policy_scores = np.asarray([
        score(list(target), list(guess))[2]
        for target, guess in zip(gold, predictions, strict=True)
    ])
    deltas = policy_scores - baseline_scores
    rng = np.random.default_rng(7)
    bootstrap = deltas[rng.integers(0, rows, size=(10_000, rows))].mean(axis=1)
    baseline_types = type_f1(gold, baseline)
    policy_types = type_f1(gold, predictions)
    summary = {
        "validation_rows": rows,
        "views": [str(path) for path in args.accepted_operation_view],
        "minimum_support": args.minimum_support,
        "view_groups": args.view_group,
        "minimum_groups": args.minimum_groups,
        "consensus_selected_without_validation_labels": True,
        "selected_operations": len(selected_operations),
        "incumbent_macro_entity_f1": float(baseline_scores.mean()),
        "policy_macro_entity_f1": float(policy_scores.mean()),
        "delta": float(deltas.mean()),
        "rows_improved": int(np.sum(deltas > 0)),
        "rows_harmed": int(np.sum(deltas < 0)),
        "type_deltas": {
            kind: policy_types[kind] - baseline_types[kind] for kind in ENTITY_TYPES
        },
        "bootstrap_p05": float(np.quantile(bootstrap, 0.05)),
        "bootstrap_p50": float(np.quantile(bootstrap, 0.50)),
        "bootstrap_p95": float(np.quantile(bootstrap, 0.95)),
        "bootstrap_probability_gain": float(np.mean(bootstrap > 0)),
        "public_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_operations.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in selected_operations
    ))
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({
            "row_index": index,
            "answer": "\n".join(
                f"{entity}, {kind}" for entity, kind in sorted(items)
            ),
        }, ensure_ascii=False) + "\n"
        for index, items in enumerate(predictions)
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
