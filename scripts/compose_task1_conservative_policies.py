#!/usr/bin/env python3
"""Compose three independently train-gated Task 1 correction policies."""

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

from fineval.task1_policy_composition import compose_policy_deltas  # noqa: E402
from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import ENTITY_TYPES  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_sets(path: Path, rows: int) -> list[set[tuple[str, str]]]:
    records = load_jsonl(path)
    if len(records) != rows:
        raise ValueError(f"{path}: expected {rows} rows, got {len(records)}")
    output = [set() for _ in range(rows)]
    for fallback, record in enumerate(records):
        index = int(record.get("row_index", fallback))
        output[index].update(parse_answer(record.get("answer", "")))
    return output


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
        true_positive = predicted_count = gold_count = 0
        for target, guess in zip(gold, predicted, strict=True):
            target_type = {item for item in target if item[1] == kind}
            guess_type = {item for item in guess if item[1] == kind}
            true_positive += len(target_type & guess_type)
            predicted_count += len(guess_type)
            gold_count += len(target_type)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / gold_count if gold_count else 0.0
        result[kind] = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", type=Path, default=
                        ROOT / "artifacts/task1_oof_annotation_program_seed7/validation_predictions.jsonl")
    parser.add_argument("--structural", type=Path, default=
                        ROOT / "artifacts/task1_gliner_boundary_repair_oof_mean/predictions.jsonl")
    parser.add_argument("--union", type=Path, default=
                        ROOT / "artifacts/task1_gliner_oof_calibration_width16_mean/predictions.jsonl")
    parser.add_argument("--incumbent", type=Path, nargs="+", default=[
        ROOT / "artifacts/task1_textual/validation_flat_ensemble_3model_release_encoded.jsonl",
        ROOT / "artifacts/task1_textual/validation_nested_release_encoded.jsonl",
    ])
    parser.add_argument("--output-dir", type=Path, default=
                        ROOT / "artifacts/task1_conservative_composition_v1")
    parser.add_argument(
        "--two-seed-confirmed", action="store_true",
        help="Record that the supplied operation policy already requires two proposer seeds.",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    rows = len(frame)
    gold = [set(parse_answer(answer)) for answer in frame.answer]
    baseline = load_incumbent(args.incumbent, rows)
    named_policies = [
        ("annotation_operation", load_sets(args.operation, rows)),
        ("structural_repair", load_sets(args.structural, rows)),
        ("high_confidence_union", load_sets(args.union, rows)),
    ]
    composed = [
        compose_policy_deltas(
            baseline[index], [policy[index] for _, policy in named_policies],
        )
        for index in range(rows)
    ]
    baseline_scores = np.asarray([
        score(list(target), list(guess))[2]
        for target, guess in zip(gold, baseline, strict=True)
    ])
    composed_scores = np.asarray([
        score(list(target), list(guess))[2]
        for target, guess in zip(gold, composed, strict=True)
    ])
    deltas = composed_scores - baseline_scores
    rng = np.random.default_rng(7)
    bootstrap = deltas[
        rng.integers(0, rows, size=(10_000, rows))
    ].mean(axis=1)
    baseline_types = type_f1(gold, baseline)
    composed_types = type_f1(gold, composed)
    changed = []
    for index, delta in enumerate(deltas):
        if abs(float(delta)) <= 1e-12:
            continue
        changed.append({
            "row_index": index,
            "delta": float(delta),
            "added": sorted([list(item) for item in composed[index] - baseline[index]]),
            "removed": sorted([list(item) for item in baseline[index] - composed[index]]),
            "component_changes": {
                name: {
                    "added": sorted([list(item) for item in policy[index] - baseline[index]]),
                    "removed": sorted([list(item) for item in baseline[index] - policy[index]]),
                }
                for name, policy in named_policies
            },
        })
    summary = {
        "validation_rows": rows,
        "policy_order": [name for name, _ in named_policies],
        "all_component_gates_selected_without_validation_labels": True,
        "incumbent_macro_entity_f1": float(baseline_scores.mean()),
        "composed_macro_entity_f1": float(composed_scores.mean()),
        "delta": float(deltas.mean()),
        "rows_improved": int(np.sum(deltas > 0)),
        "rows_harmed": int(np.sum(deltas < 0)),
        "type_deltas": {
            kind: composed_types[kind] - baseline_types[kind]
            for kind in ENTITY_TYPES
        },
        "bootstrap_p05": float(np.quantile(bootstrap, 0.05)),
        "bootstrap_p50": float(np.quantile(bootstrap, 0.50)),
        "bootstrap_p95": float(np.quantile(bootstrap, 0.95)),
        "bootstrap_probability_gain": float(np.mean(bootstrap > 0)),
        "two_seed_confirmed": bool(args.two_seed_confirmed),
        "two_seed_confirmation_required": not bool(args.two_seed_confirmed),
        "public_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({
            "row_index": index,
            "answer": "\n".join(
                f"{entity}, {kind}" for entity, kind in sorted(items)
            ),
        }, ensure_ascii=False) + "\n"
        for index, items in enumerate(composed)
    ))
    (args.output_dir / "changed_rows.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in changed
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
