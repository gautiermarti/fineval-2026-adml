#!/usr/bin/env python3
"""Evaluate a preregistered exact dual-seed raw-proposal policy for Task 1.

The operation classifier and its threshold are learned only from genuine OOF
training predictions.  Validation and the historical Plutus test set are
scored only after the OOF gate is frozen.  See
``docs/task1_dual_raw_proposer_preregistration.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_task1_oof_annotation_program import (  # noqa: E402
    evaluate_policy,
    fit_components,
    load_oof_incumbent,
    load_validation_incumbent,
    make_operations,
    transform,
)
from task1_numeric_baseline import parse_answer  # noqa: E402
from train_task1_textual import encode_release_entity  # noqa: E402


LABEL_TO_TYPE = {
    "person": "ΠΡΟΣΩΠΟ",
    "organization": "ΟΡΓΑΝΙΣΜΟΣ",
    "location": "ΤΟΠΟΘΕΣΙΑ",
}
PREREGISTRATION = ROOT / "docs/task1_dual_raw_proposer_preregistration.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_row_records(records: list[dict[str, Any]], rows: int, name: str) -> None:
    if len(records) != rows:
        raise ValueError(f"{name}: expected {rows} records, found {len(records)}")
    indices = [int(record.get("row_index", fallback)) for fallback, record in enumerate(records)]
    if sorted(indices) != list(range(rows)):
        raise ValueError(f"{name}: row indices are not an exact 0..{rows - 1} cover")


def raw_index(
    frame: pd.DataFrame,
    records: list[dict[str, Any]],
    *,
    minimum_votes: int | None,
) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    """Return the best raw entity for each exact typed character span."""
    validate_row_records(records, len(frame), "raw proposal view")
    output: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for fallback, record in enumerate(records):
        row_index = int(record.get("row_index", fallback))
        text = str(frame.iloc[row_index].text)
        for entity in record.get("entities", []):
            if minimum_votes is not None and int(entity.get("vote_count", 0)) < minimum_votes:
                continue
            start, end = int(entity["start"]), int(entity["end"])
            if start < 0 or end <= start or end > len(text):
                raise ValueError(
                    f"row {row_index}: invalid proposal span ({start}, {end}) "
                    f"for text length {len(text)}"
                )
            label = str(entity["label"])
            if label not in LABEL_TO_TYPE:
                raise ValueError(f"row {row_index}: unknown GLiNER label {label!r}")
            kind = LABEL_TO_TYPE[label]
            key = row_index, start, end, kind
            score = float(entity["score"])
            candidate = {
                "score": score,
                "surface_text": text[start:end],
                "vote_count": (
                    None if "vote_count" not in entity else int(entity["vote_count"])
                ),
            }
            if key not in output or score > float(output[key]["score"]):
                output[key] = candidate
    return output


def intersect_raw_proposals(
    frame: pd.DataFrame,
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    *,
    minimum_votes: int | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    """Intersect proposer views by exact row/span/type and use their min score."""
    left = raw_index(frame, left_records, minimum_votes=minimum_votes)
    right = raw_index(frame, right_records, minimum_votes=minimum_votes)
    shared_keys = sorted(set(left) & set(right))
    by_row: list[list[dict[str, Any]]] = [[] for _ in range(len(frame))]
    for row_index, start, end, kind in shared_keys:
        surface = str(frame.iloc[row_index].text)[start:end]
        by_row[row_index].append({
            "row_index": row_index,
            "start": start,
            "end": end,
            "type": kind,
            "surface_text": surface,
            "entity": encode_release_entity(surface, kind),
            "score": min(float(left[(row_index, start, end, kind)]["score"]),
                         float(right[(row_index, start, end, kind)]["score"])),
            "seed7_score": float(left[(row_index, start, end, kind)]["score"]),
            "seed19_score": float(right[(row_index, start, end, kind)]["score"]),
        })
    return by_row, {
        "left_unique_spans": len(left),
        "right_unique_spans": len(right),
        "shared_exact_spans": len(shared_keys),
        "rows_with_shared_spans": sum(bool(items) for items in by_row),
    }


def operation_key(operation: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(operation["row_index"]),
        tuple(operation["candidate"]),
        str(operation["operation"]),
        tuple(tuple(item) for item in operation["remove"]),
    )


def harmful_fraction(
    accepted: list[dict[str, Any]], operations: list[dict[str, Any]],
) -> tuple[float, int, int, int]:
    targets = {operation_key(operation): float(operation["target_delta"])
               for operation in operations}
    deltas = [targets[operation_key(operation)] for operation in accepted]
    beneficial = sum(delta > 1e-12 for delta in deltas)
    harmful = sum(delta < -1e-12 for delta in deltas)
    neutral = len(deltas) - beneficial - harmful
    fraction = harmful / len(deltas) if deltas else 0.0
    return fraction, beneficial, harmful, neutral


def crossfit_probabilities(
    operations: list[dict[str, Any]], *, max_features: int,
) -> np.ndarray:
    probabilities = np.zeros(len(operations), dtype=float)
    indices = np.arange(len(operations))
    folds = np.asarray([int(operation["fold"]) for operation in operations])
    for fold in sorted(set(folds.tolist())):
        fit_indices = indices[folds != fold]
        held_indices = indices[folds == fold]
        components = fit_components(operations, fit_indices, max_features)
        probabilities[held_indices] = components[-1].predict_proba(
            transform(operations, held_indices, components)
        )[:, 1]
    return probabilities


def choose_oof_gate(
    frontier: list[dict[str, Any]],
    *,
    minimum_operations: int = 20,
    maximum_harmful_fraction: float = 0.15,
) -> dict[str, Any] | None:
    admissible = [
        row for row in frontier
        if int(row["selected_operations"]) >= minimum_operations
        and float(row["bootstrap_p05"]) > 0.0
        and all(float(value) > 1e-12 for value in row["fold_deltas"].values())
        and all(float(value) >= -1e-12 for value in row["type_deltas"].values())
        and float(row["harmful_operation_fraction"]) <= maximum_harmful_fraction
    ]
    return max(
        admissible,
        key=lambda row: (
            float(row["delta"]),
            float(row["bootstrap_p05"]),
            float(row["threshold"]),
        ),
        default=None,
    )


def write_predictions(path: Path, predictions: list[set[tuple[str, str]]]) -> None:
    path.write_text("".join(
        json.dumps({
            "row_index": row_index,
            "answer": "\n".join(
                f"{entity}, {kind}" for entity, kind in sorted(items)
            ),
        }, ensure_ascii=False) + "\n"
        for row_index, items in enumerate(predictions)
    ))


def evaluate_deployment_split(
    frame: pd.DataFrame,
    gold: list[set[tuple[str, str]]],
    incumbent_path: Path,
    seed7_path: Path,
    seed19_path: Path,
    components: tuple[Any, ...],
    gate: float,
) -> tuple[list[set[tuple[str, str]]], list[dict[str, Any]], dict[str, Any]]:
    incumbent, occurrences = load_validation_incumbent(frame, [incumbent_path])
    candidates, intersection = intersect_raw_proposals(
        frame, load_jsonl(seed7_path), load_jsonl(seed19_path), minimum_votes=3,
    )
    operations = make_operations(
        frame, gold, candidates, incumbent, occurrences, folds=None,
    )
    indices = np.arange(len(operations))
    probabilities = (
        components[-1].predict_proba(transform(operations, indices, components))[:, 1]
        if operations else np.asarray([], dtype=float)
    )
    predictions, accepted, metrics = evaluate_policy(
        gold, incumbent, operations, probabilities, gate, folds=None,
    )
    fraction, beneficial, harmful, neutral = harmful_fraction(accepted, operations)
    metrics.update({
        "raw_intersection": intersection,
        "candidate_operations": len(operations),
        "accepted_beneficial_operations": beneficial,
        "accepted_harmful_operations": harmful,
        "accepted_neutral_operations": neutral,
        "accepted_harmful_fraction": fraction,
    })
    return predictions, accepted, metrics


def validation_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        float(metrics["delta"]) > 0.0
        and int(metrics["rows_improved"]) >= 2
        and int(metrics["rows_harmed"]) == 0
        and all(float(value) >= -1e-12 for value in metrics["type_deltas"].values())
        and float(metrics["bootstrap_probability_gain"]) >= 0.80
    )


def historical_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        float(metrics["delta"]) >= -1e-12
        and int(metrics["rows_improved"]) > int(metrics["rows_harmed"])
        and float(metrics["bootstrap_probability_gain"]) >= 0.60
        and min(float(value) for value in metrics["type_deltas"].values()) >= -0.005
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_dual_raw_proposer_v1",
    )
    args = parser.parse_args()

    paths = {
        "train": ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet",
        "validation": ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet",
        "historical": ROOT / "data/external/plutus_finner_text/test-00000-of-00001.parquet",
        "seed7_oof": ROOT / "artifacts/task1_gliner_deploy_seed7_width16/oof_raw_predictions.jsonl",
        "seed19_oof": ROOT / "artifacts/task1_gliner_deploy_seed19_width16/oof_raw_predictions.jsonl",
        "oof_incumbent": ROOT / "artifacts/task1_multilabel_oof_legal_seed7/oof_predictions.jsonl",
        "validation_seed7": ROOT / "artifacts/task1_gliner_deploy_seed7_width16/validation_ensemble_vote3_raw.jsonl",
        "validation_seed19": ROOT / "artifacts/task1_gliner_deploy_seed19_width16/validation_ensemble_vote3_raw.jsonl",
        "validation_incumbent": ROOT / "artifacts/task1_conservative_composition_grouped_seed_v1/predictions.jsonl",
        "historical_seed7": ROOT / "artifacts/submission_portfolio_task1_enhanced_public_test_v1/task1_enhanced_work/seed7/ensemble_vote3_raw.jsonl",
        "historical_seed19": ROOT / "artifacts/submission_portfolio_task1_enhanced_public_test_v1/task1_enhanced_work/seed19/ensemble_vote3_raw.jsonl",
        "historical_incumbent": ROOT / "artifacts/submission_portfolio_task1_enhanced_public_test_v1/task1_grouped_consensus_primary.jsonl",
    }
    train = pd.read_parquet(paths["train"])
    train_gold = [set(parse_answer(answer)) for answer in train.answer]
    seed7_oof_records = load_jsonl(paths["seed7_oof"])
    seed19_oof_records = load_jsonl(paths["seed19_oof"])
    train_candidates, train_intersection = intersect_raw_proposals(
        train, seed7_oof_records, seed19_oof_records,
    )
    train_incumbent, train_occurrences = load_oof_incumbent(
        paths["oof_incumbent"], len(train),
    )
    incumbent_records = load_jsonl(paths["oof_incumbent"])
    folds = [int(record["fold"]) for record in seed7_oof_records]
    incumbent_folds = [int(record["fold"]) for record in incumbent_records]
    if folds != incumbent_folds:
        raise ValueError("seed-7 proposer and OOF incumbent fold assignments differ")
    train_operations = make_operations(
        train, train_gold, train_candidates,
        train_incumbent, train_occurrences, folds,
    )
    if not train_operations:
        raise RuntimeError("no exact dual-seed OOF disagreements were generated")
    oof_probabilities = crossfit_probabilities(
        train_operations, max_features=args.max_features,
    )

    frontier: list[dict[str, Any]] = []
    for threshold in np.linspace(0.05, 0.99, 95):
        _, accepted, metrics = evaluate_policy(
            train_gold, train_incumbent, train_operations,
            oof_probabilities, float(threshold), folds,
        )
        fraction, beneficial, harmful, neutral = harmful_fraction(
            accepted, train_operations,
        )
        metrics.update({
            "beneficial_operations": beneficial,
            "harmful_operations": harmful,
            "neutral_operations": neutral,
            "harmful_operation_fraction": fraction,
        })
        frontier.append(metrics)
    chosen = choose_oof_gate(frontier)
    gate = float(chosen["threshold"]) if chosen is not None else 1.01

    all_indices = np.arange(len(train_operations))
    components = fit_components(train_operations, all_indices, args.max_features)

    validation = pd.read_parquet(paths["validation"])
    validation_gold = [set(parse_answer(answer)) for answer in validation.answer]
    validation_predictions, validation_accepted, validation_metrics = (
        evaluate_deployment_split(
            validation, validation_gold, paths["validation_incumbent"],
            paths["validation_seed7"], paths["validation_seed19"],
            components, gate,
        )
    )

    historical = pd.read_parquet(paths["historical"])
    historical_gold = [set(parse_answer(answer)) for answer in historical.answer]
    historical_predictions, historical_accepted, historical_metrics = (
        evaluate_deployment_split(
            historical, historical_gold, paths["historical_incumbent"],
            paths["historical_seed7"], paths["historical_seed19"],
            components, gate,
        )
    )

    validation_pass = validation_gate(validation_metrics)
    historical_pass = historical_gate(historical_metrics)
    promotion = bool(chosen is not None and validation_pass and historical_pass)
    summary = {
        "preregistration": str(PREREGISTRATION.relative_to(ROOT)),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "input_sha256": {
            name: sha256(path) for name, path in paths.items()
        },
        "train_rows": len(train),
        "train_raw_intersection": train_intersection,
        "train_operations": len(train_operations),
        "train_beneficial_operations": sum(
            int(operation["beneficial"]) for operation in train_operations
        ),
        "oof_gate_status": "accepted" if chosen is not None else "abstain",
        "chosen_oof_gate": chosen,
        "validation": validation_metrics,
        "historical_plutus_test": historical_metrics,
        "validation_promotion_gate": validation_pass,
        "historical_replication_gate": historical_pass,
        "promotion_status": "promote" if promotion else "reject",
        "validation_residuals_motivated_hypothesis": True,
        "fineval_hidden_test_used": False,
        "post_hoc_tuning_allowed": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oof_gate_frontier.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in frontier
    ))
    write_predictions(
        args.output_dir / "validation_predictions.jsonl", validation_predictions,
    )
    write_predictions(
        args.output_dir / "historical_predictions.jsonl", historical_predictions,
    )
    (args.output_dir / "validation_accepted_operations.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in validation_accepted
    ))
    (args.output_dir / "historical_accepted_operations.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in historical_accepted
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
