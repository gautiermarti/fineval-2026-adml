#!/usr/bin/env python3
"""Summarize entity-disjoint GLiNER out-of-fold predictions.

The report uses only the Task 1 training labels.  A single confidence threshold
is applied to every held-out row, and results are broken down by the fold that
produced each prediction so that a strong aggregate cannot hide a failed
entity/source group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import encode_release_entity  # noqa: E402


LABEL_TO_TYPE = {
    "person": "ΠΡΟΣΩΠΟ",
    "organization": "ΟΡΓΑΝΙΣΜΟΣ",
    "location": "ΤΟΠΟΘΕΣΙΑ",
}


def load_records(path: Path, row_count: int) -> list[dict]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    by_row = {int(record["row_index"]): record for record in records}
    if sorted(by_row) != list(range(row_count)) or len(records) != row_count:
        raise ValueError(f"{path}: predictions do not cover every row exactly once")
    return [by_row[index] for index in range(row_count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-input", type=Path,
        default=ROOT
        / "artifacts/task1_gliner_oof_seed7_width16_ensemble/oof_raw_predictions.jsonl",
    )
    parser.add_argument(
        "--input", type=Path,
        default=ROOT
        / "data/external/plutus_finner_text/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT
        / "artifacts/task1_gliner_oof_seed7_width16_ensemble/oof_threshold_summary.json",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    records = load_records(args.raw_input, len(frame))
    gold = [set(parse_answer(answer)) for answer in frame["answer"]]
    thresholds = (
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
        0.80, 0.90, 0.93, 0.95, 0.98, 0.99,
    )
    rows = []
    for threshold in thresholds:
        predictions: list[set[tuple[str, str]]] = []
        candidate_count = candidate_correct = 0
        for record, target in zip(records, gold):
            prediction = set()
            for entity in record["entities"]:
                if float(entity["score"]) < threshold:
                    continue
                kind = LABEL_TO_TYPE[str(entity["label"])]
                item = (encode_release_entity(str(entity["text"]), kind), kind)
                prediction.add(item)
                candidate_count += 1
                candidate_correct += int(item in target)
            predictions.append(prediction)

        per_row = [
            score(list(target), list(prediction))[2]
            for target, prediction in zip(gold, predictions)
        ]
        fold_metrics = {}
        for fold in sorted({int(record["fold"]) for record in records}):
            indices = [
                index for index, record in enumerate(records)
                if int(record["fold"]) == fold
            ]
            fold_hits = sum(len(gold[index] & predictions[index]) for index in indices)
            fold_predictions = sum(len(predictions[index]) for index in indices)
            fold_gold = sum(len(gold[index]) for index in indices)
            fold_metrics[str(fold)] = {
                "rows": len(indices),
                "macro_entity_f1": sum(per_row[index] for index in indices) / len(indices),
                "typed_hits": fold_hits,
                "predicted_entities": fold_predictions,
                "gold_entities": fold_gold,
            }
        rows.append({
            "threshold": threshold,
            "macro_entity_f1": sum(per_row) / len(per_row),
            "typed_hits": sum(len(target & prediction) for target, prediction in zip(gold, predictions)),
            "predicted_entities": sum(len(prediction) for prediction in predictions),
            "gold_entities": sum(len(target) for target in gold),
            "candidate_precision": candidate_correct / candidate_count if candidate_count else 0.0,
            "folds": fold_metrics,
        })

    best = max(rows, key=lambda row: row["macro_entity_f1"])
    report = {
        "raw_input": str(args.raw_input),
        "rows": len(frame),
        "selection_labels": "Task 1 train only",
        "best_common_threshold": best,
        "thresholds": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
