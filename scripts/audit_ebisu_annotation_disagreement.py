#!/usr/bin/env python3
"""Audit unused multi-annotator uncertainty in the official EBISU export."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_human_rationales import choice_label  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402
from train_icr_japanese_cv import LABELS  # noqa: E402


LABEL_VALUE = {label: value for label, value in zip(LABELS, range(-2, 3))}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotation_statistics(labels: list[str], gold: str) -> dict[str, Any]:
    if not labels:
        raise ValueError("annotation row has no valid labels")
    counts = Counter(labels)
    maximum = max(counts.values())
    plurality = sorted(label for label, count in counts.items() if count == maximum)
    probabilities = np.asarray([counts[label] / len(labels) for label in LABELS])
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    return {
        "annotation_count": len(labels),
        "label_counts": dict(sorted(counts.items())),
        "unanimous": len(counts) == 1,
        "disagreement": len(counts) > 1,
        "plurality_labels": plurality,
        "gold_present": gold in counts,
        "gold_is_plurality": gold in plurality,
        "gold_is_unique_majority": plurality == [gold],
        "gold_agreement_fraction": counts.get(gold, 0) / len(labels),
        "annotator_ordinal_mean": float(np.mean([LABEL_VALUE[x] for x in labels])),
        "gold_ordinal_value": LABEL_VALUE[gold],
        "entropy_nats": entropy,
        "normalized_entropy": entropy / math.log(len(LABELS)),
    }


def load_probabilities(path: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"gold": str, "prediction": str})
    if not np.array_equal(frame["id"].to_numpy(), ids):
        raise ValueError(f"OOF IDs are misaligned: {path}")
    logits = frame[[f"logit_{label}" for label in LABELS]].to_numpy(float)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--curated", type=Path,
        default=ROOT / "data/external/ebisu_human_rationales/rationales.jsonl",
    )
    parser.add_argument(
        "--raw-export", type=Path,
        default=ROOT / "data/external/ebisu_human_rationales/label_studio_export.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_annotation_disagreement_audit_v1",
    )
    args = parser.parse_args()

    benchmark = pd.read_parquet(
        ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    )
    folds = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    curated = [
        json.loads(line) for line in args.curated.read_text().splitlines()
        if line.strip()
    ]
    by_id = {int(row["benchmark_id"]): row for row in curated}
    if len(by_id) != len(curated):
        raise ValueError("duplicate benchmark IDs in curated annotations")

    probabilities = (
        load_probabilities(
            ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv",
            benchmark.id.to_numpy(),
        )
        + load_probabilities(
            ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv",
            benchmark.id.to_numpy(),
        )
    ) / 2
    raw_prediction = decode_probabilities(probabilities, ordinal_mean=True)
    rules = np.asarray([
        rule_prediction(query, "safe_rules") for query in benchmark["query"]
    ])
    incumbent = np.where(rules != "+1", rules, raw_prediction)
    gold = benchmark.answer.astype(str).to_numpy()

    details: list[dict[str, Any]] = []
    for position, row in benchmark.iterrows():
        benchmark_id = int(row.id)
        if benchmark_id not in by_id:
            continue
        source = by_id[benchmark_id]
        stats = annotation_statistics(source["annotation_labels"], str(row.answer))
        details.append({
            "benchmark_id": benchmark_id,
            "benchmark_row_index": int(position),
            "source_fold": int(folds.loc[benchmark_id, "source_fold"]),
            "gold_label": str(row.answer),
            "incumbent_prediction": str(incumbent[position]),
            "incumbent_correct": bool(incumbent[position] == gold[position]),
            **stats,
        })

    raw_tasks = json.loads(args.raw_export.read_text())
    annotator_counts: dict[str, Counter] = defaultdict(Counter)
    pair_agreement: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for task in raw_tasks:
        judgments = []
        for annotation in task.get("annotations", []):
            label = choice_label(annotation)
            if label is None:
                continue
            annotator = str(annotation.get("completed_by"))
            annotator_counts[annotator][label] += 1
            judgments.append((annotator, label))
        for (left, left_label), (right, right_label) in itertools.combinations(
            judgments, 2,
        ):
            key = tuple(sorted((left, right)))
            pair_agreement[key][0] += int(left_label == right_label)
            pair_agreement[key][1] += 1

    annotated_ids = {item["benchmark_id"] for item in details}
    unanimous = [item for item in details if item["unanimous"]]
    disagreement = [item for item in details if item["disagreement"]]
    unannotated_positions = [
        index for index, value in enumerate(benchmark.id) if int(value) not in annotated_ids
    ]
    summary = {
        "experiment": "ebisu_annotation_disagreement_audit_v1",
        "inputs": {
            "curated_sha256": digest(args.curated),
            "raw_export_sha256": digest(args.raw_export),
        },
        "coverage": {
            "benchmark_rows": len(benchmark),
            "annotated_rows": len(details),
            "unannotated_rows": len(benchmark) - len(details),
            "annotations": sum(item["annotation_count"] for item in details),
            "multi_annotated_rows": sum(item["annotation_count"] >= 2 for item in details),
            "unanimous_rows": len(unanimous),
            "disagreement_rows": len(disagreement),
            "disagreement_rows_by_fold": {
                str(fold): sum(item["source_fold"] == fold for item in disagreement)
                for fold in range(5)
            },
            "annotated_rows_by_fold": {
                str(fold): sum(item["source_fold"] == fold for item in details)
                for fold in range(5)
            },
        },
        "adjudication": {
            "gold_present_rows": sum(item["gold_present"] for item in details),
            "gold_plurality_rows": sum(item["gold_is_plurality"] for item in details),
            "gold_unique_majority_rows": sum(
                item["gold_is_unique_majority"] for item in details
            ),
            "gold_not_plurality_ids": [
                item["benchmark_id"] for item in details
                if not item["gold_is_plurality"]
            ],
        },
        "incumbent": {
            "all_accuracy": float(np.mean(incumbent == gold)),
            "annotated_accuracy": float(np.mean([
                item["incumbent_correct"] for item in details
            ])),
            "unannotated_accuracy": float(np.mean(
                incumbent[unannotated_positions] == gold[unannotated_positions]
            )),
            "unanimous_accuracy": float(np.mean([
                item["incumbent_correct"] for item in unanimous
            ])),
            "disagreement_accuracy": float(np.mean([
                item["incumbent_correct"] for item in disagreement
            ])),
        },
        "annotators": {
            annotator: {
                "annotations": sum(counts.values()),
                "label_counts": dict(sorted(counts.items())),
            }
            for annotator, counts in sorted(annotator_counts.items())
        },
        "raw_task_pairwise_exact_agreement": {
            "-".join(pair): {
                "agreements": values[0],
                "overlap_tasks": values[1],
                "agreement": values[0] / values[1],
            }
            for pair, values in sorted(pair_agreement.items())
        },
        "interpretation": (
            "Disagreement is a hard-example marker, but annotator-distribution "
            "targets would contradict adjudicated extreme labels. Preserve the "
            "official label and test disagreement-only loss upweighting."
        ),
        "competition_use_status": "policy_gated_pending_license_clarification",
        "current_or_hidden_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rows.jsonl").open("w") as handle:
        for item in details:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

