#!/usr/bin/env python3
"""Audit fold-pure GLiNER confidence and a conservative union candidate.

This calibration is deliberately weaker than the disagreement-conditioned
annotation program: it estimates whether a proposal is gold, not whether the
incumbent already handles that row.  Therefore it may nominate a secondary
union profile but cannot promote it without the OOF-incumbent experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_task1_annotation_policy import (  # noqa: E402
    load_incumbent,
    load_raw,
    make_candidates,
    wilson_lower,
)
from task1_numeric_baseline import parse_answer, score  # noqa: E402


def summarize_subset(rows):
    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    return {
        "selected": total,
        "correct": correct,
        "precision": correct / total if total else 0.0,
        "wilson_precision_lcb": wilson_lower(correct, total),
    }


def deduplicate_answer_operations(candidates):
    """Keep one highest-scoring proposal per official row-level answer item.

    GRFinNER answers are sets of ``(surface, type)`` pairs, not mention-level
    spans.  Counting repeated occurrences as independent calibration examples
    would overstate both sample size and precision while the union decoder can
    apply the corresponding addition only once.
    """
    best = {}
    for candidate in candidates:
        key = candidate["row_index"], candidate["entity"], candidate["type"]
        incumbent = best.get(key)
        if incumbent is None or candidate["gliner_score"] > incumbent["gliner_score"]:
            best[key] = candidate
    return list(best.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof-raw", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/oof_raw_predictions.jsonl",
    )
    parser.add_argument(
        "--validation-raw", type=Path,
        default=ROOT / "artifacts/task1_gliner_finetuned_seed7_width16_cpu80/validation_raw.jsonl",
    )
    parser.add_argument("--minimum-count", type=int, default=50)
    parser.add_argument("--minimum-overall-lcb", type=float, default=0.90)
    parser.add_argument("--minimum-type-count", type=int, default=10)
    parser.add_argument("--minimum-type-lcb", type=float, default=0.75)
    parser.add_argument("--minimum-fold-count", type=int, default=5)
    parser.add_argument("--minimum-fold-precision", type=float, default=0.80)
    parser.add_argument("--incumbent", type=Path, nargs="+", default=[
        ROOT / "artifacts/task1_textual/validation_flat_ensemble_3model_release_encoded.jsonl",
        ROOT / "artifacts/task1_textual/validation_nested_release_encoded.jsonl",
    ])
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_calibration_width16",
    )
    args = parser.parse_args()

    train = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
    )
    validation = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    oof_records = load_raw(args.oof_raw)
    validation_records = load_raw(args.validation_raw)
    train_candidates_raw = make_candidates(train, oof_records)
    validation_candidates_raw = make_candidates(validation, validation_records)
    train_candidates = deduplicate_answer_operations(train_candidates_raw)
    validation_candidates = deduplicate_answer_operations(validation_candidates_raw)
    fold_by_row = {
        int(record["row_index"]): int(record["fold"]) for record in oof_records
    }
    for candidate in train_candidates:
        candidate["fold"] = fold_by_row[candidate["row_index"]]

    frontier = []
    for threshold in np.linspace(0.50, 0.99, 50):
        selected = [
            candidate for candidate in train_candidates
            if candidate["gliner_score"] >= threshold
        ]
        overall = summarize_subset(selected)
        by_type = {
            kind: summarize_subset([row for row in selected if row["type"] == kind])
            for kind in ("ΠΡΟΣΩΠΟ", "ΟΡΓΑΝΙΣΜΟΣ", "ΤΟΠΟΘΕΣΙΑ")
        }
        by_fold = {
            str(fold): summarize_subset([row for row in selected if row["fold"] == fold])
            for fold in range(5)
        }
        frontier.append({
            "threshold": float(threshold), "overall": overall,
            "by_type": by_type, "by_fold": by_fold,
        })

    admissible = []
    for row in frontier:
        if row["overall"]["selected"] < args.minimum_count:
            continue
        if row["overall"]["wilson_precision_lcb"] < args.minimum_overall_lcb:
            continue
        if any(
            values["selected"] < args.minimum_type_count
            or values["wilson_precision_lcb"] < args.minimum_type_lcb
            for values in row["by_type"].values()
        ):
            continue
        if any(
            values["selected"] < args.minimum_fold_count
            or values["precision"] < args.minimum_fold_precision
            for values in row["by_fold"].values()
        ):
            continue
        admissible.append(row)
    chosen = max(
        admissible,
        key=lambda row: (row["overall"]["selected"], -row["threshold"]),
        default=None,
    )
    threshold = chosen["threshold"] if chosen is not None else 1.01

    incumbent = load_incumbent(args.incumbent, len(validation))
    predictions = [set(row) for row in incumbent]
    additions = []
    for candidate in sorted(
        validation_candidates, key=lambda row: row["gliner_score"], reverse=True,
    ):
        if candidate["gliner_score"] < threshold:
            continue
        row_index = candidate["row_index"]
        item = candidate["entity"], candidate["type"]
        if item in predictions[row_index]:
            continue
        predictions[row_index].add(item)
        additions.append({
            "row_index": row_index, "entity": item[0], "type": item[1],
            "score": candidate["gliner_score"],
            "validation_correct": bool(candidate["correct"]),
        })

    gold = [set(parse_answer(answer)) for answer in validation["answer"]]
    base_scores = np.asarray([
        score(list(target), list(prediction))[2]
        for target, prediction in zip(gold, incumbent)
    ])
    new_scores = np.asarray([
        score(list(target), list(prediction))[2]
        for target, prediction in zip(gold, predictions)
    ])
    deltas = new_scores - base_scores
    rng = np.random.default_rng(7)
    bootstrap = deltas[
        rng.integers(0, len(deltas), size=(10000, len(deltas)))
    ].mean(axis=1)
    summary = {
        "gate_status": "accepted" if chosen is not None else "abstain",
        "chosen_oof_proposal_gate": chosen,
        "validation_additions": len(additions),
        "validation_correct_additions": sum(row["validation_correct"] for row in additions),
        "validation_incorrect_additions": sum(not row["validation_correct"] for row in additions),
        "incumbent_macro_entity_f1": float(base_scores.mean()),
        "union_macro_entity_f1": float(new_scores.mean()),
        "delta": float(deltas.mean()),
        "rows_improved": int((deltas > 0).sum()),
        "rows_harmed": int((deltas < 0).sum()),
        "bootstrap_p05": float(np.quantile(bootstrap, 0.05)),
        "bootstrap_p50": float(np.quantile(bootstrap, 0.50)),
        "bootstrap_p95": float(np.quantile(bootstrap, 0.95)),
        "bootstrap_probability_gain": float((bootstrap > 0).mean()),
        "promotion_status": "secondary_pending_oof_incumbent_and_second_seed",
        "calibration_unit": "unique row/entity/type answer operation",
        "train_raw_mentions": len(train_candidates_raw),
        "train_unique_operations": len(train_candidates),
        "validation_raw_mentions": len(validation_candidates_raw),
        "validation_unique_operations": len(validation_candidates),
        "official_validation_used_only_after_gate_freeze": True,
        "public_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frontier.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in frontier
    ))
    (args.output_dir / "additions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in additions
    ))
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({
            "row_index": row_index,
            "answer": "\n".join(
                f"{entity}, {kind}" for entity, kind in sorted(row)
            ),
        }, ensure_ascii=False) + "\n"
        for row_index, row in enumerate(predictions)
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
