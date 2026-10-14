#!/usr/bin/env python3
"""Train-only structural gate for GLiNER legal-name boundary repairs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task1_numeric_baseline import parse_answer, score  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "task1_policy", ROOT / "scripts/evaluate_task1_annotation_policy.py"
)
POLICY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(POLICY)


def choose_threshold(candidates, minimum_lcb: float, minimum_count: int):
    eligible = [
        candidate for candidate in candidates
        if candidate["type"] == "ΟΡΓΑΝΙΣΜΟΣ"
        and POLICY.LEGAL_RE.search(candidate["entity"])
    ]
    frontier = []
    for threshold in np.linspace(0.50, 0.99, 99):
        selected = [row for row in eligible if row["gliner_score"] >= threshold]
        correct = sum(row["correct"] for row in selected)
        frontier.append({
            "threshold": float(threshold), "selected": len(selected),
            "correct": correct,
            "precision": correct / len(selected) if selected else 0.0,
            "wilson_precision_lcb": POLICY.wilson_lower(correct, len(selected)),
        })
    admissible = [
        row for row in frontier
        if row["selected"] >= minimum_count
        and row["wilson_precision_lcb"] >= minimum_lcb
    ]
    if not admissible:
        return 1.0, frontier
    chosen = max(admissible, key=lambda row: (row["selected"], -row["threshold"]))
    return chosen["threshold"], frontier


def locate_one(text: str, surface: str):
    match = re.search(re.escape(surface), text)
    return match.span() if match else None


def starts_with_unannotated_article(surface: str) -> bool:
    # None of the 706 unique released training entities includes a leading
    # standalone Greek definite article.  Keep this annotation convention out
    # of the learned score threshold so it remains transparent and auditable.
    return bool(re.match(r"^(?:Η|Ο|ΟΙ|Οι|H)\s+", surface))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-precision-lcb", type=float, default=0.90)
    parser.add_argument("--minimum-train-count", type=int, default=20)
    parser.add_argument(
        "--train-raw", type=Path,
        default=ROOT / "artifacts/task1_gliner_proposals_train/raw_predictions.jsonl",
    )
    parser.add_argument(
        "--validation-raw", type=Path,
        default=ROOT / "artifacts/task1_gliner_proposals/raw_predictions.jsonl",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/task1_gliner_boundary_repair_v1")
    parser.add_argument("--incumbent", type=Path, nargs="+", default=[
        ROOT / "artifacts/task1_textual/validation_nested_release_encoded.jsonl",
    ])
    args = parser.parse_args()

    train = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
    )
    validation = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    train_candidates = POLICY.make_candidates(
        train, POLICY.load_raw(args.train_raw)
    )
    validation_candidates = POLICY.make_candidates(
        validation, POLICY.load_raw(args.validation_raw)
    )
    threshold, frontier = choose_threshold(
        train_candidates, args.minimum_precision_lcb, args.minimum_train_count,
    )
    incumbent = POLICY.load_incumbent(args.incumbent, len(validation))
    predictions = [set(row) for row in incumbent]
    repairs = []
    for candidate in sorted(validation_candidates, key=lambda row: row["gliner_score"], reverse=True):
        if candidate["type"] != "ΟΡΓΑΝΙΣΜΟΣ":
            continue
        if candidate["gliner_score"] < threshold or not POLICY.LEGAL_RE.search(candidate["entity"]):
            continue
        if starts_with_unannotated_article(candidate["entity"]):
            continue
        row_index = candidate["row_index"]
        proposed = (candidate["entity"], candidate["type"])
        if proposed in predictions[row_index]:
            continue
        text = str(validation.iloc[row_index].text)
        incumbent_spans = []
        for entity, kind in predictions[row_index]:
            located = locate_one(text, entity)
            if located is not None:
                incumbent_spans.append((*located, entity, kind))
        overlapping = [
            row for row in incumbent_spans
            if row[0] < candidate["end"] and row[1] > candidate["start"]
        ]
        same_type = [row for row in overlapping if row[3] == candidate["type"]]
        if not same_type:
            # This gate repairs fragmentation; it does not add unrelated names.
            continue
        if not all(
            candidate["start"] <= left and right <= candidate["end"]
            for left, right, _, _ in same_type
        ):
            continue
        removed = {(entity, kind) for _, _, entity, kind in same_type}
        # Tokenizer pathologies can assign the two halves of one Latin word to
        # different types (e.g. ``Ir`` / ``aq``). Remove only tiny fragments,
        # preserving genuine train-supported nested locations.
        removed.update({
            (entity, kind) for left, right, entity, kind in overlapping
            if candidate["start"] <= left and right <= candidate["end"]
            and len(entity.strip()) <= 2
        })
        predictions[row_index].difference_update(removed)
        predictions[row_index].add(proposed)
        repairs.append({
            "row_index": row_index, "added": proposed, "removed": sorted(removed),
            "gliner_score": candidate["gliner_score"],
            "validation_correct": bool(candidate["correct"]),
        })

    gold = [set(parse_answer(answer)) for answer in validation.answer]
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
    bootstrap = deltas[rng.integers(0, len(deltas), size=(10000, len(deltas)))].mean(1)
    chosen_train = next(row for row in frontier if row["threshold"] == threshold)
    summary = {
        "gate_selected_from_train_only": chosen_train,
        "leading_article_gold_count_in_706_unique_train_entities": 0,
        "validation_repairs": len(repairs),
        "validation_correct_repair_candidates": sum(row["validation_correct"] for row in repairs),
        "incumbent_macro_entity_f1": float(base_scores.mean()),
        "repaired_macro_entity_f1": float(new_scores.mean()),
        "delta": float(deltas.mean()),
        "rows_improved": int((deltas > 0).sum()),
        "rows_harmed": int((deltas < 0).sum()),
        "bootstrap_p05": float(np.quantile(bootstrap, 0.05)),
        "bootstrap_p50": float(np.quantile(bootstrap, 0.50)),
        "bootstrap_p95": float(np.quantile(bootstrap, 0.95)),
        "bootstrap_probability_gain": float((bootstrap > 0).mean()),
        "public_test_used": False,
        "train_raw": str(args.train_raw),
        "validation_raw": str(args.validation_raw),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frontier).to_csv(args.output_dir / "train_gate_frontier.csv", index=False)
    (args.output_dir / "repairs.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in repairs
    ))
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({
            "row_index": index,
            "answer": "\n".join(f"{entity}, {kind}" for entity, kind in sorted(row)),
        }, ensure_ascii=False) + "\n"
        for index, row in enumerate(predictions)
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for repair in repairs:
        print(json.dumps(repair, ensure_ascii=False))


if __name__ == "__main__":
    main()
