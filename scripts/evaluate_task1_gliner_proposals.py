#!/usr/bin/env python3
"""Audit multilingual GLiNER as a nested Task 1 span proposer.

This is deliberately a proposal audit, not a promotion script.  It reports
standalone exact entity F1, typed proposal recall, and surface-only proposal
recall.  Threshold sweeps reuse one low-threshold inference pass.
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


def load_answer_candidates(paths: list[Path], row_count: int) -> list[set[tuple[str, str]]]:
    pool = [set() for _ in range(row_count)]
    for path in paths:
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if len(records) != row_count:
            raise ValueError(f"{path}: expected {row_count} rows, found {len(records)}")
        for index, record in enumerate(records):
            row_index = int(record.get("row_index", index))
            pool[row_index].update(parse_answer(record.get("answer", "")))
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet",
    )
    parser.add_argument("--model", default="urchade/gliner_multi-v2.1")
    parser.add_argument(
        "--raw-input", type=Path,
        help="Audit an existing low-threshold prediction JSONL without rerunning inference.",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/task1_gliner_proposals")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--minimum-threshold", type=float, default=0.05)
    parser.add_argument(
        "--existing-candidates", type=Path, nargs="*", default=[
            ROOT / "artifacts/task1_textual/validation_flat_ensemble_3model_release_encoded.jsonl",
            ROOT / "artifacts/task1_textual/validation_nested_release_encoded.jsonl",
        ],
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    if args.raw_input is not None:
        records = [
            json.loads(line) for line in args.raw_input.read_text().splitlines()
            if line.strip()
        ]
        raw = [None] * len(frame)
        for fallback_index, record in enumerate(records):
            row_index = int(record.get("row_index", fallback_index))
            raw[row_index] = record["entities"]
        if any(row is None for row in raw):
            raise ValueError(f"{args.raw_input}: missing one or more row predictions")
    else:
        from gliner import GLiNER

        model = GLiNER.from_pretrained(args.model, map_location="cpu")
        raw = model.batch_predict_entities(
            frame["text"].astype(str).tolist(), list(LABEL_TO_TYPE),
            flat_ner=False, multi_label=False, threshold=args.minimum_threshold,
            batch_size=args.batch_size,
        )
    if len(raw) != len(frame):
        raise RuntimeError(f"expected {len(frame)} GLiNER rows, found {len(raw)}")

    gold = [set(parse_answer(answer)) for answer in frame["answer"]]
    existing = load_answer_candidates(args.existing_candidates, len(frame))
    thresholds = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    summaries = []
    best_new_typed = set()
    best_new_surface = set()
    for threshold in thresholds:
        predictions = []
        for entities in raw:
            row = []
            for entity in entities:
                if float(entity["score"]) < threshold:
                    continue
                kind = LABEL_TO_TYPE[str(entity["label"])]
                surface = encode_release_entity(str(entity["text"]), kind)
                row.append((surface, kind))
            predictions.append(set(row))

        gold_total = sum(len(row) for row in gold)
        typed_hits = sum(len(g & p) for g, p in zip(gold, predictions))
        surface_hits = sum(
            len({surface for surface, _ in g} & {surface for surface, _ in p})
            for g, p in zip(gold, predictions)
        )
        existing_hits = sum(len(g & p) for g, p in zip(gold, existing))
        union_hits = sum(len(g & (p | old)) for g, p, old in zip(gold, predictions, existing))
        union_surface_hits = sum(
            len(
                {surface for surface, _ in g}
                & ({surface for surface, _ in p} | {surface for surface, _ in old})
            )
            for g, p, old in zip(gold, predictions, existing)
        )
        macro_f1 = sum(
            score(list(g), list(p))[2] for g, p in zip(gold, predictions)
        ) / len(gold)
        summary = {
            "threshold": threshold,
            "standalone_macro_entity_f1": macro_f1,
            "predicted_entities": sum(len(row) for row in predictions),
            "gold_entities": gold_total,
            "typed_proposal_hits": typed_hits,
            "typed_proposal_recall": typed_hits / gold_total,
            "surface_proposal_hits": surface_hits,
            "surface_proposal_recall": surface_hits / gold_total,
            "existing_pool_typed_hits": existing_hits,
            "union_pool_typed_hits": union_hits,
            "union_pool_typed_recall": union_hits / gold_total,
            "union_pool_surface_hits": union_surface_hits,
            "union_pool_surface_recall": union_surface_hits / gold_total,
        }
        summaries.append(summary)
        if union_hits == max(item["union_pool_typed_hits"] for item in summaries):
            best_new_typed = {
                (index, item) for index, (g, p, old) in enumerate(zip(gold, predictions, existing))
                for item in (g & p) - old
            }
        if union_surface_hits == max(item["union_pool_surface_hits"] for item in summaries):
            best_new_surface = {
                (index, surface) for index, (g, p, old) in enumerate(zip(gold, predictions, existing))
                for surface in (
                    ({value for value, _ in g} & {value for value, _ in p})
                    - {value for value, _ in old}
                )
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw_predictions.jsonl").write_text("".join(
        json.dumps({"row_index": index, "entities": entities}, ensure_ascii=False) + "\n"
        for index, entities in enumerate(raw)
    ))
    pd.DataFrame(summaries).to_csv(args.output_dir / "threshold_summary.csv", index=False)
    result = {
        "model": args.model,
        "device": "cpu",
        "flat_ner": False,
        "minimum_inference_threshold": args.minimum_threshold,
        "raw_input": str(args.raw_input) if args.raw_input is not None else None,
        "existing_candidate_files": [str(path) for path in args.existing_candidates],
        "best_union_typed_hits": max(item["union_pool_typed_hits"] for item in summaries),
        "best_union_surface_hits": max(item["union_pool_surface_hits"] for item in summaries),
        "new_typed_gold_proposals": [
            {"row_index": index, "entity": entity, "type": kind}
            for index, (entity, kind) in sorted(best_new_typed)
        ],
        "new_surface_gold_proposals": [
            {"row_index": index, "entity": entity}
            for index, entity in sorted(best_new_surface)
        ],
        "thresholds": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
