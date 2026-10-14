#!/usr/bin/env python3
"""Build entity-linked counterfactual span-choice supervision for Task 1."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.task1_span_surgery import (  # noqa: E402
    alternative_asdict, span_surgeries,
)
from task1_numeric_baseline import parse_answer  # noqa: E402
from train_task1_textual import encode_release_entity  # noqa: E402
from train_task1_textual_multilabel import align_overlapping  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--folds", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/folds.csv",
    )
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_span_surgery_v1",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).reset_index(drop=True)
    folds_frame = pd.read_csv(args.folds).sort_values("row_index")
    if folds_frame.row_index.astype(int).tolist() != list(range(len(frame))):
        raise ValueError("fold assignment is not aligned to training rows")
    folds = folds_frame.fold.astype(int).to_numpy()

    pairs = []
    aligned_gold = 0
    released_gold = 0
    rows_with_unaligned = 0
    alternative_counts = Counter()
    type_counts = Counter()
    fold_counts = Counter()
    for row_index, row in frame.iterrows():
        text = str(row.text)
        gold_items = set(parse_answer(row.answer))
        released_gold += len(gold_items)
        annotations = align_overlapping(text, list(gold_items))
        aligned_gold += len(annotations)
        rows_with_unaligned += int(len(annotations) < len(gold_items))
        for gold_index, (start, end, entity_type) in enumerate(annotations):
            surface = text[start:end]
            gold_entity = encode_release_entity(surface, entity_type)
            alternatives = []
            for alternative in span_surgeries(
                text, start, end, entity_type, width=args.width,
            ):
                candidate_item = (
                    encode_release_entity(alternative.surface, alternative.entity_type),
                    alternative.entity_type,
                )
                # The released answer is a row-level set. A different occurrence
                # of an already-gold entity is not a negative under the metric.
                if candidate_item in gold_items:
                    continue
                alternatives.append(alternative)
            if not alternatives:
                continue
            pair_weight = 1.0 / len(alternatives)
            context_left = max(0, min([start] + [item.start for item in alternatives]) - 100)
            context_right = min(
                len(text), max([end] + [item.end for item in alternatives]) + 100,
            )
            for alternative in alternatives:
                for operation in alternative.operations:
                    alternative_counts[operation] += 1
                type_counts[entity_type] += 1
                fold_counts[int(folds[row_index])] += 1
                pairs.append({
                    "row_index": int(row_index),
                    "fold": int(folds[row_index]),
                    "gold_index": int(gold_index),
                    "pair_weight": pair_weight,
                    "context_start": context_left,
                    "context_end": context_right,
                    "context": text[context_left:context_right],
                    "preferred": {
                        "start": int(start), "end": int(end),
                        "entity_type": entity_type, "surface": surface,
                        "released_entity": gold_entity,
                    },
                    "rejected": alternative_asdict(alternative),
                })

    if not all(
        record["preferred"]["surface"]
        == str(frame.iloc[record["row_index"]].text)[
            record["preferred"]["start"]:record["preferred"]["end"]
        ]
        and record["rejected"]["surface"]
        == str(frame.iloc[record["row_index"]].text)[
            record["rejected"]["start"]:record["rejected"]["end"]
        ]
        for record in pairs
    ):
        raise AssertionError("span surgery produced a non-source substring")

    summary = {
        "training_rows": len(frame),
        "released_gold_items": released_gold,
        "aligned_gold_occurrences": aligned_gold,
        "alignment_rate": aligned_gold / released_gold,
        "rows_with_any_unaligned_gold": rows_with_unaligned,
        "pairwise_span_choices": len(pairs),
        "mean_alternatives_per_aligned_gold": len(pairs) / aligned_gold,
        "operation_counts": dict(alternative_counts),
        "preferred_type_pair_counts": dict(type_counts),
        "fold_pair_counts": {str(key): value for key, value in sorted(fold_counts.items())},
        "all_surfaces_exact_input_substrings": True,
        "row_level_gold_set_alternatives_removed": True,
        "entity_linked_fold_assignment_reused": True,
        "validation_or_public_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "span_choice_pairs.jsonl").write_text("".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in pairs
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
