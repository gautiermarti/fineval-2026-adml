#!/usr/bin/env python3
"""Build a fold-aware gold-anchored teacher calibration queue for JF-ICR."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_events import candidates_are_exact, event_candidates  # noqa: E402
from fineval.icr_silver import protected_tokens  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402


LABEL_QUOTAS = {"-2": 2, "-1": 11, "0": 25, "+1": 42, "+2": 20}


def fold_round_robin(rows: list[dict], quota: int) -> list[dict]:
    by_fold = defaultdict(list)
    for row in rows:
        by_fold[int(row["source_fold"])].append(row)
    for values in by_fold.values():
        values.sort(key=lambda row: int(row["id"]))
    selected = []
    while len(selected) < quota:
        progressed = False
        for fold in sorted(by_fold):
            if by_fold[fold] and len(selected) < quota:
                selected.append(by_fold[fold].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def teacher_prompts(row: dict) -> list[dict]:
    common = {
        "id": row["id"], "source_fold": row["source_fold"],
        "gold_label": row["gold_label"], "question": row["question"],
        "response": row["response"],
    }
    return [
        common | {
            "rubric": "event_factors",
            "instruction": (
                "Gold label is fixed. Quote the shortest exact response span that "
                "causally determines it, then annotate source, time, conditionality, "
                "modality, actuality, evaluation, focus, and responsiveness."
            ),
        },
        common | {
            "rubric": "ordinal_boundary",
            "instruction": (
                "Gold label is fixed. Quote the shortest exact response span that "
                "separates this label from each adjacent commitment level. Identify "
                "counterevidence; never invent or normalize the Japanese quote."
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet",
    )
    parser.add_argument(
        "--folds", type=Path,
        default=ROOT / "data/derived/icr_source_folds.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_decisive_event_queue_v1",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    folds = pd.read_csv(args.folds).set_index("id")
    records = []
    for row in frame.itertuples(index=False):
        parsed = parse_icr_query(row.query)
        candidates = event_candidates(parsed.question, parsed.response)
        if not candidates or not candidates_are_exact(parsed.response, candidates):
            raise AssertionError(f"row {row.id} has invalid event candidates")
        records.append({
            "id": int(row.id), "source_fold": int(folds.loc[row.id, "source_fold"]),
            "gold_label": str(row.answer), "question": parsed.question,
            "response": parsed.response,
            "response_protected_tokens": list(protected_tokens(parsed.response)),
            "event_candidates": [candidate.to_dict() for candidate in candidates],
            "top_input_only_event": candidates[0].to_dict(),
            "labels_used_for_event_ranking": False,
        })

    selected = []
    for label, quota in LABEL_QUOTAS.items():
        chosen = fold_round_robin(
            [row for row in records if row["gold_label"] == label], quota,
        )
        if len(chosen) != quota:
            raise ValueError(f"label {label}: requested {quota}, found {len(chosen)}")
        selected.extend(chosen)
    selected.sort(key=lambda row: int(row["id"]))
    prompts = [prompt for row in selected for prompt in teacher_prompts(row)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("all_events.jsonl", records),
        ("calibration_queue.jsonl", selected),
        ("teacher_prompts.jsonl", prompts),
    ):
        (args.output_dir / name).write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ))
    summary = {
        "official_training_rows": len(records),
        "calibration_rows": len(selected),
        "teacher_prompt_records": len(prompts),
        "label_quotas": LABEL_QUOTAS,
        "selected_label_counts": dict(Counter(row["gold_label"] for row in selected)),
        "selected_fold_counts": dict(Counter(row["source_fold"] for row in selected)),
        "all_event_candidates": sum(len(row["event_candidates"]) for row in records),
        "mean_event_candidates_per_row": sum(
            len(row["event_candidates"]) for row in records
        ) / len(records),
        "all_candidates_exact_substrings": True,
        "event_ranking_uses_labels": False,
        "gold_labels_used_only_for_calibration_sampling_and_teacher_anchoring": True,
        "current_or_hidden_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
