#!/usr/bin/env python3
"""Collect and align the official EBISU Label Studio rationale export."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_human_rationales import (  # noqa: E402
    choice_label, compact, locate_unique_fragment, note_fragments,
)
from fineval.parsing import parse_icr_query  # noqa: E402


REVISION = "1259fa3f23801e3501067ecd562d02e4820f1818"
SOURCE_URL = (
    "https://raw.githubusercontent.com/The-FinAI/Ebisu/" + REVISION +
    "/data/Japanese/project-27-at-2026-01-22-15-03-5125843f.json"
)
EXPECTED_SHA256 = "970c6ca44d1319b962ce15855c0df345752b45ff89b02d3db76cce9dafb2e375"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/external/ebisu_human_rationales",
    )
    args = parser.parse_args()

    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()
    payload = response.content
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != EXPECTED_SHA256:
        raise ValueError(f"upstream annotation hash changed: {payload_hash}")
    tasks = json.loads(payload)

    benchmark = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    folds = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    parsed = [parse_icr_query(query) for query in benchmark["query"]]
    key_to_indices = defaultdict(list)
    for index, item in enumerate(parsed):
        key_to_indices[(compact(item.question), compact(item.response))].append(index)

    grouped = defaultdict(list)
    unmatched = []
    for task in tasks:
        key = (compact(task["data"]["question"]), compact(task["data"]["answer"]))
        indices = key_to_indices.get(key, [])
        if len(indices) != 1:
            unmatched.append({"task_id": task.get("id"), "matches": indices})
            continue
        grouped[indices[0]].append(task)
    if unmatched:
        raise ValueError(f"annotation tasks did not match uniquely: {unmatched[:10]}")

    records = []
    annotation_label_counts = Counter()
    gold_counts = Counter()
    rows_with_span = Counter()
    for index in sorted(grouped):
        row = benchmark.iloc[index]
        gold = str(row.answer)
        item = parsed[index]
        spans = []
        observed_labels = []
        matching_gold_annotations = 0
        raw_task_ids = []
        for task in grouped[index]:
            raw_task_ids.append(int(task["id"]))
            for annotation_index, annotation in enumerate(task.get("annotations", [])):
                label = choice_label(annotation)
                if label is not None:
                    observed_labels.append(label)
                    annotation_label_counts[label] += 1
                if label != gold:
                    continue
                matching_gold_annotations += 1
                for fragment in note_fragments(annotation):
                    geometry = locate_unique_fragment(item.response, fragment)
                    if geometry is None:
                        continue
                    start, end = geometry
                    spans.append({
                        "start": start,
                        "end": end,
                        "text": item.response[start:end],
                        "normalized_fragment": compact(fragment),
                        "raw_task_id": int(task["id"]),
                        "annotation_index": annotation_index,
                    })
        unique_spans = []
        seen = set()
        for span in sorted(spans, key=lambda row: (row["start"], row["end"], row["text"])):
            key = span["start"], span["end"]
            if key not in seen:
                unique_spans.append(span)
                seen.add(key)
        source_fold = int(folds.loc[int(row.id), "source_fold"])
        gold_counts[gold] += 1
        if unique_spans:
            rows_with_span[gold] += 1
        records.append({
            "benchmark_id": int(row.id),
            "benchmark_row_index": int(index),
            "gold_label": gold,
            "source_fold": source_fold,
            "question": item.question,
            "response": item.response,
            "raw_task_ids": sorted(raw_task_ids),
            "annotation_labels": observed_labels,
            "matching_gold_annotations": matching_gold_annotations,
            "rationale_spans": unique_spans,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "label_studio_export.json").write_bytes(payload)
    with (args.output_dir / "rationales.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "source_sha256": payload_hash,
        "source_repository_license": None,
        "linked_huggingface_dataset_license": "apache-2.0",
        "competition_use_status": "policy_gated_pending_license_clarification",
        "raw_tasks": len(tasks),
        "matched_unique_benchmark_rows": len(records),
        "duplicate_raw_tasks": len(tasks) - len(records),
        "benchmark_rows": len(benchmark),
        "rows_with_exact_gold_aligned_rationale": sum(
            bool(row["rationale_spans"]) for row in records
        ),
        "exact_rationale_spans": sum(len(row["rationale_spans"]) for row in records),
        "gold_label_counts": dict(sorted(gold_counts.items())),
        "rows_with_exact_rationale_by_label": dict(sorted(rows_with_span.items())),
        "annotation_label_counts": dict(sorted(annotation_label_counts.items())),
        "source_folds_covered": sorted({row["source_fold"] for row in records}),
        "matched_rows_by_source_fold": {
            str(fold): sum(row["source_fold"] == fold for row in records)
            for fold in sorted({row["source_fold"] for row in records})
        },
        "rationale_rows_by_source_fold": {
            str(fold): sum(
                row["source_fold"] == fold and bool(row["rationale_spans"])
                for row in records
            )
            for fold in sorted({row["source_fold"] for row in records})
        },
        "current_or_hidden_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
