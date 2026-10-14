#!/usr/bin/env python3
"""Quantify where immutable-value evidence translation can affect Task 3."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.polyfiqa_translation import (  # noqa: E402
    extract_translatable_evidence, mask_immutables,
)


def audit(name: str, path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    segments = []
    for row_index, row in enumerate(rows):
        answer = str(row["answer"])
        for segment_index, segment in enumerate(extract_translatable_evidence(answer)):
            masked = mask_immutables(segment["text"])
            segments.append({
                "corpus": name, "row_index": row_index,
                "segment_index": segment_index, "task_id": row.get("task_id"),
                "language": segment["language"],
                "segment_words": len(segment["text"].split()),
                "answer_words": len(answer.split()),
                "immutable_count": len(masked.mapping),
            })
    languages = Counter(row["language"] for row in segments)
    row_ids = {row["row_index"] for row in segments}
    def stats(field):
        values = [row[field] for row in segments]
        return {
            "minimum": min(values) if values else 0,
            "median": statistics.median(values) if values else 0,
            "maximum": max(values) if values else 0,
        }
    return {
        "name": name, "path": str(path), "rows": len(rows),
        "rows_with_translatable_evidence": len(row_ids),
        "segments": len(segments), "segments_by_language": dict(languages),
        "segment_word_stats": stats("segment_words"),
        "answer_word_stats_for_affected_rows": stats("answer_words"),
        "immutable_count_stats": stats("immutable_count"),
    }, segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/polyfiqa_translation_opportunity")
    parser.add_argument("--historical", type=Path,
                        default=ROOT / "artifacts/polyfiqa_easy_july/hybrid_input_only_cjk_budget_v15.jsonl")
    parser.add_argument("--current", type=Path,
                        default=ROOT / "artifacts/submission_portfolio_all_current/task3_input_only.jsonl")
    args = parser.parse_args()
    summaries, segments = [], []
    for name, path in (("historical_easy_v15", args.historical),
                       ("current_task3", args.current)):
        summary, rows = audit(name, path)
        summaries.append(summary)
        segments.extend(rows)
    result = {
        "corpora": summaries,
        "decision": (
            "historical_ablation_only"
            if summaries[1]["rows_with_translatable_evidence"] == 0
            else "current_packet_pilot_applicable"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(segments).to_csv(args.output_dir / "segments.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
