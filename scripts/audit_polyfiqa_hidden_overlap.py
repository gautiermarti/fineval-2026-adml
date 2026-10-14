#!/usr/bin/env python3
"""Audit exact public-reference overlap in a PolyFiQA input-only JSONL."""

from argparse import ArgumentParser
from pathlib import Path
import json
import hashlib
import re
import sys
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_polyfiqa_query  # noqa: E402
SOURCES = {
    "easy": ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
    "expert": ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip())


def fingerprint(query: str, question: str) -> str:
    context = parse_polyfiqa_query(query)
    fields = (
        question, context.financial_statements, context.english_news,
        context.chinese_news, context.japanese_news, context.spanish_news,
        context.greek_news,
    )
    return hashlib.sha256("\x1f".join(map(normalize, fields)).encode()).hexdigest()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/polyfiqa_system/hidden_overlap_audit.csv",
    )
    args = parser.parse_args()
    if args.input.suffix == ".parquet":
        inputs = pd.read_parquet(args.input).to_dict("records")
    else:
        inputs = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    lookups, normalized_lookups, fingerprint_lookups = {}, {}, {}
    for tier, path in SOURCES.items():
        frame = pd.read_parquet(path)
        lookups[tier] = {
            row.query: (row.task_id, row.question, row.answer)
            for _, row in frame.iterrows()
        }
        normalized_lookups[tier] = {
            normalize(row.query): (row.task_id, row.question, row.answer)
            for _, row in frame.iterrows()
        }
        fingerprint_lookups[tier] = {
            fingerprint(row.query, row.question): (row.task_id, row.question, row.answer)
            for _, row in frame.iterrows()
        }
    rows = []
    for position, item in enumerate(inputs):
        tier = item.get("tier")
        if tier is None:
            tier = "easy" if item["question"].startswith((
                "What trends", "How does", "Are there", "What is the company's R&D",
            )) else "expert"
        match = lookups.get(tier, {}).get(item["query"])
        normalized_match = normalized_lookups.get(tier, {}).get(normalize(item["query"]))
        packet_match = fingerprint_lookups.get(tier, {}).get(
            fingerprint(item["query"], item["question"])
        )
        strongest_match = match or normalized_match or packet_match
        rows.append({
            "row_index": position, "task_id": item["task_id"], "tier": tier,
            "exact_query_match": match is not None,
            "normalized_query_match": normalized_match is not None,
            "exact_packet_match": packet_match is not None,
            "public_task_id": strongest_match[0] if strongest_match else "",
            "question_consistent": bool(strongest_match and strongest_match[1] == item["question"]),
            "public_answer_words": len(strongest_match[2].split()) if strongest_match else 0,
        })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.groupby("tier").agg(
        rows=("task_id", "size"), exact_matches=("exact_query_match", "sum"),
        normalized_matches=("normalized_query_match", "sum"),
        packet_matches=("exact_packet_match", "sum"),
        question_consistent=("question_consistent", "sum"),
    ).to_string())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
