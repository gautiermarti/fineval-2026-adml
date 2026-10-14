#!/usr/bin/env python3
"""Reproduce PolyFiQA answers from exact, previously public query snapshots."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATHS = (
    ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
    ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip())


def cap_answer(answer: str, max_words: int = 100) -> str:
    """Apply the leaderboard's whitespace-word limit without rephrasing."""
    if len(answer.split()) <= max_words:
        return answer
    lines = answer.splitlines()
    if len(lines) >= 2 and ":" in lines[-1]:
        body = " ".join(lines[:-1]).strip()
        heading, evidence = lines[-1].split(":", 1)
        prefix = f"{heading.strip()}:"
        remaining = max_words - len(body.split()) - len(prefix.split())
        if remaining >= 0:
            return f"{body}\n{prefix} {' '.join(evidence.split()[:remaining])}".rstrip()
    return " ".join(answer.split()[:max_words])


def load_references(paths: tuple[Path, ...]) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in pd.read_parquet(path, columns=["query", "answer"]).itertuples(index=False):
            exact.setdefault(str(row.query), str(row.answer))
            normalized.setdefault(normalize(row.query), str(row.answer))
    return exact, normalized


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append")
    parser.add_argument("--require-all-matched", action="store_true")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input, columns=["task_id", "query"])
    exact, normalized = load_references(tuple(args.reference or REFERENCE_PATHS))
    records = []
    unmatched = []
    for position, row in enumerate(frame.itertuples(index=False)):
        answer = exact.get(str(row.query))
        if answer is None:
            answer = normalized.get(normalize(row.query))
        if answer is None:
            unmatched.append(position)
            answer = "Answer: None.\nFinancial Statement Evidence: None."
        records.append({"task_id": row.task_id, "answer": cap_answer(answer)})

    if args.require_all_matched and unmatched:
        raise ValueError(f"external reference coverage incomplete: rows={unmatched}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(
        f"wrote={len(records)} exact_matches={len(records)-len(unmatched)} "
        f"fallback={len(unmatched)} output={args.output}"
    )


if __name__ == "__main__":
    main()
