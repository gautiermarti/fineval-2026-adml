#!/usr/bin/env python3
"""Evaluate explicitly family-aligned post-2021 Easy-July references."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.metrics import rouge1_f1  # noqa: E402


def family(question: str) -> str:
    if question.startswith("What trends"):
        return "revenue"
    if question.startswith("How does"):
        return "balance"
    if question.startswith("Are there"):
        return "cash"
    return "rd"


def aligned(reference: str, expected: str) -> bool:
    body = reference.casefold().split("news evidence", 1)[0]
    if expected == "revenue":
        trend = any(cue in body for cue in ("growth", "grew", "increas", "rose", "upward", "trend", " up "))
        return (
            ("revenue" in body or "sales" in body) and trend
            and "current assets" not in body and "cash flow" not in body
        )
    if expected == "balance":
        return "current assets" in body and "liabilit" in body and "equity" in body
    if expected == "cash":
        return all(cue in body for cue in ("operating", "investing", "financing"))
    return "r&d ratio" in body or ("research and development" in body and "%" in body)


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet")
    predictions = [
        json.loads(line)["answer"]
        for line in (ROOT / "artifacts/polyfiqa_easy_july/predictions_with_retrieval.jsonl").read_text().splitlines()
    ]
    families = [family(question) for question in frame["question"]]
    years = frame["task_id"].str.extract(r"_(\d{4})", expand=False).astype(int)
    usable = [year > 2021 and aligned(answer, name) for year, answer, name in zip(years, frame["answer"], families)]
    score = rouge1_f1(
        frame.loc[usable, "answer"].tolist(), [p for p, keep in zip(predictions, usable) if keep]
    )
    print(f"aligned_rows={sum(usable)} rouge1_f1={score:.6f}")
    for name in sorted(set(families)):
        mask = [keep and observed == name for keep, observed in zip(usable, families)]
        score = rouge1_f1(
            frame.loc[mask, "answer"].tolist(), [p for p, keep in zip(predictions, mask) if keep]
        )
        print(f"{name}_rows={sum(mask)} rouge1_f1={score:.6f}")


if __name__ == "__main__":
    main()
