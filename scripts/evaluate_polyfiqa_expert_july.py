#!/usr/bin/env python3
"""Evaluate only lexically family-aligned rows in the noisy Expert-July set."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.metrics import rouge1_f1  # noqa: E402


def family(question: str) -> str:
    if question.startswith("Please list"):
        return "revenue"
    if question.startswith("How is"):
        return "allocation"
    if question.startswith("What is the company"):
        return "margin"
    return "capex"


def aligned(reference: str, expected: str) -> bool:
    """Require explicit answer-body cues; evidence overlap alone is insufficient."""
    body = reference.casefold().split("financial statement", 1)[0]
    if expected == "revenue":
        return "revenue" in body and ("top three" in body or "focus" in body)
    if expected == "allocation":
        allocation = "allocat" in body or ("repurchase" in body and "dividend" in body)
        capex_only = "capital expenditure" in body and len(body.split()) < 35
        return allocation and not capex_only
    if expected == "margin":
        return "margin" in body and any(
            cue in body for cue in ("maintain", "strateg", "cost", "efficien", "focus")
        )
    return (
        ("capital expenditure" in body or "capex" in body)
        and any(cue in body for cue in ("strateg", "signific", "investment", "increased", "decreased", "totaled"))
    )


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet")
    records = [
        json.loads(line)
        for line in (ROOT / "artifacts/polyfiqa_expert_july/predictions.jsonl").read_text().splitlines()
    ]
    predictions = [record["answer"] for record in records]
    families = [family(question) for question in frame["question"]]
    usable = [aligned(answer, name) for answer, name in zip(frame["answer"], families)]
    years = frame["task_id"].str.extract(r"_(\d{4})", expand=False).astype(int)
    for split, period in (("all", years > 0), ("old", years <= 2021), ("new", years > 2021)):
        mask = [keep and include for keep, include in zip(usable, period)]
        score = rouge1_f1(
            frame.loc[mask, "answer"].tolist(), [prediction for prediction, keep in zip(predictions, mask) if keep]
        )
        print(f"{split}_rows={sum(mask)} {split}_rouge1_f1={score:.6f}")
        for name in sorted(set(families)):
            family_mask = [keep and observed == name for keep, observed in zip(mask, families)]
            if not any(family_mask):
                continue
            family_score = rouge1_f1(
                frame.loc[family_mask, "answer"].tolist(),
                [prediction for prediction, keep in zip(predictions, family_mask) if keep],
            )
            print(f"  {name}_rows={sum(family_mask)} rouge1_f1={family_score:.6f}")


if __name__ == "__main__":
    main()
