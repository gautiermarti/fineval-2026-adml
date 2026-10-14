#!/usr/bin/env python3
"""CPU grid over recent revenue evidence budgets and safe trend vocabulary."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.finance import extract_revenue  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from baseline_polyfiqa_revenue_rd import revenue_answer  # noqa: E402
from evaluate_polyfiqa_easy_july import aligned, family  # noqa: E402
from predict_polyfiqa import attach_answer_phrase, attach_revenue_evidence  # noqa: E402


PHRASES = {
    "none": "",
    "upward": "Overall, revenue has shown a consistent upward trend with steady year-over-year growth.",
    "drivers": "Revenue trends show steady growth, driven by strong performance across key businesses.",
    "trajectory": "Total revenue continued its positive growth trajectory during the period.",
    "mixed": "Overall revenue trends reflect growth across products, services, segments, and markets.",
}


def load(path: Path) -> dict[str, list[dict]]:
    return {r["task_id"]: r["candidates"] for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet")
    years = frame.task_id.str.extract(r"_(\d{4})", expand=False).astype(int)
    families = frame.question.map(family)
    mask = [
        year > 2021 and fam == "revenue" and aligned(gold, fam)
        for year, fam, gold in zip(years, families, frame.answer)
    ]
    data = frame.loc[mask].copy()
    candidates = load(ROOT / "artifacts/polyfiqa_easy_july/revenue_candidates.jsonl")
    bases = []
    for _, row in data.iterrows():
        context = parse_polyfiqa_query(row.query)
        bases.append(revenue_answer(row.task_id, extract_revenue(context.financial_statements)))
    rows = []
    for evidence_count in range(5):
        for phrase_name, phrase in PHRASES.items():
            predictions = []
            for base, task_id in zip(bases, data.task_id):
                answer = attach_answer_phrase(base, phrase) if phrase else base
                answer = attach_revenue_evidence(answer, candidates.get(task_id, []), evidence_count)
                predictions.append(answer)
            rows.append({
                "evidence_count": evidence_count, "phrase": phrase_name,
                "rouge1_f1": rouge1_f1(data.answer.tolist(), predictions),
                "changed": sum(a != b for a, b in zip(bases, predictions)),
                "max_words": max(len(answer.split()) for answer in predictions),
                **{
                    f"{company}_rouge1_f1": rouge1_f1(
                        data.loc[data.task_id.str.startswith(company), "answer"].tolist(),
                        [p for p, task in zip(predictions, data.task_id) if task.startswith(company)],
                    )
                    for company in ("MSFT", "JNJ", "HON", "UVV")
                },
            })
    result = pd.DataFrame(rows).sort_values("rouge1_f1", ascending=False)
    out = ROOT / "artifacts/polyfiqa_easy_july/revenue_realization_ablation.csv"
    result.to_csv(out, index=False)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
