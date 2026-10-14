#!/usr/bin/env python3
"""CPU ablations for evidence count and cash-flow evidence gates on Easy-July."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.finance import extract_cash_flows, extract_revenue  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from baseline_polyfiqa_cashflow import realize as cashflow_answer  # noqa: E402
from baseline_polyfiqa_revenue_rd import revenue_answer  # noqa: E402
from evaluate_polyfiqa_easy_july import aligned, family  # noqa: E402
from predict_polyfiqa import attach_revenue_evidence  # noqa: E402


def load(path: Path) -> dict[str, list[dict]]:
    return {
        record["task_id"]: record["candidates"]
        for record in map(json.loads, path.read_text().splitlines())
    }


def cash_with_gate(
    task_id: str, answer: str, evidence: list[dict], threshold: float,
    language: str, exclude_jnj: bool,
) -> str:
    year = int(task_id.split("_", 1)[1][:4])
    if year < 2024 or (exclude_jnj and task_id.startswith("JNJ")):
        return answer
    eligible = evidence if language == "any" else [
        item for item in evidence if item.get("language") == language
    ]
    if not eligible or eligible[0].get("semantic_score", 0.0) < threshold:
        return answer
    candidate = f"{eligible[0]['language'].title()} news: \"{eligible[0]['text']}\""
    prefix = answer.replace("News Evidence: None.", "News Evidence: ")
    return prefix + candidate if len((prefix + candidate).split()) <= 100 else answer


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet")
    revenue_candidates = load(ROOT / "artifacts/polyfiqa_easy_july/revenue_candidates.jsonl")
    cash_candidates = load(ROOT / "artifacts/polyfiqa_easy_july/cash_candidates.jsonl")
    years = frame["task_id"].str.extract(r"_(\d{4})", expand=False).astype(int)
    families = frame["question"].map(family)
    rows = []

    revenue_mask = np_mask = [
        year > 2021 and fam == "revenue" and aligned(answer, fam)
        for year, fam, answer in zip(years, families, frame["answer"])
    ]
    revenue_rows = frame.loc[revenue_mask]
    revenue_base = []
    for _, row in revenue_rows.iterrows():
        context = parse_polyfiqa_query(row["query"])
        revenue_base.append(revenue_answer(row["task_id"], extract_revenue(context.financial_statements)))
    for count in range(5):
        predictions = [
            attach_revenue_evidence(
                answer, revenue_candidates.get(task_id, []), max_evidence=count,
            ) if count else answer
            for answer, task_id in zip(revenue_base, revenue_rows["task_id"])
        ]
        rows.append({
            "family": "revenue", "variant": f"top_{count}", "rows": len(revenue_rows),
            "rouge1_f1": rouge1_f1(revenue_rows["answer"].tolist(), predictions),
        })

    cash_mask = [
        year > 2021 and fam == "cash" and aligned(answer, fam)
        for year, fam, answer in zip(years, families, frame["answer"])
    ]
    cash_rows = frame.loc[cash_mask]
    cash_base = []
    for _, row in cash_rows.iterrows():
        context = parse_polyfiqa_query(row["query"])
        cash_base.append(cashflow_answer(row["task_id"], extract_cash_flows(context.financial_statements)))
    for language in ("english", "any"):
        for exclude_jnj in (False, True):
          for threshold in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
            predictions = [
                cash_with_gate(
                    task_id, answer, cash_candidates.get(task_id, []), threshold, language,
                    exclude_jnj,
                )
                for answer, task_id in zip(cash_base, cash_rows["task_id"])
            ]
            rows.append({
                "family": "cash",
                "variant": f"{language}_{threshold:.1f}_exclude_jnj={exclude_jnj}",
                "rows": len(cash_rows),
                "rouge1_f1": rouge1_f1(cash_rows["answer"].tolist(), predictions),
            })
    output = ROOT / "artifacts/polyfiqa_easy_july/retrieval_ablation.csv"
    results = pd.DataFrame(rows).sort_values(["family", "rouge1_f1"], ascending=[True, False])
    results.to_csv(output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
