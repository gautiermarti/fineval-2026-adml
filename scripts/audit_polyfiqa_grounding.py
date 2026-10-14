#!/usr/bin/env python3
"""Audit source attribution and numerical support in generated Task 3 answers."""

from pathlib import Path
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.finance import extract_revenue  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from baseline_polyfiqa_revenue_rd import revenue_answer  # noqa: E402


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask3/public-00000-of-00001.parquet")
    frame = frame[frame["question"].str.startswith("What trends")]
    retrieved = {
        record["task_id"]: record["candidates"]
        for record in map(
            json.loads,
            (ROOT / "artifacts/polyfiqa_retrieval/revenue_candidates.jsonl").read_text().splitlines(),
        )
    }
    rows = []
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        base = revenue_answer(row["task_id"], extract_revenue(context.financial_statements))
        source_news = normalized(" ".join(getattr(context, f"{lang}_news") for lang in (
            "english", "chinese", "japanese", "spanish", "greek"
        )))
        evidence = retrieved[row["task_id"]][:2]
        grounded = [normalized(item["text"]) in source_news for item in evidence]
        rows.append({
            "task_id": row["task_id"],
            "quoted_sentences": len(evidence),
            "grounded_sentences": sum(grounded),
            "all_quotes_grounded": all(grounded),
            "base_answer": base,
        })
    result = pd.DataFrame(rows)
    output = ROOT / "artifacts/polyfiqa_audit/revenue_grounding.csv"
    result.to_csv(output, index=False)
    print(result[["task_id", "quoted_sentences", "grounded_sentences", "all_quotes_grounded"]].to_string(index=False))
    print(f"grounded_quote_rate={result['grounded_sentences'].sum() / result['quoted_sentences'].sum():.6f}")


if __name__ == "__main__":
    main()
