#!/usr/bin/env python3
"""Measure top-k evidence contribution for deterministic cash-flow answers."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.finance import extract_cash_flows  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402

from baseline_polyfiqa_cashflow import realize  # noqa: E402


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask3/public-00000-of-00001.parquet")
    frame = frame[frame["question"].str.startswith("Are there any significant")].copy()
    candidates = {
        row["task_id"]: row["candidates"]
        for row in (json.loads(line) for line in (ROOT / "artifacts/polyfiqa_retrieval/cashflow_candidates.jsonl").read_text().splitlines())
    }
    bases = {}
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        bases[row["task_id"]] = realize(row["task_id"], extract_cash_flows(context.financial_statements)).split("\nNews Evidence:")[0]
    rows = []
    for top_k in range(0, 6):
        predictions = []
        for _, row in frame.iterrows():
            evidence = candidates[row["task_id"]][:top_k]
            evidence_text = "None." if not evidence else " ".join(
                f"{item['language'].title()} news: \"{item['text']}\"" for item in evidence
            )
            predictions.append(f"{bases[row['task_id']]}\nNews Evidence: {evidence_text}")
        rows.append({"top_k": top_k, "rouge1_f1": rouge1_f1(frame["answer"].tolist(), predictions)})
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv(ROOT / "artifacts/polyfiqa_retrieval/cashflow_topk_scores.csv", index=False)


if __name__ == "__main__":
    main()
