#!/usr/bin/env python3
"""Measure ROUGE contribution of top-k retrieved revenue evidence."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.finance import extract_revenue  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from baseline_polyfiqa_revenue_rd import revenue_answer  # noqa: E402


def base_answer(task_id: str, result) -> str:
    return revenue_answer(task_id, result).split("\nNews Evidence:", 1)[0]


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask3/public-00000-of-00001.parquet")
    frame = frame[frame["question"].str.startswith("What trends")]
    candidates = {
        row["task_id"]: row["candidates"]
        for row in (json.loads(line) for line in (ROOT / "artifacts/polyfiqa_retrieval/revenue_candidates.jsonl").read_text().splitlines())
    }
    bases = {}
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        bases[row["task_id"]] = base_answer(row["task_id"], extract_revenue(context.financial_statements))
    rows = []
    for top_k in range(0, 6):
        predictions = []
        for _, row in frame.iterrows():
            evidence = candidates[row["task_id"]][:top_k]
            evidence_text = "None." if not evidence else " ".join(
                f"{item['language'].title()} news: \"{item['text']}\"" for item in evidence
            )
            predictions.append(f"{bases[row['task_id']]}\nNews Evidence: {evidence_text}")
        score = rouge1_f1(frame["answer"].tolist(), predictions)
        rows.append({"top_k": top_k, "rouge1_f1": score})
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv(ROOT / "artifacts/polyfiqa_retrieval/revenue_topk_scores.csv", index=False)

    # Leave-one-company-out selection of top-k avoids choosing evidence length
    # on the same company references used for evaluation.
    frame = frame.assign(company=frame["task_id"].str.split("_").str[0])
    loco_rows = []
    for held_out in sorted(frame["company"].unique()):
        train = frame["company"] != held_out
        test = ~train
        train_scores = {}
        test_predictions_by_k = {}
        for top_k in range(0, 6):
            predictions = []
            for _, row in frame.iterrows():
                evidence = candidates[row["task_id"]][:top_k]
                evidence_text = "None." if not evidence else " ".join(
                    f"{item['language'].title()} news: \"{item['text']}\"" for item in evidence
                )
                predictions.append(f"{bases[row['task_id']]}\nNews Evidence: {evidence_text}")
            prediction_series = pd.Series(predictions, index=frame.index)
            train_scores[top_k] = rouge1_f1(
                frame.loc[train, "answer"].tolist(), prediction_series.loc[train].tolist()
            )
            test_predictions_by_k[top_k] = prediction_series.loc[test].tolist()
        chosen_k = max(train_scores, key=train_scores.get)
        test_score = rouge1_f1(
            frame.loc[test, "answer"].tolist(), test_predictions_by_k[chosen_k]
        )
        loco_rows.append(
            {
                "held_out_company": held_out,
                "selected_top_k": chosen_k,
                "training_rouge1_f1": train_scores[chosen_k],
                "test_rouge1_f1": test_score,
                "test_rows": int(test.sum()),
            }
        )
    loco = pd.DataFrame(loco_rows)
    print("leave_one_company_out")
    print(loco.to_string(index=False))
    print(f"loco_macro_rouge1_f1={loco['test_rouge1_f1'].mean():.6f}")
    loco.to_csv(ROOT / "artifacts/polyfiqa_retrieval/revenue_loco_scores.csv", index=False)


if __name__ == "__main__":
    main()
