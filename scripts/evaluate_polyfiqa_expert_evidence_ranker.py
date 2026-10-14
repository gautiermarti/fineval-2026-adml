#!/usr/bin/env python3
"""Company-held-out ranker for deciding whether one Expert news sentence helps."""

from pathlib import Path
import json
import re
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from baseline_polyfiqa_expert import (  # noqa: E402
    KNOWN_MISALIGNED_BUNDLES, capital_allocation_answer, capex_answer,
    margin_answer, revenue_focus_answer,
)


def family(question: str) -> str:
    if question.startswith("Please list"):
        return "revenue_focus"
    if question.startswith("How is"):
        return "capital_allocation"
    if question.startswith("What is the company’s strategy"):
        return "margin"
    return "capex"


def add_sentence(answer: str, sentence: str) -> str:
    return answer.replace("\nFinancial Statement Evidence:", f" {sentence}\nFinancial Statement Evidence:")


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/polyfiqa_expert/test-00000-of-00001.parquet")
    frame = frame[~frame["task_id"].isin(KNOWN_MISALIGNED_BUNDLES)].copy()
    frame["family"] = frame["question"].map(family)
    frame["company"] = frame["task_id"].str.split("_").str[0]
    news = {
        record["task_id"]: record["families"]
        for record in map(
            json.loads,
            (ROOT / "artifacts/polyfiqa_expert/news_candidates.jsonl").read_text().splitlines(),
        )
    }
    revenue = {
        record["task_id"]: record["candidates"]
        for record in map(
            json.loads,
            (ROOT / "artifacts/polyfiqa_retrieval/revenue_candidates.jsonl").read_text().splitlines(),
        )
    }
    examples, bases = [], {}
    for row_index, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        if row["family"] == "revenue_focus":
            base = revenue_focus_answer(row["task_id"], context.financial_statements, revenue[row["task_id"]])
        elif row["family"] == "capital_allocation":
            base = capital_allocation_answer(row["task_id"], context.financial_statements)
        elif row["family"] == "margin":
            base = margin_answer(row["task_id"], context.financial_statements)
        else:
            base = capex_answer(row["task_id"], context.financial_statements, row["question"])
        bases[row_index] = base
        base_score = rouge1_f1([row["answer"]], [base])
        for rank, candidate in enumerate(news[row["task_id"]][row["family"]]):
            text = candidate["text"]
            prediction = add_sentence(base, text)
            examples.append({
                "row_index": row_index, "company": row["company"], "family": row["family"],
                "language": candidate["language"], "rank": rank,
                "semantic_score": candidate["score"], "characters": len(text),
                "words": len(text.split()), "numbers": len(re.findall(r"\d", text)),
                "percent": int("%" in text), "currency": int("$" in text),
                "target_delta": rouge1_f1([row["answer"]], [prediction]) - base_score,
                "prediction": prediction,
            })
    examples = pd.DataFrame(examples)
    numeric = ["rank", "semantic_score", "characters", "words", "numbers", "percent", "currency"]
    categorical = ["family", "language"]
    transformer = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    selected = {}
    for held_out in sorted(frame["company"].unique()):
        train = examples["company"] != held_out
        test = ~train
        model = make_pipeline(
            transformer,
            HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=10, l2_regularization=1.0, random_state=2026),
        )
        model.fit(examples.loc[train, numeric + categorical], examples.loc[train, "target_delta"])
        test_examples = examples.loc[test].copy()
        test_examples["predicted_delta"] = model.predict(test_examples[numeric + categorical])
        for row_index, group in test_examples.groupby("row_index"):
            best = group.loc[group["predicted_delta"].idxmax()]
            selected[row_index] = best["prediction"] if best["predicted_delta"] > 0 else bases[row_index]
    predictions = [selected[index] for index in frame.index]
    base_predictions = [bases[index] for index in frame.index]
    print(f"base_clean_rouge1_f1={rouge1_f1(frame['answer'].tolist(), base_predictions):.6f}")
    print(f"ranked_clean_rouge1_f1={rouge1_f1(frame['answer'].tolist(), predictions):.6f}")
    for name, group in frame.groupby("family"):
        print(name, f"base={rouge1_f1(group['answer'].tolist(), [bases[i] for i in group.index]):.6f}",
              f"ranked={rouge1_f1(group['answer'].tolist(), [selected[i] for i in group.index]):.6f}")


if __name__ == "__main__":
    main()
