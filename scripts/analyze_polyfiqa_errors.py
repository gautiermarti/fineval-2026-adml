#!/usr/bin/env python3
"""Build a family-level lexical error atlas for aligned PolyFiQA-July rows."""

from collections import Counter
from pathlib import Path
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import rouge1_f1  # noqa: E402
from evaluate_polyfiqa_easy_july import aligned as easy_aligned, family as easy_family  # noqa: E402
from evaluate_polyfiqa_expert_july import aligned as expert_aligned, family as expert_family  # noqa: E402

TOKEN = re.compile(r"[a-z]+(?:['&-][a-z]+)*|\d+(?:\.\d+)?%?", re.I)
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "from", "as", "is",
    "was", "were", "are", "with", "by", "on", "at", "that", "this", "it", "its",
    "answer", "evidence", "financial", "statement", "news", "company", "none",
}


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


def analyze_dataset(
    name: str, data_path: Path, prediction_path: Path, family_fn, aligned_fn,
) -> tuple[list[dict], list[dict]]:
    frame = pd.read_parquet(data_path)
    predictions = [json.loads(line)["answer"] for line in prediction_path.read_text().splitlines()]
    rows = []
    summaries = []
    families = [family_fn(question) for question in frame["question"]]
    years = frame["task_id"].str.extract(r"_(\d{4})", expand=False).astype(int)
    usable = [aligned_fn(answer, fam) for answer, fam in zip(frame["answer"], families)]
    if name == "easy":
        usable = [keep and year > 2021 for keep, year in zip(usable, years)]
    for index, (gold, prediction, fam, keep) in enumerate(
        zip(frame["answer"], predictions, families, usable)
    ):
        if not keep:
            continue
        gold_tokens = tokens(gold)
        prediction_tokens = tokens(prediction)
        gold_counts = Counter(gold_tokens)
        prediction_counts = Counter(prediction_tokens)
        overlap = gold_counts & prediction_counts
        missing = list((gold_counts - prediction_counts).elements())
        extra = list((prediction_counts - gold_counts).elements())
        rows.append({
            "dataset": name, "row_index": index, "task_id": frame.iloc[index]["task_id"],
            "family": fam, "year": int(years.iloc[index]),
            "rouge1_f1": rouge1_f1([gold], [prediction]),
            "gold_words": len(gold_tokens), "prediction_words": len(prediction_tokens),
            "overlap_words": sum(overlap.values()),
            "missing_tokens": " ".join(missing), "extra_tokens": " ".join(extra),
            "gold": gold, "prediction": prediction,
        })
    detail = pd.DataFrame(rows)
    for fam, group in detail.groupby("family"):
        missing = Counter()
        extra = Counter()
        for value in group["missing_tokens"]:
            missing.update(token for token in value.split() if token not in STOP and len(token) > 1)
        for value in group["extra_tokens"]:
            extra.update(token for token in value.split() if token not in STOP and len(token) > 1)
        summaries.append({
            "dataset": name, "family": fam, "rows": len(group),
            "rouge1_f1": rouge1_f1(group["gold"].tolist(), group["prediction"].tolist()),
            "mean_gold_words": group["gold_words"].mean(),
            "mean_prediction_words": group["prediction_words"].mean(),
            "top_missing": " | ".join(f"{term}:{count}" for term, count in missing.most_common(35)),
            "top_extra": " | ".join(f"{term}:{count}" for term, count in extra.most_common(35)),
        })
    return rows, summaries


def main() -> None:
    configurations = [
        (
            "easy", ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
            ROOT / "artifacts/polyfiqa_easy_july/predictions_with_retrieval.jsonl",
            easy_family, easy_aligned,
        ),
        (
            "expert", ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
            ROOT / "artifacts/polyfiqa_expert_july/predictions.jsonl",
            expert_family, expert_aligned,
        ),
    ]
    rows, summaries = [], []
    for configuration in configurations:
        dataset_rows, dataset_summaries = analyze_dataset(*configuration)
        rows.extend(dataset_rows)
        summaries.extend(dataset_summaries)
    output = ROOT / "artifacts/polyfiqa_error_atlas"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "aligned_row_errors.csv", index=False)
    summary = pd.DataFrame(summaries).sort_values(["dataset", "rouge1_f1"])
    summary.to_csv(output / "family_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
