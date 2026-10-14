#!/usr/bin/env python3
"""Measure whether template-phrase gains are broad or driven by a few rows."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import rouge1_f1  # noqa: E402
from ablate_polyfiqa_template_phrases import CANDIDATES, insert_phrase  # noqa: E402
from evaluate_polyfiqa_easy_july import aligned as easy_aligned, family as easy_family  # noqa: E402
from evaluate_polyfiqa_expert_july import aligned as expert_aligned, family as expert_family  # noqa: E402


def analyze(name: str, data: Path, predictions_path: Path, family_fn, aligned_fn) -> list[dict]:
    frame = pd.read_parquet(data)
    predictions = [json.loads(line)["answer"] for line in predictions_path.read_text().splitlines()]
    years = frame["task_id"].str.extract(r"_(\d{4})", expand=False).astype(int)
    families = frame["question"].map(family_fn)
    rows = []
    for family in sorted(families.unique()):
        candidates = CANDIDATES.get(f"{name}:{family}", [])
        for candidate_index, phrase in enumerate(candidates, 1):
            deltas = []
            companies = []
            for index, row in frame.iterrows():
                if families.iloc[index] != family or not aligned_fn(row["answer"], family):
                    continue
                if name == "easy" and years.iloc[index] <= 2021:
                    continue
                revised = insert_phrase(predictions[index], phrase)
                baseline = rouge1_f1([row["answer"]], [predictions[index]])
                changed = rouge1_f1([row["answer"]], [revised])
                deltas.append(changed - baseline)
                companies.append(row["task_id"].split("_", 1)[0])
            delta = np.asarray(deltas)
            rows.append({
                "dataset": name, "family": family, "candidate": candidate_index,
                "phrase": phrase, "rows": len(delta),
                "wins": int((delta > 1e-12).sum()), "ties": int((abs(delta) <= 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
                "mean_row_gain": float(delta.mean()), "median_row_gain": float(np.median(delta)),
                "positive_companies": len(set(company for company, gain in zip(companies, delta) if gain > 0)),
                "negative_companies": len(set(company for company, gain in zip(companies, delta) if gain < 0)),
            })
    return rows


def main() -> None:
    rows = analyze(
        "easy", ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
        ROOT / "artifacts/polyfiqa_easy_july/predictions_with_retrieval.jsonl",
        easy_family, easy_aligned,
    )
    rows += analyze(
        "expert", ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
        ROOT / "artifacts/polyfiqa_expert_july/predictions.jsonl",
        expert_family, expert_aligned,
    )
    results = pd.DataFrame(rows).sort_values(
        ["mean_row_gain", "wins"], ascending=False,
    )
    output = ROOT / "artifacts/polyfiqa_error_atlas/template_phrase_robustness.csv"
    results.to_csv(output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
