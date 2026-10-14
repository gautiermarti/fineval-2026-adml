#!/usr/bin/env python3
"""Test generic, evidence-neutral template phrases on aligned July references."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import rouge1_f1  # noqa: E402
from evaluate_polyfiqa_easy_july import aligned as easy_aligned, family as easy_family  # noqa: E402
from evaluate_polyfiqa_expert_july import aligned as expert_aligned, family as expert_family  # noqa: E402


CANDIDATES = {
    "easy:revenue": [
        "Overall, revenue has shown a consistent upward trend with steady year-over-year growth.",
        "Revenue trends show steady growth, driven by strong performance across key businesses.",
        "Total revenue continued its positive growth trajectory during the period.",
    ],
    "easy:balance": [
        "This reflects the company's liquidity, leverage, and overall financial health.",
        "Current assets indicate liquidity, while the liabilities-to-equity ratio reflects leverage and financial health.",
        "The balance sheet shows its liquidity position and degree of financial leverage.",
    ],
    "easy:cash": [
        "These significant changes reflect shifts in operating activities, investments or acquisitions, debt, and shareholder returns.",
        "The changes indicate substantial movements in operations, investment activity, and financing decisions.",
        "The cash flows show significant year-over-year changes across operating, investing, and financing activities.",
    ],
    "easy:rd": [
        "This ratio indicates the company's level of investment in research and development.",
        "The R&D ratio reflects continued investment in research, development, and innovation.",
        "This indicates the share of revenue invested in research and development.",
    ],
    "expert:revenue": [
        "The company has diversified revenue streams across products, services, and business segments.",
        "The revenue focus is on growth across key products, services, segments, and markets.",
        "These focuses support diversified revenue growth and business performance.",
    ],
    "expert:allocation": [
        "Overall, capital is allocated among investments, acquisitions, shareholder returns, and debt management.",
        "This allocation balances strategic investment with dividends, share repurchases, and financial flexibility.",
        "The approach supports long-term growth while returning capital to shareholders.",
    ],
    "expert:margin": [
        "The strategy maintains profit margins through revenue growth, pricing, cost control, operational efficiency, and a higher-margin mix.",
        "The company focuses on productivity, cost management, pricing, innovation, and higher-margin products and services.",
        "These measures support stable profitability despite cost pressures.",
    ],
    "expert:capex": [
        "These strategic investments support long-term growth, technology, infrastructure, operational capabilities, and business expansion.",
        "The expenditures expand capacity and capabilities while supporting strategic growth initiatives.",
        "Their strategic significance is investment in infrastructure, technology, and future growth.",
    ],
}

DEPLOYED = {
    "easy:revenue": CANDIDATES["easy:revenue"][0],
    "easy:balance": CANDIDATES["easy:balance"][1],
    "easy:cash": CANDIDATES["easy:cash"][0],
    "expert:revenue": CANDIDATES["expert:revenue"][1],
    "expert:margin": CANDIDATES["expert:margin"][0],
    "expert:capex": CANDIDATES["expert:capex"][0],
}


def insert_phrase(answer: str, phrase: str) -> str:
    for marker in ("Financial Statement Evidence:", "News Evidence:"):
        if marker in answer:
            body, evidence = answer.split(marker, 1)
            candidate = f"{body.rstrip()} {phrase}\n{marker}{evidence}"
            return candidate if len(candidate.split()) <= 100 else answer
    candidate = f"{answer.rstrip()} {phrase}"
    return candidate if len(candidate.split()) <= 100 else answer


def evaluate(name: str, data_path: Path, prediction_path: Path, family_fn, aligned_fn) -> list[dict]:
    frame = pd.read_parquet(data_path)
    predictions = [json.loads(line)["answer"] for line in prediction_path.read_text().splitlines()]
    years = frame["task_id"].str.extract(r"_(\d{4})", expand=False).astype(int).to_numpy()
    families = [family_fn(question) for question in frame["question"]]
    rows = []
    for family in sorted(set(families)):
        base_mask = np.asarray([
            observed == family and aligned_fn(answer, observed)
            for observed, answer in zip(families, frame["answer"])
        ])
        if name == "easy":
            base_mask &= years > 2021
        family_predictions = [
            answer.replace(f" {DEPLOYED.get(f'{name}:{family}', '')}", "")
            if DEPLOYED.get(f"{name}:{family}") else answer
            for answer in predictions
        ]
        variants = [("baseline", "")] + [
            (f"phrase_{index}", phrase)
            for index, phrase in enumerate(CANDIDATES.get(f"{name}:{family}", []), 1)
        ]
        for variant, phrase in variants:
            revised = [
                insert_phrase(answer, phrase) if phrase else answer
                for answer in family_predictions
            ]
            for period, period_mask in (
                ("all", np.ones(len(frame), dtype=bool)),
                ("old", years <= 2021),
                ("new", years > 2021),
            ):
                mask = base_mask & period_mask
                if not mask.any():
                    continue
                rows.append({
                    "dataset": name, "family": family, "period": period,
                    "variant": variant, "phrase": phrase, "rows": int(mask.sum()),
                    "rouge1_f1": rouge1_f1(
                        frame.loc[mask, "answer"].tolist(),
                        [answer for answer, keep in zip(revised, mask) if keep],
                    ),
                    "changed": sum(
                        original != changed for original, changed, keep
                        in zip(family_predictions, revised, mask) if keep
                    ),
                })
    return rows


def main() -> None:
    rows = []
    rows.extend(evaluate(
        "easy", ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
        ROOT / "artifacts/polyfiqa_easy_july/predictions_with_retrieval.jsonl",
        easy_family, easy_aligned,
    ))
    rows.extend(evaluate(
        "expert", ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
        ROOT / "artifacts/polyfiqa_expert_july/predictions.jsonl",
        expert_family, expert_aligned,
    ))
    results = pd.DataFrame(rows)
    baseline = results[results["variant"] == "baseline"][
        ["dataset", "family", "period", "rouge1_f1"]
    ].rename(columns={"rouge1_f1": "baseline"})
    results = results.merge(baseline, on=["dataset", "family", "period"])
    results["gain"] = results["rouge1_f1"] - results["baseline"]
    output = ROOT / "artifacts/polyfiqa_error_atlas/template_phrase_ablation.csv"
    results.to_csv(output, index=False)
    print(results.sort_values("gain", ascending=False).groupby(
        ["dataset", "family", "period"], group_keys=False,
    ).head(2).to_string(index=False))


if __name__ == "__main__":
    import numpy as np
    main()
