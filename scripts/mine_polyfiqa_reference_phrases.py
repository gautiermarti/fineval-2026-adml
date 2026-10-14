#!/usr/bin/env python3
"""Company-held-out test of generic reference-derived answer vocabulary."""

from collections import Counter
from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import POLYFIQA_TOKEN_RE, rouge1_f1  # noqa: E402
from ablate_polyfiqa_template_phrases import insert_phrase  # noqa: E402
from evaluate_polyfiqa_easy_july import family as easy_family  # noqa: E402
from evaluate_polyfiqa_expert_july import family as expert_family  # noqa: E402


def candidates(references: list[str], companies: list[str]) -> list[str]:
    support: dict[tuple[str, ...], set[str]] = {}
    frequency = Counter()
    for answer, company in zip(references, companies):
        # Mine only alphabetic English spans; figures and copied evidence are
        # deliberately excluded to keep phrases report-independent.
        tokens = [token.lower() for token in POLYFIQA_TOKEN_RE.findall(answer)]
        finance_words = {
            "revenue", "sales", "growth", "capital", "investment", "investments",
            "cash", "flow", "financial", "company", "margin", "margins", "operating",
            "expenditures", "strategy", "strategic", "dividends", "repurchases",
            "assets", "liabilities", "equity", "research", "development",
        }
        observed = set()
        for width in range(3, 9):
            for start in range(len(tokens) - width + 1):
                ngram = tuple(tokens[start:start + width])
                if not all(re.fullmatch(r"[a-z]+", token) for token in ngram):
                    continue
                if not finance_words.intersection(ngram):
                    continue
                if ngram in observed:
                    continue
                observed.add(ngram)
                support.setdefault(ngram, set()).add(company)
                frequency[ngram] += 1
    ranked = sorted(
        (ngram for ngram in frequency if len(support[ngram]) >= 3),
        key=lambda x: (len(support[x]), frequency[x], len(x)), reverse=True,
    )
    return [" ".join(ngram).capitalize() + "." for ngram in ranked[:250]]


def analyze(tier: str, data_path: Path, prediction_path: Path, family_fn) -> list[dict]:
    frame = pd.read_parquet(data_path).reset_index(drop=True)
    predictions = [json.loads(line)["answer"] for line in prediction_path.read_text().splitlines()]
    frame = frame.assign(
        prediction=predictions,
        family=frame.question.map(family_fn),
        company=frame.task_id.str.split("_", n=1).str[0],
    )
    rows = []
    for family, part in frame.groupby("family"):
        for held_out in sorted(part.company.unique()):
            train = part.company != held_out
            test = ~train
            phrases = candidates(
                part.loc[train, "answer"].tolist(), part.loc[train, "company"].tolist(),
            )
            baseline_train = rouge1_f1(
                part.loc[train, "answer"].tolist(), part.loc[train, "prediction"].tolist(),
            )
            scored = []
            for phrase in phrases:
                revised = [insert_phrase(answer, phrase) for answer in part.loc[train, "prediction"]]
                score = rouge1_f1(part.loc[train, "answer"].tolist(), revised)
                scored.append((score - baseline_train, -len(phrase.split()), phrase))
            gain, _, phrase = max(scored, default=(0.0, 0, ""))
            baseline_test = rouge1_f1(
                part.loc[test, "answer"].tolist(), part.loc[test, "prediction"].tolist(),
            )
            revised_test = [
                insert_phrase(answer, phrase) if gain > 0 else answer
                for answer in part.loc[test, "prediction"]
            ]
            test_score = rouge1_f1(part.loc[test, "answer"].tolist(), revised_test)
            rows.append({
                "tier": tier, "family": family, "held_out_company": held_out,
                "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                "phrase": phrase if gain > 0 else "", "train_gain": max(gain, 0.0),
                "test_baseline": baseline_test, "test_score": test_score,
                "test_gain": test_score - baseline_test,
            })
    return rows


def main() -> None:
    rows = analyze(
        "easy", ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
        ROOT / "artifacts/polyfiqa_system/easy_july_input_only_diagnostic.jsonl", easy_family,
    )
    rows += analyze(
        "expert", ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
        ROOT / "artifacts/polyfiqa_system/expert_july_input_only_diagnostic.jsonl", expert_family,
    )
    result = pd.DataFrame(rows)
    output = ROOT / "artifacts/polyfiqa_error_atlas/company_heldout_phrase_mining.csv"
    result.to_csv(output, index=False)
    summary = result.groupby(["tier", "family"]).agg(
        folds=("held_out_company", "size"), mean_test_gain=("test_gain", "mean"),
        positive_folds=("test_gain", lambda x: int(np.sum(x > 0))),
        negative_folds=("test_gain", lambda x: int(np.sum(x < 0))),
    )
    print(summary.to_string())
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
