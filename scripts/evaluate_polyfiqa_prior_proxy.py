#!/usr/bin/env python3
"""Evaluate predictions only where a prior input packet matches public references."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_polyfiqa_hidden_overlap import fingerprint  # noqa: E402
from evaluate_polyfiqa_easy_july import family as easy_family  # noqa: E402
from evaluate_polyfiqa_expert_july import family as expert_family  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from predict_polyfiqa_hybrid import is_easy  # noqa: E402


REFERENCES = {
    "easy": ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
    "expert": ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/finmmeval_task2/task2_test_questions.jsonl",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    inputs = load_jsonl(args.input)
    predictions = load_jsonl(args.predictions)
    if len(inputs) != len(predictions):
        raise ValueError("input and prediction counts differ")
    references = {}
    for tier, path in REFERENCES.items():
        for row in pd.read_parquet(path).itertuples():
            references[(tier, fingerprint(row.query, row.question))] = row.answer

    rows = []
    for item, prediction in zip(inputs, predictions, strict=True):
        tier = "easy" if is_easy(item["question"]) else "expert"
        reference = references.get((tier, fingerprint(item["query"], item["question"])))
        if reference is None:
            continue
        family_fn = easy_family if tier == "easy" else expert_family
        rows.append({
            "tier": tier,
            "family": family_fn(item["question"]),
            "reference": reference,
            "prediction": prediction["answer"],
        })
    frame = pd.DataFrame(rows)
    print(
        f"matched_rows={len(frame)} "
        f"rouge1_f1={rouge1_f1(frame.reference.tolist(), frame.prediction.tolist()):.6f}"
    )
    for (tier, family), group in frame.groupby(["tier", "family"], sort=True):
        print(
            f"{tier}_{family}_rows={len(group)} "
            f"rouge1_f1={rouge1_f1(group.reference.tolist(), group.prediction.tolist()):.6f}"
        )


if __name__ == "__main__":
    main()
