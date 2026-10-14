#!/usr/bin/env python3
"""Create leakage-resistant leave-one-company-out realizer manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/polyfiqa_graph_to_reference_v1",
    )
    args = parser.parse_args()

    originals = load_jsonl(args.artifact_dir / "training.jsonl")
    swaps = load_jsonl(args.artifact_dir / "counterfactual_training.jsonl")
    companies = sorted({row["company"] for row in originals})
    summary = {}
    for held in companies:
        fold_dir = args.artifact_dir / "loco" / held
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_original = [row for row in originals if row["company"] != held]
        train_swaps = [
            row for row in swaps
            if row["donor_company"] != held and row["target_company"] != held
        ]
        evaluation = [row for row in originals if row["company"] == held]
        training = [
            {"source_kind": "historical", **row} for row in train_original
        ] + train_swaps
        dump_jsonl(fold_dir / "train.jsonl", training)
        dump_jsonl(fold_dir / "evaluation.jsonl", evaluation)
        summary[held] = {
            "training_rows": len(training),
            "training_historical": len(train_original),
            "training_counterfactual": len(train_swaps),
            "evaluation_rows": len(evaluation),
            "evaluation_families": dict(Counter(
                row["plan"]["family"] for row in evaluation
            )),
            "held_company_as_original_donor_or_target_in_training": 0,
        }
    manifest = {
        "companies": companies,
        "folds": summary,
        "split_rule": (
            "held company excluded from original examples and from both donor and "
            "target sides of counterfactuals"
        ),
        "public_test_answers_used": False,
    }
    (args.artifact_dir / "loco_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
