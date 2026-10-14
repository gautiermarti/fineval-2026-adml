#!/usr/bin/env python3
"""Test whether the revenue-focus compiler transfers to complete answers.

The upstream compiler diagnostic scores only the answer core against a
delexicalized historical core.  This stricter control replaces the core of the
current deterministic input-only answer while preserving its independently
generated evidence section, then applies the official item-level ROUGE-1
implementation to the complete historical answer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.polyfiqa_family_gate import exact_cluster_bootstrap  # noqa: E402
from predict_polyfiqa import predict  # noqa: E402


COMPANIES = ("HON", "JNJ", "MSFT", "UVV")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v5",
    )
    parser.add_argument(
        "--compiler-predictions", type=Path,
        default=ROOT / "artifacts/polyfiqa_family_compiler_v1/planned_predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_family_compiler_full_answer_v1",
    )
    args = parser.parse_args()

    targets = {}
    for held_company in COMPANIES:
        for target in load_jsonl(
            args.corpus_dir / "loco" / held_company / "evaluation.jsonl"
        ):
            key = (
                held_company, target["plan"]["task_id"],
                target["plan"]["family"],
            )
            if key in targets:
                raise AssertionError(f"non-unique historical target key: {key}")
            targets[key] = target

    frame_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for compiler in load_jsonl(args.compiler_predictions):
        if compiler["family"] != "revenue_focus":
            continue
        key = (
            compiler["held_company"], compiler["task_id"], compiler["family"],
        )
        target = targets[key]
        source = str(target["source"])
        if source not in frame_cache:
            frame_cache[source] = pd.read_parquet(source)
        source_row = frame_cache[source].iloc[int(target["source_index"])].copy()
        reference = str(source_row["answer"])
        baseline = predict(source_row, {}, {})
        heading = "Financial Statement Evidence:"
        if heading not in baseline or heading not in reference:
            raise AssertionError(f"missing expert evidence heading for {key}")
        evidence = baseline.split(heading, 1)[1].strip()
        compiled = (
            compiler["nested_prediction"].rstrip()
            + f"\n{heading} " + evidence
        )
        baseline_score = rouge1_f1([reference], [baseline])
        compiled_score = rouge1_f1([reference], [compiled])
        rows.append({
            "held_company": compiler["held_company"],
            "task_id": compiler["task_id"],
            "family": compiler["family"],
            "nested_route": compiler["nested_route"],
            "baseline_answer": baseline,
            "compiled_answer": compiled,
            "reference": reference,
            "baseline_score": baseline_score,
            "compiled_score": compiled_score,
            "delta": compiled_score - baseline_score,
        })

    by_company = {}
    for company in sorted({row["held_company"] for row in rows}):
        subset = [row for row in rows if row["held_company"] == company]
        by_company[company] = {
            "rows": len(subset),
            "baseline_rouge": mean(subset, "baseline_score"),
            "compiled_rouge": mean(subset, "compiled_score"),
            "delta": mean(subset, "delta"),
        }
    bootstrap = exact_cluster_bootstrap(
        rows, cluster_key="held_company", delta_key="delta",
    )
    summary = {
        "evaluation": "complete historical answers with official item-level ROUGE-1",
        "scope": "fact-only revenue-focus rows covered by input-only planner replay",
        "rows": len(rows),
        "baseline_rouge": mean(rows, "baseline_score"),
        "compiled_rouge": mean(rows, "compiled_score"),
        "delta": mean(rows, "delta"),
        "company_cluster_bootstrap": bootstrap,
        "by_company": by_company,
        "decision": "reject_complete_answer_replacement",
        "historical_reference_used_for_prediction": False,
        "current_or_public_test_answers_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
