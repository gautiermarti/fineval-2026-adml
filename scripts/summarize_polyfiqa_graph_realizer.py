#!/usr/bin/env python3
"""Aggregate four leakage-controlled graph-realizer folds.

The realizer receives a historical reference-oracle slot/evidence plan, so this
is a bounded calibration diagnostic rather than an end-to-end competition
result.  We nevertheless score complete answers by replacing only the frozen
deterministic core, preserving its independently selected evidence, applying
the same 100-word cap, and falling back on the baseline after any marker error.
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
from fineval.polyfiqa_edits import join_answer, split_answer  # noqa: E402
from fineval.polyfiqa_family_gate import exact_cluster_bootstrap  # noqa: E402
from fineval.validation import validate_answer  # noqa: E402
from predict_polyfiqa_hybrid import cap_generated_answer  # noqa: E402


COMPANIES = ("HON", "JNJ", "MSFT", "UVV")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--companies", nargs="+", choices=list(COMPANIES),
        default=list(COMPANIES),
        help="Fold directories to aggregate; use a subset only for a pilot audit",
    )
    parser.add_argument(
        "--easy-baseline", type=Path,
        default=ROOT / "artifacts/polyfiqa_easy_july/hybrid_input_only_cjk_budget_v15.jsonl",
    )
    parser.add_argument(
        "--expert-baseline", type=Path,
        default=ROOT / "artifacts/polyfiqa_expert_july/predictions_margin_gross_v13.jsonl",
    )
    args = parser.parse_args()

    baselines = {
        "easy": load_jsonl(args.easy_baseline),
        "expert": load_jsonl(args.expert_baseline),
    }
    summaries = []
    predictions = []
    companies = tuple(dict.fromkeys(args.companies))
    for company in companies:
        fold_dir = args.run_dir / company
        summaries.append(json.loads((fold_dir / "summary.json").read_text()))
        fold_rows = load_jsonl(fold_dir / "evaluation_predictions.jsonl")
        if any(row["held_company"] != company for row in fold_rows):
            raise ValueError(f"{company}: prediction fold identity mismatch")
        predictions.extend(fold_rows)

    immutable = {
        (row["model"], row["model_revision"], row["corpus_dir"])
        for row in summaries
    }
    if len(immutable) != 1:
        raise ValueError("folds use different model revisions or corpora")
    if {row["held_company"] for row in summaries} != set(companies):
        raise ValueError("held-company summaries do not match the requested folds")

    frames: dict[str, pd.DataFrame] = {}
    audit = []
    for row in predictions:
        tier = str(row["tier"])
        index = int(row["source_index"])
        baseline = str(baselines[tier][index]["answer"])
        baseline_core, baseline_evidence = split_answer(baseline)
        source = str(row["source"])
        if source not in frames:
            frames[source] = pd.read_parquet(
                source, columns=["task_id", "query", "answer"],
            )
        source_row = frames[source].iloc[index]
        if str(source_row.task_id) != str(row["task_id"]):
            raise ValueError("historical row alignment mismatch")
        reference = str(source_row.answer)
        markers_valid = bool(row["markers_valid"] and row["restored"])
        uncapped = (
            join_answer(str(row["restored"]), baseline_evidence)
            if markers_valid else baseline
        )
        attempted_candidate = cap_generated_answer(uncapped, 100)
        attempted_validation = validate_answer(
            attempted_candidate, str(source_row.query), tier,
        )
        valid = markers_valid and attempted_validation.valid
        candidate = attempted_candidate
        if not valid:
            candidate = baseline
        baseline_score = rouge1_f1([reference], [baseline])
        candidate_score = rouge1_f1([reference], [candidate])
        baseline_core_score = rouge1_f1([row["reference_core"]], [baseline_core])
        candidate_core_score = (
            rouge1_f1([row["reference_core"]], [str(row["restored"])])
            if valid else baseline_core_score
        )
        audit.append({
            "task_id": row["task_id"],
            "company": row["held_company"],
            "tier": tier,
            "family": row["family"],
            "source_index": index,
            "markers_valid": bool(row["markers_valid"]),
            "used_realizer": valid,
            "attempted_valid_structure": attempted_validation.valid_structure,
            "attempted_within_word_limit": attempted_validation.within_word_limit,
            "attempted_unsupported_numbers": list(
                attempted_validation.unsupported_numbers
            ),
            "uncapped_words": len(uncapped.split()),
            "candidate_was_capped": attempted_candidate != uncapped,
            "baseline_core_score": baseline_core_score,
            "candidate_core_score": candidate_core_score,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "delta": candidate_score - baseline_score,
            "candidate": candidate,
            "reference": reference,
        })

    expected_rows = sum(int(row["evaluation_rows"]) for row in summaries)
    if len(audit) != expected_rows:
        raise ValueError(f"expected {expected_rows} predictions, received {len(audit)}")

    by_family = {}
    for family in sorted({row["family"] for row in audit}):
        subset = [row for row in audit if row["family"] == family]
        by_family[family] = {
            "rows": len(subset),
            "valid_realizations": sum(row["used_realizer"] for row in subset),
            "baseline_score": mean(subset, "baseline_score"),
            "candidate_score": mean(subset, "candidate_score"),
            "gain": mean(subset, "delta"),
        }
    by_company = {}
    for company in companies:
        subset = [row for row in audit if row["company"] == company]
        by_company[company] = {
            "rows": len(subset),
            "valid_realizations": sum(row["used_realizer"] for row in subset),
            "baseline_score": mean(subset, "baseline_score"),
            "candidate_score": mean(subset, "candidate_score"),
            "gain": mean(subset, "delta"),
        }

    model, revision, corpus = next(iter(immutable))
    summary = {
        "evaluation": (
            "four-fold leave-company-out historical complete answers"
            if set(companies) == set(COMPANIES)
            else "partial leave-company-out pilot on historical complete answers"
        ),
        "evaluated_companies": list(companies),
        "four_fold_confirmation": set(companies) == set(COMPANIES),
        "model": model,
        "model_revision": revision,
        "corpus_dir": corpus,
        "rows": len(audit),
        "valid_realizations": sum(row["used_realizer"] for row in audit),
        "capped_candidates": sum(row["candidate_was_capped"] for row in audit),
        "baseline_core_rouge1": mean(audit, "baseline_core_score"),
        "candidate_core_rouge1": mean(audit, "candidate_core_score"),
        "baseline_complete_rouge1": mean(audit, "baseline_score"),
        "candidate_complete_rouge1": mean(audit, "candidate_score"),
        "complete_answer_gain": mean(audit, "delta"),
        "improved_rows": sum(row["delta"] > 0 for row in audit),
        "harmed_rows": sum(row["delta"] < 0 for row in audit),
        "tied_rows": sum(row["delta"] == 0 for row in audit),
        "company_cluster_bootstrap": exact_cluster_bootstrap(
            audit, cluster_key="company", delta_key="delta",
        ),
        "by_company": by_company,
        "by_family": by_family,
        "invalid_marker_policy": "fall back to frozen baseline",
        "realizer_only_oracle_plan": True,
        "end_to_end_competition_claim_allowed": False,
        "current_or_public_test_answers_used": False,
    }
    (args.run_dir / "aggregate_audit.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in audit
    ))
    (args.run_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
