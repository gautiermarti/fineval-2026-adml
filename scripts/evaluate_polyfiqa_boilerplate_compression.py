#!/usr/bin/env python3
"""LOCO evaluation of exact delexicalized boilerplate deletion around v15."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_polyfiqa_candidate_lattice import candidate_valid  # noqa: E402
from fineval.polyfiqa_edits import sentences, split_answer  # noqa: E402
from fineval.polyfiqa_family_gate import exact_cluster_bootstrap  # noqa: E402


COMPANIES = ("HON", "JNJ", "MSFT", "UVV")
ISSUER_PATTERN = re.compile(
    r"\b(?:microsoft|honeywell|johnson\s*&\s*johnson|"
    r"universal(?:\s+corporation)?)\b",
    flags=re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"\$?\(?[-+]?\d[\d,.]*(?:\s*(?:%|billion|million|thousand|b|m))?\)?",
    flags=re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_sentence(sentence: str) -> str:
    value = ISSUER_PATTERN.sub("<company>", sentence.lower())
    value = NUMBER_PATTERN.sub("<num>", value)
    return re.sub(r"\s+", " ", value).strip()


def removed_sentence(baseline: str, candidate: str, operation: str) -> str:
    prefix = "delete_sentence:"
    if not operation.startswith(prefix):
        raise ValueError(f"not a deletion operation: {operation}")
    index = int(operation[len(prefix):])
    baseline_core, _ = split_answer(baseline)
    candidate_core, _ = split_answer(candidate)
    before = sentences(baseline_core)
    after = sentences(candidate_core)
    if not 0 <= index < len(before):
        raise ValueError(f"deletion index out of range: {operation}")
    expected = before[:index] + before[index + 1:]
    if expected != after:
        raise ValueError(f"candidate does not delete exactly {operation}")
    return before[index]


def company(internal_id: str) -> str:
    value = str(internal_id).split("_", 1)[0]
    if value not in COMPANIES:
        raise ValueError(f"unexpected company: {value}")
    return value


def signature_statistics(examples: list[dict], held: str) -> dict[str, dict]:
    by_signature = defaultdict(list)
    for row in examples:
        if row["company"] != held:
            by_signature[row["signature"]].append(row)
    statistics = {}
    training_companies = sorted(set(COMPANIES) - {held})
    for signature, rows in by_signature.items():
        grouped = {
            source: [row for row in rows if row["company"] == source]
            for source in training_companies
        }
        counts = {source: len(values) for source, values in grouped.items()}
        means = {
            source: float(np.mean([row["delta"] for row in values]))
            for source, values in grouped.items() if values
        }
        supported = all(counts[source] >= 3 for source in training_companies)
        balanced_mean = float(np.mean(list(means.values()))) if means else float("nan")
        worst_mean = min(means.values()) if means else float("nan")
        statistics[signature] = {
            "counts": counts,
            "company_means": means,
            "source_balanced_mean": balanced_mean,
            "worst_company_mean": worst_mean,
            "total_support": len(rows),
            "eligible": bool(
                supported and balanced_mean >= 0.005 and worst_mean >= 0.0
            ),
        }
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lattice", type=Path,
        default=ROOT / "artifacts/polyfiqa_sparse_edit_oracle_v1/lattice.jsonl",
    )
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/finmmeval_task2/task2_test_questions.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_boilerplate_compression_v1",
    )
    args = parser.parse_args()

    lattice = load_jsonl(args.lattice)
    inputs = load_jsonl(args.input)
    deletion_examples = []
    rows = []
    for row in lattice:
        held_company = company(row["internal_id"])
        baseline = row["candidates"]["baseline"]
        row_record = {
            "row_index": int(row["row_index"]),
            "task_id": row["task_id"],
            "internal_id": row["internal_id"],
            "company": held_company,
            "tier": row["tier"],
            "family": row["family"],
            "baseline": baseline,
            "baseline_score": float(row["scores"]["baseline"]),
            "deletions": [],
        }
        for operation, answer in row["candidates"].items():
            if not operation.startswith("delete_sentence:"):
                continue
            removed = removed_sentence(baseline, answer, operation)
            example = {
                "row_index": int(row["row_index"]),
                "company": held_company,
                "tier": row["tier"],
                "family": row["family"],
                "operation": operation,
                "removed_sentence": removed,
                "signature": normalize_sentence(removed),
                "answer": answer,
                "score": float(row["scores"][operation]),
                "delta": float(row["scores"][operation] - row["scores"]["baseline"]),
            }
            deletion_examples.append(example)
            row_record["deletions"].append(example)
        rows.append(row_record)

    selected_rows = []
    signature_audit = {}
    for held in COMPANIES:
        statistics = signature_statistics(deletion_examples, held)
        signature_audit[held] = statistics
        for row in (item for item in rows if item["company"] == held):
            eligible = [
                deletion for deletion in row["deletions"]
                if statistics.get(deletion["signature"], {}).get("eligible", False)
            ]
            if eligible:
                chosen = min(eligible, key=lambda deletion: (
                    -statistics[deletion["signature"]]["source_balanced_mean"],
                    -statistics[deletion["signature"]]["total_support"],
                    deletion["signature"],
                ))
                operation = chosen["operation"]
                answer = chosen["answer"]
                selected_score = chosen["score"]
                signature = chosen["signature"]
                training_stat = statistics[signature]
            else:
                operation = "baseline"
                answer = row["baseline"]
                selected_score = row["baseline_score"]
                signature = None
                training_stat = None
            input_row = inputs[row["row_index"]]
            if str(input_row["task_id"]) != str(row["task_id"]):
                raise ValueError(f"input alignment mismatch at {row['row_index']}")
            if not candidate_valid(
                answer, pd.Series(input_row), row["internal_id"],
            ):
                raise ValueError(f"selected answer is invalid at {row['row_index']}")
            selected_rows.append({
                "row_index": row["row_index"], "task_id": row["task_id"],
                "internal_id": row["internal_id"], "held_company": held,
                "tier": row["tier"], "family": row["family"],
                "selected": operation, "signature": signature,
                "training_source_balanced_mean": (
                    None if training_stat is None
                    else training_stat["source_balanced_mean"]
                ),
                "training_worst_company_mean": (
                    None if training_stat is None
                    else training_stat["worst_company_mean"]
                ),
                "baseline_score": row["baseline_score"],
                "selected_score": selected_score,
                "delta": float(selected_score - row["baseline_score"]),
                "answer": answer,
            })

    selected_rows.sort(key=lambda row: row["row_index"])
    baseline = float(np.mean([row["baseline_score"] for row in selected_rows]))
    selected = float(np.mean([row["selected_score"] for row in selected_rows]))
    fold_metrics = []
    for held in COMPANIES:
        subset = [row for row in selected_rows if row["held_company"] == held]
        fold_metrics.append({
            "held_company": held,
            "rows": len(subset),
            "changed_rows": sum(row["selected"] != "baseline" for row in subset),
            "improved_rows": sum(row["delta"] > 0 for row in subset),
            "harmed_rows": sum(row["delta"] < 0 for row in subset),
            "baseline": float(np.mean([row["baseline_score"] for row in subset])),
            "selected": float(np.mean([row["selected_score"] for row in subset])),
            "delta": float(np.mean([row["delta"] for row in subset])),
        })
    bootstrap = exact_cluster_bootstrap(
        selected_rows, cluster_key="held_company", delta_key="delta",
    )
    gate = {
        "overall_delta_positive": selected > baseline,
        "at_least_three_folds_nonnegative": sum(
            row["delta"] >= 0 for row in fold_metrics
        ) >= 3,
        "at_most_one_fold_regresses": sum(
            row["delta"] < 0 for row in fold_metrics
        ) <= 1,
        "regression_smaller_than_0.002": all(
            row["delta"] > -0.002 for row in fold_metrics
        ),
        "bootstrap_p_gain_at_least_0.90": bootstrap["p_gain"] >= 0.90,
        "bootstrap_p05_nonnegative": bootstrap["p05"] >= 0.0,
        "all_selected_answers_valid": True,
    }
    gate["passed"] = all(gate.values())

    by_family = {}
    for (tier, family), subset in pd.DataFrame(selected_rows).groupby(
        ["tier", "family"], sort=True,
    ):
        by_family[f"{tier}:{family}"] = {
            "rows": len(subset),
            "changed_rows": int((subset.selected != "baseline").sum()),
            "delta": float(subset.delta.mean()),
        }
    summary = {
        "evaluation": "leave-one-company-out exact boilerplate deletion",
        "matched_rows": len(selected_rows),
        "deletion_examples": len(deletion_examples),
        "baseline_rouge": baseline,
        "selected_rouge": selected,
        "delta": selected - baseline,
        "changed_rows": sum(row["selected"] != "baseline" for row in selected_rows),
        "improved_rows": sum(row["delta"] > 0 for row in selected_rows),
        "harmed_rows": sum(row["delta"] < 0 for row in selected_rows),
        "selected_signatures": dict(Counter(
            row["signature"] for row in selected_rows if row["signature"] is not None
        )),
        "folds": fold_metrics,
        "by_family": by_family,
        "company_cluster_bootstrap": bootstrap,
        "promotion_gate": gate,
        "decision": "promote" if gate["passed"] else "retain_v15",
        "current_or_hidden_test_used": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).drop(columns=["answer"]).to_csv(
        args.output_dir / "audit.csv", index=False,
    )
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({"task_id": row["task_id"], "answer": row["answer"]}, ensure_ascii=False)
        + "\n" for row in selected_rows
    ))
    (args.output_dir / "signature_audit.json").write_text(
        json.dumps(signature_audit, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
