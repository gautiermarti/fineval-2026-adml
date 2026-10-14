#!/usr/bin/env python3
"""Evaluate sparse edits from older packets on source/time-disjoint future rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_polyfiqa_sparse_edit_risk_model import (  # noqa: E402
    ALPHAS, COMPANIES, THRESHOLDS, company, fit_predict, flatten, select_rows,
)
from fineval.polyfiqa_family_gate import exact_cluster_bootstrap  # noqa: E402


MINIMUM_ROWS_PER_COMPANY = 3
MINIMUM_MEAN_GAIN = 0.005
MINIMUM_COMPANY_GAIN = -0.002


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean_score(rows: list[dict]) -> float:
    return float(np.mean([row["selected_score"] for row in rows]))


def fit_signature_gates(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    observations: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        baseline = float(row["scores"]["baseline"])
        for operation, score in row["scores"].items():
            if operation == "baseline":
                continue
            observations[(row["tier"], row["family"], operation)].append({
                "company": company(row), "delta": float(score) - baseline,
            })
    training_companies = sorted({company(row) for row in records})
    gates = {}
    for signature, values in observations.items():
        company_gains = {}
        for issuer in training_companies:
            deltas = [row["delta"] for row in values if row["company"] == issuer]
            if len(deltas) >= MINIMUM_ROWS_PER_COMPANY:
                company_gains[issuer] = float(np.mean(deltas))
        mean_gain = float(np.mean([row["delta"] for row in values]))
        admitted = bool(
            len(company_gains) == len(training_companies)
            and mean_gain >= MINIMUM_MEAN_GAIN
            and min(company_gains.values()) >= MINIMUM_COMPANY_GAIN
        )
        gates[signature] = {
            "admitted": admitted, "training_rows": len(values),
            "mean_gain": mean_gain, "company_gains": company_gains,
        }
    return gates


def apply_signature_gates(records: list[dict], gates: dict) -> list[dict]:
    selected = []
    for row in records:
        choices = [(0.0, "baseline", row["candidates"]["baseline"])]
        for operation, answer in row["candidates"].items():
            if operation == "baseline":
                continue
            gate = gates.get((row["tier"], row["family"], operation))
            if gate and gate["admitted"]:
                choices.append((float(gate["mean_gain"]), operation, answer))
        predicted_gain, operation, answer = max(
            choices,
            key=lambda item: (item[0], item[1] == "baseline", -len(item[2])),
        )
        baseline = float(row["scores"]["baseline"])
        score = float(row["scores"][operation])
        selected.append({
            "row_index": int(row["row_index"]), "task_id": row["task_id"],
            "held_company": company(row), "tier": row["tier"],
            "family": row["family"], "selected": operation,
            "predicted_gain": predicted_gain, "baseline_score": baseline,
            "selected_score": score, "delta": score - baseline,
            "answer": answer,
        })
    return selected


def choose_ridge_hyperparameters(
    training_records: list[dict], training_examples: list[dict],
) -> dict:
    issuers = sorted({company(row) for row in training_records})
    options = []
    for alpha in ALPHAS:
        predicted_pairs = []
        for inner_held in issuers:
            train = [row for row in training_examples if row["company"] != inner_held]
            test = [row for row in training_examples if row["company"] == inner_held]
            prediction = fit_predict(train, test, alpha)
            predicted_pairs.extend(zip(test, prediction, strict=True))
        examples = [row for row, _ in predicted_pairs]
        predictions = np.asarray([value for _, value in predicted_pairs])
        for threshold in THRESHOLDS:
            selected = select_rows(
                training_records, examples, predictions, threshold,
            )
            options.append({
                "alpha": alpha, "threshold": threshold,
                "score": mean_score(selected),
                "delta": float(np.mean([row["delta"] for row in selected])),
                "changed": sum(row["selected"] != "baseline" for row in selected),
            })
    best = max(row["score"] for row in options)
    return max(
        (row for row in options if row["score"] == best),
        key=lambda row: (row["threshold"], row["alpha"]),
    )


def summarize(name: str, selected: list[dict], folds: list[dict]) -> dict:
    selected = sorted(selected, key=lambda row: row["row_index"])
    baseline = float(np.mean([row["baseline_score"] for row in selected]))
    score = mean_score(selected)
    bootstrap = exact_cluster_bootstrap(
        selected, cluster_key="held_company", delta_key="delta",
    )
    result = {
        "evaluation": name, "rows": len(selected),
        "baseline_rouge": baseline, "selected_rouge": score,
        "delta": score - baseline,
        "changed_rows": sum(row["selected"] != "baseline" for row in selected),
        "improved_rows": sum(row["delta"] > 0 for row in selected),
        "harmed_rows": sum(row["delta"] < 0 for row in selected),
        "folds": folds, "company_cluster_bootstrap": bootstrap,
        "current_or_hidden_test_used": False,
    }
    result["promotion_eligible"] = promotion_decision(result)
    return result


def promotion_decision(result: dict) -> bool:
    folds = result.get("folds", [])
    bootstrap = result.get("company_cluster_bootstrap", {})
    return bool(
        result.get("delta", 0.0) > 0
        and result.get("changed_rows", 0) >= 4
        and result.get("improved_rows", 0) > result.get("harmed_rows", 0)
        and sum(row["delta"] >= 0 for row in folds) >= 3
        and sum(row["delta"] < -0.002 for row in folds) <= 1
        and bootstrap.get("p_gain", 0.0) >= 0.90
        and bootstrap.get("p05", -1.0) >= 0.0
    )


def fold_summary(held: str, selected: list[dict], extras: dict) -> dict:
    return {
        "held_company": held, **extras, "rows": len(selected),
        "changed": sum(row["selected"] != "baseline" for row in selected),
        "baseline": float(np.mean([row["baseline_score"] for row in selected])),
        "selected": mean_score(selected),
        "delta": float(np.mean([row["delta"] for row in selected])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lattice", type=Path,
        default=ROOT / "artifacts/polyfiqa_historical_sparse_lattice_v1/lattice.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_temporal_sparse_selector_v1",
    )
    args = parser.parse_args()

    records = load_jsonl(args.lattice)
    early = [row for row in records if row["partition"] == "early_confirmation"]
    late = [row for row in records if row["partition"] == "late_discovery"]
    if len(early) != 151 or len(late) != 192:
        raise ValueError("unexpected temporal partition sizes")
    sources = {
        int(row["row_index"]): f"{row['question']}\n{row['query']}" for row in records
    }
    examples = flatten(records, sources)
    early_examples = [
        row for row in examples
        if next(item for item in records if item["row_index"] == row["row_index"])["partition"]
        == "early_confirmation"
    ]
    late_examples = [
        row for row in examples
        if next(item for item in records if item["row_index"] == row["row_index"])["partition"]
        == "late_discovery"
    ]

    variants = {}
    audits = {}

    # Signature gate: diagnostic time-only evaluation.
    gates = fit_signature_gates(early)
    selected = apply_signature_gates(late, gates)
    folds = [
        fold_summary(issuer, [row for row in selected if row["held_company"] == issuer], {})
        for issuer in COMPANIES
    ]
    variants["signature_time_only"] = summarize(
        "exact signature gate, early to late", selected, folds,
    )
    audits["signature_time_only"] = selected

    # Signature gate: primary source-and-time-disjoint evaluation.
    strict_selected, strict_folds = [], []
    for held in COMPANIES:
        training = [row for row in early if company(row) != held]
        test = [row for row in late if company(row) == held]
        held_gates = fit_signature_gates(training)
        held_selected = apply_signature_gates(test, held_gates)
        strict_selected.extend(held_selected)
        strict_folds.append(fold_summary(held, held_selected, {
            "admitted_signatures": sum(gate["admitted"] for gate in held_gates.values()),
        }))
    variants["signature_source_time_disjoint"] = summarize(
        "exact signature gate, source-and-time disjoint",
        strict_selected, strict_folds,
    )
    audits["signature_source_time_disjoint"] = strict_selected

    # Ridge: diagnostic time-only evaluation with early LOCO selection.
    chosen = choose_ridge_hyperparameters(early, early_examples)
    predictions = fit_predict(early_examples, late_examples, chosen["alpha"])
    selected = select_rows(late, late_examples, predictions, chosen["threshold"])
    folds = [
        fold_summary(issuer, [row for row in selected if row["held_company"] == issuer], {
            "selected_alpha": chosen["alpha"],
            "selected_threshold": chosen["threshold"],
            "inner_delta": chosen["delta"], "inner_changed": chosen["changed"],
        })
        for issuer in COMPANIES
    ]
    variants["ridge_time_only"] = summarize(
        "ridge risk model, early to late", selected, folds,
    )
    audits["ridge_time_only"] = selected

    # Ridge: primary source-and-time-disjoint evaluation.
    strict_selected, strict_folds = [], []
    for held in COMPANIES:
        training_records = [row for row in early if company(row) != held]
        training_examples = [row for row in early_examples if row["company"] != held]
        test_records = [row for row in late if company(row) == held]
        test_examples = [row for row in late_examples if row["company"] == held]
        chosen = choose_ridge_hyperparameters(training_records, training_examples)
        predictions = fit_predict(training_examples, test_examples, chosen["alpha"])
        held_selected = select_rows(
            test_records, test_examples, predictions, chosen["threshold"],
        )
        strict_selected.extend(held_selected)
        strict_folds.append(fold_summary(held, held_selected, {
            "selected_alpha": chosen["alpha"],
            "selected_threshold": chosen["threshold"],
            "inner_delta": chosen["delta"], "inner_changed": chosen["changed"],
        }))
    variants["ridge_source_time_disjoint"] = summarize(
        "ridge risk model, source-and-time disjoint",
        strict_selected, strict_folds,
    )
    audits["ridge_source_time_disjoint"] = strict_selected

    result = {
        "early_training_rows": len(early), "late_evaluation_rows": len(late),
        "late_labels_used_for_training_or_selection": False,
        "primary_variants": [
            "signature_source_time_disjoint", "ridge_source_time_disjoint",
        ],
        "variants": variants, "current_or_hidden_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in audits.items():
        pd.DataFrame(rows).drop(columns=["answer"]).sort_values("row_index").to_csv(
            args.output_dir / f"audit_{name}.csv", index=False,
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
