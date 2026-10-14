#!/usr/bin/env python3
"""Calibrate Task 1 edit operations with an external labelled split.

The gate and classifier are selected from 320 genuine OOF rows plus the
independently published 100-row Plutus test split.  The 80-row validation split
is loaded only after calibration is complete and is evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.evaluate_task1_dual_raw_proposer import (  # noqa: E402
    choose_oof_gate,
    crossfit_probabilities,
    evaluate_deployment_split,
    harmful_fraction,
    intersect_raw_proposals,
    load_jsonl,
    sha256,
    write_predictions,
)
from scripts.evaluate_task1_oof_annotation_program import (  # noqa: E402
    evaluate_policy,
    fit_components,
    load_oof_incumbent,
    load_validation_incumbent,
    make_operations,
)
from scripts.task1_numeric_baseline import parse_answer  # noqa: E402


PREREGISTRATION = ROOT / "docs/task1_external_operation_calibration_preregistration.md"
PREREGISTRATION_SHA256 = (
    "a1666ac4e13025d2dbd8590f6578d6db0069033b62b4408ac99ae8d8c6753263"
)
VALIDATION_BASELINE = 0.9331570512820513


def offset_operations(
    operations: list[dict[str, Any]], offset: int,
) -> list[dict[str, Any]]:
    return [operation | {"row_index": int(operation["row_index"]) + offset}
            for operation in operations]


def promotion_gate(metrics: dict[str, Any], calibration_exists: bool) -> dict[str, bool]:
    checks = {
        "validation_delta_at_least_0_005": float(metrics["delta"]) >= 0.005 - 1e-12,
        "at_least_two_rows_improve": int(metrics["rows_improved"]) >= 2,
        "zero_rows_harmed": int(metrics["rows_harmed"]) == 0,
        "all_type_deltas_nonnegative": min(
            float(value) for value in metrics["type_deltas"].values()
        ) >= -1e-12,
        "bootstrap_probability_gain_at_least_0_90": (
            float(metrics["bootstrap_probability_gain"]) >= 0.90
        ),
        "six_group_calibration_gate_exists": bool(calibration_exists),
    }
    return checks | {"passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=20000, choices=[20000])
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_external_operation_calibration_v1",
    )
    args = parser.parse_args()

    preregistration_sha = hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()
    if preregistration_sha != PREREGISTRATION_SHA256:
        raise ValueError(
            f"preregistration changed: {preregistration_sha} != {PREREGISTRATION_SHA256}"
        )

    paths = {
        "train": ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet",
        "external": ROOT / "data/external/plutus_finner_text/test-00000-of-00001.parquet",
        "validation": ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet",
        "seed7_oof": ROOT / "artifacts/task1_gliner_deploy_seed7_width16/oof_raw_predictions.jsonl",
        "seed19_oof": ROOT / "artifacts/task1_gliner_deploy_seed19_width16/oof_raw_predictions.jsonl",
        "oof_incumbent": ROOT / "artifacts/task1_multilabel_oof_legal_seed7/oof_predictions.jsonl",
        "external_seed7": ROOT / "artifacts/submission_portfolio_task1_enhanced_public_test_v1/task1_enhanced_work/seed7/ensemble_vote3_raw.jsonl",
        "external_seed19": ROOT / "artifacts/submission_portfolio_task1_enhanced_public_test_v1/task1_enhanced_work/seed19/ensemble_vote3_raw.jsonl",
        "external_incumbent": ROOT / "artifacts/submission_portfolio_task1_enhanced_public_test_v1/task1_grouped_consensus_primary.jsonl",
        "validation_seed7": ROOT / "artifacts/task1_gliner_deploy_seed7_width16/validation_ensemble_vote3_raw.jsonl",
        "validation_seed19": ROOT / "artifacts/task1_gliner_deploy_seed19_width16/validation_ensemble_vote3_raw.jsonl",
        "validation_incumbent": ROOT / "artifacts/task1_conservative_composition_grouped_seed_v1/predictions.jsonl",
    }

    # Calibration phase.  The validation frame and answers are deliberately not
    # loaded until the threshold and final classifier have been frozen.
    train = pd.read_parquet(paths["train"])
    train_gold = [set(parse_answer(answer)) for answer in train.answer]
    train_seed7 = load_jsonl(paths["seed7_oof"])
    train_seed19 = load_jsonl(paths["seed19_oof"])
    train_candidates, train_intersection = intersect_raw_proposals(
        train, train_seed7, train_seed19,
    )
    train_incumbent, train_occurrences = load_oof_incumbent(
        paths["oof_incumbent"], len(train),
    )
    incumbent_records = load_jsonl(paths["oof_incumbent"])
    train_folds = [int(record["fold"]) for record in train_seed7]
    if train_folds != [int(record["fold"]) for record in incumbent_records]:
        raise ValueError("seed-7 and incumbent OOF fold assignments differ")
    train_operations = make_operations(
        train, train_gold, train_candidates, train_incumbent,
        train_occurrences, train_folds,
    )

    external = pd.read_parquet(paths["external"])
    external_gold = [set(parse_answer(answer)) for answer in external.answer]
    external_candidates, external_intersection = intersect_raw_proposals(
        external, load_jsonl(paths["external_seed7"]),
        load_jsonl(paths["external_seed19"]), minimum_votes=3,
    )
    external_incumbent, external_occurrences = load_validation_incumbent(
        external, [paths["external_incumbent"]],
    )
    external_operations = make_operations(
        external, external_gold, external_candidates, external_incumbent,
        external_occurrences, [5] * len(external),
    )

    combined_gold = train_gold + external_gold
    combined_incumbent = train_incumbent + external_incumbent
    combined_folds = train_folds + [5] * len(external)
    combined_operations = train_operations + offset_operations(
        external_operations, len(train),
    )
    if not combined_operations:
        raise RuntimeError("no combined operations")
    operation_folds = {int(operation["fold"]) for operation in combined_operations}
    if operation_folds != set(range(6)):
        raise ValueError(f"expected operation groups 0..5, found {operation_folds}")

    probabilities = crossfit_probabilities(
        combined_operations, max_features=args.max_features,
    )
    frontier = []
    for threshold in np.linspace(0.05, 0.99, 95):
        _, accepted, metrics = evaluate_policy(
            combined_gold, combined_incumbent, combined_operations,
            probabilities, float(threshold), combined_folds,
        )
        fraction, beneficial, harmful, neutral = harmful_fraction(
            accepted, combined_operations,
        )
        metrics.update({
            "beneficial_operations": beneficial,
            "harmful_operations": harmful,
            "neutral_operations": neutral,
            "harmful_operation_fraction": fraction,
        })
        frontier.append(metrics)
    chosen = choose_oof_gate(
        frontier, minimum_operations=25, maximum_harmful_fraction=0.15,
    )
    gate = float(chosen["threshold"]) if chosen is not None else 1.01
    all_indices = np.arange(len(combined_operations))
    components = fit_components(
        combined_operations, all_indices, args.max_features,
    )

    # Single validation evaluation after all model selection has completed.
    validation = pd.read_parquet(paths["validation"])
    validation_gold = [set(parse_answer(answer)) for answer in validation.answer]
    validation_predictions, validation_accepted, validation_metrics = (
        evaluate_deployment_split(
            validation, validation_gold, paths["validation_incumbent"],
            paths["validation_seed7"], paths["validation_seed19"],
            components, gate,
        )
    )
    observed_baseline = float(validation_metrics["baseline_macro_entity_f1"])
    if abs(observed_baseline - VALIDATION_BASELINE) > 1e-12:
        raise ValueError(
            f"validation incumbent drift: {observed_baseline} != {VALIDATION_BASELINE}"
        )
    promotion = promotion_gate(validation_metrics, chosen is not None)

    summary = {
        "preregistration": str(PREREGISTRATION.relative_to(ROOT)),
        "preregistration_sha256": preregistration_sha,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "calibration": {
            "train_rows": len(train),
            "external_rows": len(external),
            "combined_rows": len(combined_gold),
            "train_intersection": train_intersection,
            "external_intersection": external_intersection,
            "train_operations": len(train_operations),
            "external_operations": len(external_operations),
            "combined_operations": len(combined_operations),
            "operation_groups": sorted(operation_folds),
            "beneficial_operations": sum(
                int(operation["beneficial"]) for operation in combined_operations
            ),
            "gate_status": "accepted" if chosen is not None else "abstain",
            "chosen_gate": chosen,
        },
        "validation": validation_metrics,
        "promotion_gate": promotion,
        "decision": "promote" if promotion["passed"] else "stop",
        "validation_residuals_motivated_hypothesis": True,
        "external_public_test_labels_used_for_calibration": True,
        "current_or_hidden_test_used": False,
        "post_hoc_tuning_allowed": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration_frontier.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in frontier
    ))
    write_predictions(
        args.output_dir / "validation_predictions.jsonl", validation_predictions,
    )
    (args.output_dir / "validation_accepted_operations.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in validation_accepted
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
