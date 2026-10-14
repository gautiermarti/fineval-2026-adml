#!/usr/bin/env python3
"""Evaluate the preregistered rank-invariant Task 1 hurdle selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.evaluate_task1_dual_raw_proposer import (  # noqa: E402
    choose_oof_gate,
    harmful_fraction,
    intersect_raw_proposals,
    load_jsonl,
    sha256,
    write_predictions,
)
from scripts.evaluate_task1_external_operation_calibration import (  # noqa: E402
    offset_operations,
)
from scripts.evaluate_task1_harm_aware_hurdle import (  # noqa: E402
    crossfit_hurdle_scores,
    fit_hurdle_components,
    hurdle_scores,
    promotion_gate,
)
from scripts.evaluate_task1_oof_annotation_program import (  # noqa: E402
    evaluate_policy,
    load_oof_incumbent,
    load_validation_incumbent,
    make_operations,
)
from scripts.task1_numeric_baseline import parse_answer  # noqa: E402


PREREGISTRATION = ROOT / "docs/task1_rank_invariant_hurdle_preregistration.md"
PREREGISTRATION_SHA256 = (
    "010c7b953b879c0f27f5db8999d622309968678382512cc7b626a1524f9cb1e9"
)
VALIDATION_BASELINE = 0.9331570512820513


def empirical_group_percentiles(
    scores: np.ndarray, groups: np.ndarray,
) -> np.ndarray:
    """Map scores to frozen average-rank percentiles inside each group."""
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    if scores.shape != groups.shape:
        raise ValueError("score and group arrays have different shapes")
    if not np.isfinite(scores).all():
        raise ValueError("operation scores must be finite")
    output = np.zeros(len(scores), dtype=float)
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        count = int(mask.sum())
        if count == 1:
            output[mask] = 1.0
        else:
            output[mask] = (rankdata(scores[mask], method="average") - 1.0) / (
                count - 1.0
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=20000, choices=[20000])
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_rank_invariant_hurdle_v1",
    )
    args = parser.parse_args()

    preregistration_sha = hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()
    if preregistration_sha != PREREGISTRATION_SHA256:
        raise ValueError(
            f"preregistration changed: {preregistration_sha} != "
            f"{PREREGISTRATION_SHA256}"
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

    # Calibration: no validation data or answer is loaded before gate selection.
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
    combined_folds = np.asarray(train_folds + [5] * len(external), dtype=int)
    combined_operations = train_operations + offset_operations(
        external_operations, len(train),
    )
    operation_groups = np.asarray(
        [int(operation["fold"]) for operation in combined_operations], dtype=int,
    )
    if sorted(set(operation_groups.tolist())) != list(range(6)):
        raise ValueError("expected operation groups 0..5")

    raw_scores = crossfit_hurdle_scores(
        combined_operations, max_features=args.max_features,
    )
    percentile_scores = empirical_group_percentiles(raw_scores, operation_groups)
    frontier = []
    for threshold in np.linspace(0.50, 0.99, 50):
        _, accepted, metrics = evaluate_policy(
            combined_gold, combined_incumbent, combined_operations,
            percentile_scores, float(threshold), combined_folds.tolist(),
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
    components = fit_hurdle_components(
        combined_operations, all_indices, args.max_features,
    )

    # Frozen single deployment replay.
    validation = pd.read_parquet(paths["validation"])
    validation_gold = [set(parse_answer(answer)) for answer in validation.answer]
    validation_incumbent, validation_occurrences = load_validation_incumbent(
        validation, [paths["validation_incumbent"]],
    )
    validation_candidates, validation_intersection = intersect_raw_proposals(
        validation, load_jsonl(paths["validation_seed7"]),
        load_jsonl(paths["validation_seed19"]), minimum_votes=3,
    )
    validation_operations = make_operations(
        validation, validation_gold, validation_candidates,
        validation_incumbent, validation_occurrences, folds=None,
    )
    validation_indices = np.arange(len(validation_operations))
    validation_raw_scores = hurdle_scores(
        validation_operations, validation_indices, components,
    )
    validation_percentiles = empirical_group_percentiles(
        validation_raw_scores, np.zeros(len(validation_operations), dtype=int),
    )
    validation_predictions, validation_accepted, validation_metrics = evaluate_policy(
        validation_gold, validation_incumbent, validation_operations,
        validation_percentiles, gate, folds=None,
    )
    fraction, beneficial, harmful, neutral = harmful_fraction(
        validation_accepted, validation_operations,
    )
    validation_metrics.update({
        "raw_intersection": validation_intersection,
        "candidate_operations": len(validation_operations),
        "raw_score_min": float(validation_raw_scores.min()),
        "raw_score_max": float(validation_raw_scores.max()),
        "percentile_min": float(validation_percentiles.min()),
        "percentile_max": float(validation_percentiles.max()),
        "accepted_beneficial_operations": beneficial,
        "accepted_harmful_operations": harmful,
        "accepted_neutral_operations": neutral,
        "accepted_harmful_fraction": fraction,
    })
    observed_baseline = float(validation_metrics["baseline_macro_entity_f1"])
    if abs(observed_baseline - VALIDATION_BASELINE) > 1e-12:
        raise ValueError(
            f"validation incumbent drift: {observed_baseline} != {VALIDATION_BASELINE}"
        )
    promotion = promotion_gate(validation_metrics, chosen is not None)

    summary = {
        "experiment": "task1_rank_invariant_harm_aware_hurdle",
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
            "operation_groups": sorted(set(operation_groups.tolist())),
            "raw_score_min": float(raw_scores.min()),
            "raw_score_max": float(raw_scores.max()),
            "percentile_min": float(percentile_scores.min()),
            "percentile_max": float(percentile_scores.max()),
            "gate_status": "accepted" if chosen is not None else "abstain",
            "chosen_gate": chosen,
        },
        "validation": validation_metrics,
        "promotion_gate": promotion,
        "decision": "promote" if promotion["passed"] else "stop",
        "validation_hurdle_outcomes_unobserved_before_preregistration": True,
        "validation_not_pristine_discovery": True,
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
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in validation_accepted
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
