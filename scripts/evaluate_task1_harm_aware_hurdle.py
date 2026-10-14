#!/usr/bin/env python3
"""Evaluate the preregistered harm-aware Task 1 operation hurdle model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
from scripts.evaluate_task1_oof_annotation_program import (  # noqa: E402
    evaluate_policy,
    load_oof_incumbent,
    load_validation_incumbent,
    make_operations,
)
from scripts.task1_numeric_baseline import parse_answer  # noqa: E402


PREREGISTRATION = ROOT / "docs/task1_harm_aware_hurdle_preregistration.md"
PREREGISTRATION_SHA256 = (
    "55f879bdab0f2b0af141252edc3ab9cb26b24a1dc778b8accf532ff2bb339672"
)
VALIDATION_BASELINE = 0.9331570512820513

HurdleComponents = tuple[
    TfidfVectorizer,
    DictVectorizer,
    StandardScaler,
    LogisticRegression,
    LogisticRegression,
]


def combine_hurdle_probabilities(
    consequence_probability: np.ndarray,
    beneficial_given_consequence_probability: np.ndarray,
) -> np.ndarray:
    """Return the frozen signed expected-consequence operation score."""
    consequence_probability = np.asarray(consequence_probability, dtype=float)
    direction_probability = np.asarray(
        beneficial_given_consequence_probability, dtype=float,
    )
    if consequence_probability.shape != direction_probability.shape:
        raise ValueError("hurdle probability arrays have different shapes")
    return consequence_probability * (2.0 * direction_probability - 1.0)


def _vectorizers_and_matrix(
    operations: list[dict[str, Any]],
    indices: np.ndarray,
    max_features: int,
) -> tuple[TfidfVectorizer, DictVectorizer, StandardScaler, Any]:
    text_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        max_features=max_features, sublinear_tf=True,
    )
    numeric_vectorizer = DictVectorizer()
    scaler = StandardScaler(with_mean=False)
    text_matrix = text_vectorizer.fit_transform(
        [operations[index]["text_features"] for index in indices]
    )
    numeric_matrix = scaler.fit_transform(numeric_vectorizer.fit_transform(
        [operations[index]["features"] for index in indices]
    ))
    matrix = hstack([text_matrix, numeric_matrix], format="csr")
    return text_vectorizer, numeric_vectorizer, scaler, matrix


def fit_hurdle_components(
    operations: list[dict[str, Any]],
    indices: np.ndarray,
    max_features: int,
) -> HurdleComponents:
    indices = np.asarray(indices, dtype=int)
    text_vectorizer, numeric_vectorizer, scaler, matrix = _vectorizers_and_matrix(
        operations, indices, max_features,
    )
    deltas = np.asarray(
        [float(operations[index]["target_delta"]) for index in indices],
        dtype=float,
    )
    consequential = np.abs(deltas) > 1e-12
    consequence_labels = consequential.astype(int)
    direction_labels = (deltas[consequential] > 1e-12).astype(int)
    if len(set(consequence_labels.tolist())) < 2:
        raise ValueError("consequence training subset has only one class")
    if len(set(direction_labels.tolist())) < 2:
        raise ValueError("direction training subset has only one class")
    consequence_classifier = LogisticRegression(
        C=0.5, max_iter=1500, solver="liblinear", random_state=7,
    ).fit(matrix, consequence_labels)
    direction_classifier = LogisticRegression(
        C=0.5, max_iter=1500, solver="liblinear", random_state=7,
    ).fit(matrix[consequential], direction_labels)
    return (
        text_vectorizer, numeric_vectorizer, scaler,
        consequence_classifier, direction_classifier,
    )


def transform_hurdle(
    operations: list[dict[str, Any]],
    indices: np.ndarray,
    components: HurdleComponents,
) -> Any:
    text_vectorizer, numeric_vectorizer, scaler, _, _ = components
    text_matrix = text_vectorizer.transform(
        [operations[index]["text_features"] for index in indices]
    )
    numeric_matrix = scaler.transform(numeric_vectorizer.transform(
        [operations[index]["features"] for index in indices]
    ))
    return hstack([text_matrix, numeric_matrix], format="csr")


def hurdle_scores(
    operations: list[dict[str, Any]],
    indices: np.ndarray,
    components: HurdleComponents,
) -> np.ndarray:
    matrix = transform_hurdle(operations, indices, components)
    consequence_classifier = components[3]
    direction_classifier = components[4]
    consequence_probability = consequence_classifier.predict_proba(matrix)[:, 1]
    direction_probability = direction_classifier.predict_proba(matrix)[:, 1]
    return combine_hurdle_probabilities(
        consequence_probability, direction_probability,
    )


def crossfit_hurdle_scores(
    operations: list[dict[str, Any]], *, max_features: int,
) -> np.ndarray:
    scores = np.zeros(len(operations), dtype=float)
    indices = np.arange(len(operations))
    groups = np.asarray([int(operation["fold"]) for operation in operations])
    for group in sorted(set(groups.tolist())):
        fit_indices = indices[groups != group]
        held_indices = indices[groups == group]
        components = fit_hurdle_components(
            operations, fit_indices, max_features,
        )
        scores[held_indices] = hurdle_scores(
            operations, held_indices, components,
        )
    return scores


def promotion_gate(
    metrics: dict[str, Any], calibration_exists: bool,
) -> dict[str, bool]:
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


def evaluate_deployment_split(
    frame: pd.DataFrame,
    gold: list[set[tuple[str, str]]],
    incumbent_path: Path,
    seed7_path: Path,
    seed19_path: Path,
    components: HurdleComponents,
    gate: float,
) -> tuple[list[set[tuple[str, str]]], list[dict[str, Any]], dict[str, Any]]:
    incumbent, occurrences = load_validation_incumbent(frame, [incumbent_path])
    candidates, intersection = intersect_raw_proposals(
        frame, load_jsonl(seed7_path), load_jsonl(seed19_path), minimum_votes=3,
    )
    operations = make_operations(
        frame, gold, candidates, incumbent, occurrences, folds=None,
    )
    indices = np.arange(len(operations))
    scores = (
        hurdle_scores(operations, indices, components)
        if operations else np.asarray([], dtype=float)
    )
    predictions, accepted, metrics = evaluate_policy(
        gold, incumbent, operations, scores, gate, folds=None,
    )
    fraction, beneficial, harmful, neutral = harmful_fraction(accepted, operations)
    metrics.update({
        "raw_intersection": intersection,
        "candidate_operations": len(operations),
        "score_min": float(scores.min()) if len(scores) else None,
        "score_max": float(scores.max()) if len(scores) else None,
        "accepted_beneficial_operations": beneficial,
        "accepted_harmful_operations": harmful,
        "accepted_neutral_operations": neutral,
        "accepted_harmful_fraction": fraction,
    })
    return predictions, accepted, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=20000, choices=[20000])
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_harm_aware_hurdle_v1",
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

    # Calibration phase: validation data and labels are not loaded here.
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
    operation_groups = sorted({int(operation["fold"])
                               for operation in combined_operations})
    if operation_groups != list(range(6)):
        raise ValueError(f"expected operation groups 0..5, found {operation_groups}")

    scores = crossfit_hurdle_scores(
        combined_operations, max_features=args.max_features,
    )
    frontier = []
    for threshold in np.linspace(0.00, 0.95, 96):
        _, accepted, metrics = evaluate_policy(
            combined_gold, combined_incumbent, combined_operations,
            scores, float(threshold), combined_folds,
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

    # Single preregistered validation replay after the calibration decision.
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

    target_counts = {
        "beneficial": sum(float(operation["target_delta"]) > 1e-12
                          for operation in combined_operations),
        "harmful": sum(float(operation["target_delta"]) < -1e-12
                       for operation in combined_operations),
        "neutral": sum(abs(float(operation["target_delta"])) <= 1e-12
                       for operation in combined_operations),
    }
    summary = {
        "experiment": "task1_harm_aware_operation_hurdle",
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
            "operation_groups": operation_groups,
            "target_counts": target_counts,
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "gate_status": "accepted" if chosen is not None else "abstain",
            "chosen_gate": chosen,
        },
        "validation": validation_metrics,
        "promotion_gate": promotion,
        "decision": "promote" if promotion["passed"] else "stop",
        "validation_not_pristine_discovery": True,
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
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in validation_accepted
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
