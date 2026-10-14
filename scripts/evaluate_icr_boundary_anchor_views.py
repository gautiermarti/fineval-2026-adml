#!/usr/bin/env python3
"""CPU source-disjoint diagnostic for anchor-conditioned ordinal boundaries."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.icr_anchors import (  # noqa: E402
    LABELS, boundary_view_text, decode_cumulative_probabilities,
    project_cumulative_probabilities,
)
from predict_icr import rule_prediction  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_label(value) -> str:
    text = str(value)
    if text in LABELS:
        return text
    number = int(value)
    if not -2 <= number <= 2:
        raise ValueError(f"invalid ordinal label {value!r}")
    return f"+{number}" if number > 0 else str(number)


def source_equal_weights(groups: list[str]) -> np.ndarray:
    counts = Counter(groups)
    weights = np.asarray([1.0 / counts[group] for group in groups])
    return weights / weights.mean()


def summarize(
    gold: np.ndarray, prediction: np.ndarray, folds: np.ndarray,
) -> dict:
    matrix = np.zeros((5, 5), dtype=int)
    for target, guess in zip(gold, prediction, strict=True):
        matrix[LABELS.index(target), LABELS.index(guess)] += 1
    return {
        "accuracy": float(np.mean(gold == prediction)),
        "mean_ordinal_distance": float(np.mean([
            abs(LABELS.index(target) - LABELS.index(guess))
            for target, guess in zip(gold, prediction, strict=True)
        ])),
        "fold_accuracy": {
            str(fold): float(np.mean(prediction[folds == fold] == gold[folds == fold]))
            for fold in sorted(set(folds))
        },
        "class_recall": {
            label: (
                float(matrix[index, index] / matrix[index].sum())
                if matrix[index].sum() else 0.0
            )
            for index, label in enumerate(LABELS)
        },
        "predicted_counts": dict(Counter(prediction.tolist())),
        "confusion_matrix": matrix.tolist(),
    }


def clustered_delta(
    gold: np.ndarray, candidate: np.ndarray, control: np.ndarray,
    groups: np.ndarray, draws: int = 20_000,
) -> dict:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(20260805)
    deltas = np.empty(draws)
    for draw in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        rows = np.concatenate([indices[group] for group in sampled])
        deltas[draw] = (
            np.mean(candidate[rows] == gold[rows])
            - np.mean(control[rows] == gold[rows])
        )
    return {
        "observed_delta": float(np.mean(candidate == gold) - np.mean(control == gold)),
        "p05": float(np.quantile(deltas, 0.05)),
        "median": float(np.quantile(deltas, 0.50)),
        "p95": float(np.quantile(deltas, 0.95)),
        "probability_gain": float(np.mean(deltas > 0)),
        "probability_nonnegative": float(np.mean(deltas >= 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "artifacts/icr_boundary_anchors_v2",
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "data/derived/icr_source_folds.csv",
    )
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet",
    )
    parser.add_argument(
        "--incumbent", type=Path,
        default=ROOT / "artifacts/icr_nested_lexical_blend/predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_boundary_anchor_views_v1",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).sort_values("id").reset_index(drop=True)
    metadata = (
        pd.read_csv(args.metadata).set_index("id").loc[frame.id].reset_index(drop=True)
    )
    id_to_position = {int(identifier): position for position, identifier in enumerate(frame.id)}
    gold = frame.answer.astype(str).to_numpy()
    folds = metadata.source_fold.astype(int).to_numpy()
    groups = metadata.source_group.astype(str).to_numpy()
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])
    incumbent_frame = pd.read_csv(args.incumbent).set_index("id").loc[frame.id]
    incumbent = incumbent_frame.incumbent.map(canonical_label).to_numpy()
    incumbent_gold = incumbent_frame.gold.map(canonical_label).to_numpy()
    if not np.array_equal(incumbent_gold, gold):
        raise ValueError("incumbent predictions are not aligned with official rows")

    output_rows = []
    view_results = {}
    for view, include_anchors in (("target_only", False), ("boundary_anchors", True)):
        probabilities = np.full((len(frame), 4), np.nan, dtype=float)
        boundary_diagnostics = {str(index): [] for index in range(4)}
        for fold in sorted(set(folds)):
            fold_dir = args.corpus_dir / f"fold_{fold}"
            training = load_jsonl(fold_dir / "train.jsonl")
            evaluation = load_jsonl(fold_dir / "evaluation.jsonl")
            for boundary_index in range(4):
                train_pairs = [
                    (record, boundary)
                    for record in training for boundary in record["boundaries"]
                    if int(boundary["boundary_index"]) == boundary_index
                ]
                valid_pairs = [
                    (record, boundary)
                    for record in evaluation for boundary in record["boundaries"]
                    if int(boundary["boundary_index"]) == boundary_index
                ]
                train_texts = [
                    boundary_view_text(record, boundary, include_anchors)
                    for record, boundary in train_pairs
                ]
                valid_texts = [
                    boundary_view_text(record, boundary, include_anchors)
                    for record, boundary in valid_pairs
                ]
                train_labels = np.asarray([
                    int(boundary["gold_above_boundary"])
                    for _, boundary in train_pairs
                ])
                if set(train_labels) != {0, 1}:
                    raise ValueError(
                        f"fold {fold} boundary {boundary_index} lacks both classes"
                    )
                vectorizer = TfidfVectorizer(
                    analyzer="char", ngram_range=(2, 5), min_df=2,
                    max_features=60_000, sublinear_tf=True,
                )
                train_matrix = vectorizer.fit_transform(train_texts)
                valid_matrix = vectorizer.transform(valid_texts)
                model = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=3000,
                    solver="liblinear", random_state=7,
                )
                model.fit(
                    train_matrix, train_labels,
                    sample_weight=source_equal_weights([
                        str(record["source_group"]) for record, _ in train_pairs
                    ]),
                )
                valid_probability = model.predict_proba(valid_matrix)[:, 1]
                valid_labels = np.asarray([
                    int(boundary["gold_above_boundary"])
                    for _, boundary in valid_pairs
                ])
                for (record, _), probability in zip(
                    valid_pairs, valid_probability, strict=True,
                ):
                    probabilities[id_to_position[int(record["id"])], boundary_index] = probability
                boundary_diagnostics[str(boundary_index)].append({
                    "fold": int(fold),
                    "rows": len(valid_pairs),
                    "accuracy_at_0.5": float(np.mean(
                        (valid_probability >= 0.5) == valid_labels
                    )),
                    "brier": float(np.mean((valid_probability - valid_labels) ** 2)),
                    "train_negative": int(np.sum(train_labels == 0)),
                    "train_positive": int(np.sum(train_labels == 1)),
                })
        if np.isnan(probabilities).any():
            raise AssertionError(f"incomplete OOF probabilities for {view}")
        violations = np.any(probabilities[:, :-1] < probabilities[:, 1:], axis=1)
        projected = project_cumulative_probabilities(probabilities)
        raw_prediction = decode_cumulative_probabilities(probabilities)
        ruled_prediction = np.where(rules != "+1", rules, raw_prediction)
        view_results[view] = {
            "raw": summarize(gold, raw_prediction, folds),
            "with_frozen_safe_rules": summarize(gold, ruled_prediction, folds),
            "raw_monotonic_violation_rate": float(np.mean(violations)),
            "boundary_diagnostics": boundary_diagnostics,
            "versus_incumbent_after_rules": clustered_delta(
                gold, ruled_prediction, incumbent, groups,
            ),
        }
        output_rows.extend({
            "id": int(identifier),
            "view": view,
            "gold": target,
            "raw_prediction": raw_guess,
            "ruled_prediction": ruled_guess,
            "incumbent": control,
            "source_fold": int(fold),
            **{f"probability_above_boundary_{index}": float(probabilities[position, index])
               for index in range(4)},
            **{f"projected_probability_{index}": float(projected[position, index])
               for index in range(4)},
        } for position, (identifier, target, raw_guess, ruled_guess, control, fold) in enumerate(
            zip(frame.id, gold, raw_prediction, ruled_prediction, incumbent, folds, strict=True)
        ))

    target_predictions = np.asarray([
        row["ruled_prediction"] for row in output_rows if row["view"] == "target_only"
    ])
    anchor_predictions = np.asarray([
        row["ruled_prediction"] for row in output_rows if row["view"] == "boundary_anchors"
    ])
    summary = {
        "evaluation": "five reconstructed-source-disjoint folds",
        "model": "independent source-equal character TF-IDF boundary classifiers",
        "decoder": "least-squares monotone projection then rounded expected ordinal rank",
        "views": view_results,
        "anchor_vs_target_after_rules": clustered_delta(
            gold, anchor_predictions, target_predictions, groups,
        ),
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "neural_promotion_claim": False,
        "current_or_hidden_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
