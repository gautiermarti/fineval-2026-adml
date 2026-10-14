#!/usr/bin/env python3
"""Evaluate an input-only numeric content planner for PolyFiQA.

Historical reference alignment supplies training labels indicating which typed
graph facts appeared in the answer.  At evaluation time the planner sees only
the input-derived graph structure and numeric values.  Every company is held
out in turn, including exclusion from both sides of counterfactual graph swaps.
This is a content-selection experiment, not yet a complete answer generator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
COMPANIES = ("HON", "JNJ", "MSFT", "UVV")


def sklearn_sparse(matrix):
    """Work around SciPy arm64 matrices defaulting to unsupported int64 indices."""
    matrix = matrix.tocsr()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def company(record: dict) -> str:
    return str(record.get("company", record.get("target_company", "")))


def fact_only(record: dict) -> bool:
    return all(
        slot["source"] == "fact" for slot in record["plan"]["requested_slots"]
    )


def candidate_rows(record: dict) -> list[dict]:
    plan = record["plan"]
    target = {
        (str(slot["metric"]), int(slot["index"]))
        for slot in plan["requested_slots"] if slot["source"] == "fact"
    }
    flattened = []
    total_candidates = sum(len(fact["values"]) for fact in plan["numeric"])
    for fact_position, fact in enumerate(plan["numeric"]):
        values = [float(row["value"]) for row in fact["values"]]
        absolute_order = {
            original_index: rank for rank, original_index in enumerate(
                sorted(range(len(values)), key=lambda index: abs(values[index]), reverse=True)
            )
        }
        for index, value in enumerate(values):
            metric = str(fact["metric"])
            key = metric, index
            features = {
                "family": str(plan["family"]),
                "tier": str(plan["tier"]),
                "metric": metric,
                "unit": str(fact["unit"]),
                "family_metric": f"{plan['family']}::{metric}",
                "family_metric_index": f"{plan['family']}::{metric}::{index}",
                "metric_index": f"{metric}::{index}",
                "index": index,
                "reverse_index": len(values) - 1 - index,
                "values_in_metric": len(values),
                "metric_position": fact_position,
                "metrics_in_plan": len(plan["numeric"]),
                "candidates_in_plan": total_candidates,
                "derived": int(bool(fact["derived"])),
                "negative": int(value < 0),
                "zero": int(value == 0),
                "absolute_rank": absolute_order[index],
                "log_absolute_value": math.log1p(abs(value)),
                "relative_to_first": (
                    value / values[0] if values and values[0] != 0 else 0.0
                ),
            }
            flattened.append({
                "key": key, "label": int(key in target), "features": features,
            })
    return flattened


def set_f1(target: set, prediction: set) -> float:
    if not target and not prediction:
        return 1.0
    if not target or not prediction:
        return 0.0
    return 2 * len(target & prediction) / (len(target) + len(prediction))


def family_median_counts(records: list[dict]) -> dict[tuple[str, str], int]:
    counts = defaultdict(list)
    for record in records:
        plan = record["plan"]
        target = {
            (slot["metric"], int(slot["index"]))
            for slot in plan["requested_slots"] if slot["source"] == "fact"
        }
        counts[(str(plan["tier"]), str(plan["family"]))].append(len(target))
    return {
        key: int(round(statistics.median(values))) for key, values in counts.items()
    }


def record_features(record: dict) -> dict:
    plan = record["plan"]
    values = [
        float(value["value"])
        for fact in plan["numeric"] for value in fact["values"]
    ]
    features = {
        "family": str(plan["family"]),
        "tier": str(plan["tier"]),
        "tier_family": f"{plan['tier']}::{plan['family']}",
        "metrics": len(plan["numeric"]),
        "numeric_candidates": len(values),
        "derived_metrics": sum(bool(fact["derived"]) for fact in plan["numeric"]),
        "negative_values": sum(value < 0 for value in values),
        "zero_values": sum(value == 0 for value in values),
    }
    for fact in plan["numeric"]:
        metric = str(fact["metric"])
        features[f"has_metric={metric}"] = 1
        fact_values = [float(value["value"]) for value in fact["values"]]
        if len(fact_values) >= 2:
            direction = "up" if fact_values[0] > fact_values[1] else "down_or_equal"
            features[f"direction={metric}={direction}"] = 1
    return features


def build_question_lookup(records: list[dict]):
    frame_cache = {}
    lookup = {}
    for record in records:
        if "source" not in record:
            continue
        source = str(record["source"])
        if source not in frame_cache:
            frame_cache[source] = pd.read_parquet(source, columns=["task_id", "question"])
        row = frame_cache[source].iloc[int(record["source_index"])]
        key = (
            str(record["plan"]["task_id"]), str(record["plan"]["tier"]),
            str(record["plan"]["family"]),
        )
        lookup[key] = str(row.question)
    return lookup


def record_question(record: dict, lookup: dict) -> str:
    task_id = str(record.get("donor_task_id", record["plan"]["task_id"]))
    return lookup.get(
        (task_id, str(record["plan"]["tier"]), str(record["plan"]["family"])),
        "",
    )


def fit_planner(training: list[dict], question_lookup: dict) -> dict:
    """Fit fact and record gates on one leakage-controlled training partition."""
    training_candidates = [
        candidate for record in training for candidate in candidate_rows(record)
    ]
    vectorizer = DictVectorizer(sparse=True)
    train_matrix = sklearn_sparse(vectorizer.fit_transform(
        [row["features"] for row in training_candidates]
    ))
    labels = np.asarray([row["label"] for row in training_candidates])
    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000,
        solver="liblinear", random_state=7,
    ).fit(train_matrix, labels)

    record_vectorizer = DictVectorizer(sparse=True)
    question_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        max_features=5000, sublinear_tf=True,
    )
    training_record_matrix = sklearn_sparse(hstack([
        record_vectorizer.fit_transform([record_features(row) for row in training]),
        question_vectorizer.fit_transform([
            record_question(row, question_lookup) for row in training
        ]),
    ], format="csr"))
    record_labels = np.asarray([
        int(bool(record["plan"]["requested_slots"])) for record in training
    ])
    record_classifier = LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=2000,
        solver="liblinear", random_state=7,
    ).fit(training_record_matrix, record_labels)
    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "record_vectorizer": record_vectorizer,
        "question_vectorizer": question_vectorizer,
        "record_classifier": record_classifier,
        "median_counts": family_median_counts(training),
        "training_candidates": len(training_candidates),
    }


def score_plan_record(record: dict, bundle: dict, question_lookup: dict) -> dict:
    """Return input-only fact and record probabilities before thresholding."""
    candidates = candidate_rows(record)
    probabilities = (
        bundle["classifier"].predict_proba(sklearn_sparse(
            bundle["vectorizer"].transform(
                [row["features"] for row in candidates]
            )
        ))[:, 1]
        if candidates else np.asarray([])
    )
    record_matrix = sklearn_sparse(hstack([
        bundle["record_vectorizer"].transform([record_features(record)]),
        bundle["question_vectorizer"].transform([
            record_question(record, question_lookup)
        ]),
    ], format="csr"))
    numeric_probability = float(
        bundle["record_classifier"].predict_proba(record_matrix)[0, 1]
    )
    return {
        "record": record,
        "candidates_raw": candidates,
        "probabilities": probabilities,
        "target": {row["key"] for row in candidates if row["label"]},
        "numeric_answer_probability": numeric_probability,
    }


def fact_threshold_prediction(scored: dict, fact_threshold: float) -> set:
    return {
        row["key"]
        for row, probability in zip(
            scored["candidates_raw"], scored["probabilities"]
        )
        if probability >= fact_threshold
    }


def nested_objective(rows: list[dict], prediction_key: str) -> float:
    """Balance qualitative abstention with quality on nonempty content plans."""
    metrics = summarize(rows, prediction_key)
    return 0.5 * (
        metrics["macro_set_f1"] + metrics["nonempty_macro_set_f1"]
    )


def select_nested_thresholds(scored_rows: list[dict]) -> tuple[float, float, dict]:
    """Choose both gates using only inner held-company predictions."""
    best = None
    for fact_threshold in np.round(np.arange(0.30, 0.71, 0.05), 2):
        for record_threshold in np.round(np.arange(0.30, 0.71, 0.05), 2):
            materialized = []
            for scored in scored_rows:
                guess = fact_threshold_prediction(scored, float(fact_threshold))
                if scored["numeric_answer_probability"] < record_threshold:
                    guess = set()
                materialized.append({
                    "target": scored["target"], "nested_candidate": guess,
                })
            metrics = summarize(materialized, "nested_candidate")
            objective = nested_objective(materialized, "nested_candidate")
            key = (
                objective,
                metrics["micro_f1"],
                -abs(float(fact_threshold) - 0.5),
                -abs(float(record_threshold) - 0.5),
            )
            if best is None or key > best[0]:
                best = (
                    key, float(fact_threshold), float(record_threshold), metrics,
                )
    if best is None:
        raise ValueError("no inner predictions available for nested calibration")
    return best[1], best[2], {
        "balanced_macro_objective": best[0][0], **best[3],
    }


def summarize(rows: list[dict], prediction_key: str) -> dict:
    true_positive = predicted = gold = exact = 0
    macro = []
    for row in rows:
        target = row["target"]
        guess = row[prediction_key]
        true_positive += len(target & guess)
        predicted += len(guess)
        gold += len(target)
        exact += int(target == guess)
        macro.append(set_f1(target, guess))
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    nonempty = [row for row in rows if row["target"]]
    nonempty_exact = sum(
        row["target"] == row[prediction_key] for row in nonempty
    )
    nonempty_macro = [
        set_f1(row["target"], row[prediction_key]) for row in nonempty
    ]
    return {
        "records": len(rows),
        "records_with_numeric_target": len(nonempty),
        "exact_set_accuracy": exact / len(rows) if rows else 0.0,
        "macro_set_f1": float(np.mean(macro)) if macro else 0.0,
        "nonempty_exact_set_accuracy": (
            nonempty_exact / len(nonempty) if nonempty else 0.0
        ),
        "nonempty_macro_set_f1": (
            float(np.mean(nonempty_macro)) if nonempty_macro else 0.0
        ),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positive": true_positive,
        "predicted": predicted,
        "gold": gold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v1",
        help="Strict graph-to-reference artifact containing training.jsonl.",
    )
    parser.add_argument(
        "--loco-dir", type=Path, default=None,
        help="LOCO split directory; defaults to <corpus-dir>/loco.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_numeric_planner_v1",
    )
    parser.add_argument(
        "--exclude-counterfactual", action="store_true",
        help="Train only on historical originals; retained as a swap-data ablation.",
    )
    args = parser.parse_args()
    loco_dir = args.loco_dir or args.corpus_dir / "loco"

    originals = load_jsonl(
        args.corpus_dir / "training.jsonl"
    )
    question_lookup = build_question_lookup(originals)

    all_evaluations = []
    fold_summaries = {}
    for held_company in COMPANIES:
        split_dir = loco_dir / held_company
        training = [row for row in load_jsonl(split_dir / "train.jsonl") if fact_only(row)]
        if args.exclude_counterfactual:
            training = [
                row for row in training if row.get("source_kind") == "historical"
            ]
        evaluation = [
            row for row in load_jsonl(split_dir / "evaluation.jsonl")
            if fact_only(row)
        ]
        if any(company(row) == held_company for row in training):
            raise AssertionError(f"{held_company} leaked into its planner training fold")

        inner_scored = []
        inner_fold_counts = {}
        for inner_company in COMPANIES:
            if inner_company == held_company:
                continue
            inner_evaluation = [
                row for row in training
                if row.get("source_kind") == "historical"
                and company(row) == inner_company
            ]
            inner_training = []
            for row in training:
                if row.get("source_kind") == "historical":
                    keep = company(row) != inner_company
                else:
                    keep = (
                        row.get("donor_company") != inner_company
                        and row.get("target_company") != inner_company
                    )
                if keep:
                    inner_training.append(row)
            if not inner_evaluation or not inner_training:
                continue
            inner_bundle = fit_planner(inner_training, question_lookup)
            inner_scored.extend(
                score_plan_record(row, inner_bundle, question_lookup)
                for row in inner_evaluation
            )
            inner_fold_counts[inner_company] = {
                "training_records": len(inner_training),
                "evaluation_records": len(inner_evaluation),
            }
        fact_threshold, record_threshold, calibration_metrics = (
            select_nested_thresholds(inner_scored)
        )

        bundle = fit_planner(training, question_lookup)
        median_counts = bundle["median_counts"]

        fold_rows = []
        for record_index, record in enumerate(evaluation):
            scored = score_plan_record(record, bundle, question_lookup)
            candidates = scored["candidates_raw"]
            probabilities = scored["probabilities"]
            target = scored["target"]
            numeric_probability = scored["numeric_answer_probability"]
            threshold_prediction = fact_threshold_prediction(scored, 0.5)
            two_stage_prediction = (
                threshold_prediction if numeric_probability >= 0.5 else set()
            )
            nested_prediction = fact_threshold_prediction(scored, fact_threshold)
            if numeric_probability < record_threshold:
                nested_prediction = set()
            count_key = str(record["plan"]["tier"]), str(record["plan"]["family"])
            count = min(median_counts.get(count_key, 0), len(candidates))
            ranked = sorted(
                zip(probabilities, candidates),
                key=lambda item: (-float(item[0]), item[1]["key"]),
            )
            topk_prediction = {row["key"] for _, row in ranked[:count]}
            all_prediction = {row["key"] for row in candidates}
            fold_rows.append({
                "held_company": held_company,
                "record_index": record_index,
                "task_id": record["plan"]["task_id"],
                "tier": record["plan"]["tier"],
                "family": record["plan"]["family"],
                "target": target,
                "threshold": threshold_prediction,
                "two_stage": two_stage_prediction,
                "nested": nested_prediction,
                "topk": topk_prediction,
                "all": all_prediction,
                "numeric_answer_probability": numeric_probability,
                "candidates": [
                    {"metric": row["key"][0], "index": row["key"][1],
                     "probability": float(probability), "selected": bool(row["label"])}
                    for row, probability in zip(candidates, probabilities)
                ],
            })
        all_evaluations.extend(fold_rows)
        fold_summaries[held_company] = {
            "training_records": len(training),
            "training_candidates": bundle["training_candidates"],
            "evaluation_records": len(evaluation),
            "nested_calibration": {
                "fact_threshold": fact_threshold,
                "record_threshold": record_threshold,
                "inner_fold_counts": inner_fold_counts,
                "inner_metrics": calibration_metrics,
            },
            "fixed_threshold_0.5": summarize(fold_rows, "threshold"),
            "two_stage_fixed_0.5": summarize(fold_rows, "two_stage"),
            "nested_train_company_thresholds": summarize(fold_rows, "nested"),
            "family_median_topk": summarize(fold_rows, "topk"),
            "select_all_facts": summarize(fold_rows, "all"),
        }

    summary = {
        "corpus_dir": str(args.corpus_dir.resolve()),
        "loco_dir": str(loco_dir.resolve()),
        "selection_labels": "historical references only",
        "evaluation": "leave one company out; held company excluded from original and swap data",
        "input_features": "typed numeric graph only; no reference text or oracle evidence",
        "fact_only_records": True,
        "counterfactual_training_included": not args.exclude_counterfactual,
        "folds": fold_summaries,
        "aggregate": {
            "fixed_threshold_0.5": summarize(all_evaluations, "threshold"),
            "two_stage_fixed_0.5": summarize(all_evaluations, "two_stage"),
            "nested_train_company_thresholds": summarize(all_evaluations, "nested"),
            "family_median_topk": summarize(all_evaluations, "topk"),
            "select_all_facts": summarize(all_evaluations, "all"),
        },
        "family_counts": dict(Counter(row["family"] for row in all_evaluations)),
        "end_to_end_answer_claim": False,
        "public_test_answers_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({
            **{key: value for key, value in row.items()
               if key not in {"target", "threshold", "two_stage", "nested", "topk", "all"}},
            "target": sorted(row["target"]),
            "threshold": sorted(row["threshold"]),
            "two_stage": sorted(row["two_stage"]),
            "nested": sorted(row["nested"]),
            "topk": sorted(row["topk"]),
            "all": sorted(row["all"]),
        }, ensure_ascii=False) + "\n"
        for row in all_evaluations
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
