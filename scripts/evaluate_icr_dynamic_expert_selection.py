#!/usr/bin/env python3
"""Nested source-disjoint dynamic selection over frozen JF-ICR experts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import probabilities, source_bootstrap_delta  # noqa: E402
from fineval.icr_dynamic_selection import dynamic_select  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


OOF_PATHS = {
    "general_seed7": ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv",
    "general_seed19": ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv",
    "edinet_seed19": ROOT / "artifacts/task2_edinet_cv/seed19/oof_predictions.csv",
    "rinna_seed7": ROOT / "artifacts/task2_rinna_roberta_cv/seed7_full/oof_predictions.csv",
    "finance_seed7": ROOT / "artifacts/task2_japanese_finance_cv/bert_base_additional_seed7/oof_predictions.csv",
}
LEVELS = np.arange(-2, 3)
NEIGHBOURS = (8, 16, 32, 64)
MARGINS = (0.0, 0.03, 0.06, 0.10)
CONDITIONAL = (False, True)


def similarity(train_text: np.ndarray, target_text: np.ndarray) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2,
        max_features=80_000, sublinear_tf=True, norm="l2",
    )
    train_matrix = vectorizer.fit_transform(train_text)
    target_matrix = vectorizer.transform(target_text)
    return (target_matrix @ train_matrix.T).toarray()


def source_accuracy(predicted, gold, sources) -> float:
    groups = sorted(set(sources.tolist()))
    return float(np.mean([
        np.mean(predicted[sources == source] == gold[sources == source])
        for source in groups
    ]))


def decode_experts(cube: np.ndarray, rules: np.ndarray) -> tuple[np.ndarray, list[str]]:
    incumbent_probability = cube[:, :2].mean(axis=1)
    incumbent = decode_probabilities(incumbent_probability, ordinal_mean=True).astype(int)
    incumbent = np.where(rules != 1, rules, incumbent)
    bases = []
    for expert in range(cube.shape[1]):
        decoded = np.clip(np.rint(cube[:, expert] @ LEVELS), -2, 2).astype(int)
        bases.append(np.where(rules != 1, rules, decoded))
    names = ["incumbent", *OOF_PATHS.keys()]
    return np.column_stack([incumbent, *bases]), names


def main() -> None:
    output = ROOT / "artifacts/icr_dynamic_expert_selection_v1"
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    ids = frame.id.to_numpy()
    gold = frame.answer.astype(int).to_numpy()
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id").loc[ids]
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    parsed = frame["query"].map(parse_icr_query)
    texts = np.asarray([
        f"質問: {item.question}\n回答: {item.response}" for item in parsed
    ])
    rules = np.asarray([int(rule_prediction(query, "safe_rules")) for query in frame["query"]])
    cube = np.stack([probabilities(path, ids) for path in OOF_PATHS.values()], axis=1)
    expert_predictions, expert_names = decode_experts(cube, rules)
    incumbent = expert_predictions[:, 0]

    nested = np.zeros(len(frame), dtype=int)
    selected_experts = np.zeros(len(frame), dtype=int)
    fold_rows = []
    for outer_fold in sorted(np.unique(folds)):
        outer_train = folds != outer_fold
        outer_test = ~outer_train
        candidates = []
        for inner_fold in sorted(np.unique(folds[outer_train])):
            inner_train = outer_train & (folds != inner_fold)
            inner_valid = outer_train & (folds == inner_fold)
            similarities = similarity(texts[inner_train], texts[inner_valid])
            for neighbours in NEIGHBOURS:
                for margin in MARGINS:
                    for conditional in CONDITIONAL:
                        prediction, _ = dynamic_select(
                            similarities, gold[inner_train], sources[inner_train],
                            expert_predictions[inner_train], expert_predictions[inner_valid],
                            neighbours=neighbours, margin=margin,
                            conditional_on_prediction=conditional,
                        )
                        candidates.append({
                            "neighbours": neighbours, "margin": margin,
                            "conditional": conditional,
                            "correct": int(np.sum(prediction == gold[inner_valid])),
                            "rows": int(inner_valid.sum()),
                            "source_accuracy": source_accuracy(
                                prediction, gold[inner_valid], sources[inner_valid],
                            ),
                        })
        aggregate = {}
        for row in candidates:
            key = row["neighbours"], row["margin"], row["conditional"]
            value = aggregate.setdefault(key, {"correct": 0, "rows": 0, "source": []})
            value["correct"] += row["correct"]
            value["rows"] += row["rows"]
            value["source"].append(row["source_accuracy"])
        scored = []
        for key, value in aggregate.items():
            scored.append((
                value["correct"] / value["rows"],
                float(np.mean(value["source"])), key,
            ))
        best_row_accuracy = max(row[0] for row in scored)
        best_source_accuracy = max(row[1] for row in scored if row[0] == best_row_accuracy)
        tied = [row for row in scored if row[:2] == (best_row_accuracy, best_source_accuracy)]
        # Conservative tie-break: larger margin, more neighbours, unconditional.
        _, _, selected_key = max(
            tied, key=lambda row: (row[2][1], row[2][0], not row[2][2]),
        )
        neighbours, margin, conditional = selected_key
        similarities = similarity(texts[outer_train], texts[outer_test])
        prediction, audits = dynamic_select(
            similarities, gold[outer_train], sources[outer_train],
            expert_predictions[outer_train], expert_predictions[outer_test],
            neighbours=neighbours, margin=margin,
            conditional_on_prediction=conditional,
        )
        nested[outer_test] = prediction
        selected_experts[outer_test] = [row["selected_expert"] for row in audits]
        fold_rows.append({
            "held_out_fold": int(outer_fold),
            "selected_neighbours": neighbours,
            "selected_margin": margin,
            "selected_conditional": conditional,
            "inner_row_accuracy": best_row_accuracy,
            "inner_source_accuracy": best_source_accuracy,
            "incumbent_accuracy": float(np.mean(incumbent[outer_test] == gold[outer_test])),
            "dynamic_accuracy": float(np.mean(prediction == gold[outer_test])),
            "rows": int(outer_test.sum()),
        })

    bootstrap = source_bootstrap_delta(
        nested.astype(str), incumbent.astype(str), gold.astype(str), sources,
    )
    selection_counts = Counter(expert_names[index] for index in selected_experts)
    summary = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "dynamic_accuracy": float(np.mean(nested == gold)),
        "rescues": int(np.sum((nested == gold) & (incumbent != gold))),
        "harms": int(np.sum((nested != gold) & (incumbent == gold))),
        "folds_improved": sum(row["dynamic_accuracy"] > row["incumbent_accuracy"] for row in fold_rows),
        "folds_tied": sum(row["dynamic_accuracy"] == row["incumbent_accuracy"] for row in fold_rows),
        "folds_worsened": sum(row["dynamic_accuracy"] < row["incumbent_accuracy"] for row in fold_rows),
        "selection_counts": dict(selection_counts),
        "oracle_any_expert_accuracy": float(np.mean(
            np.any(expert_predictions == gold[:, None], axis=1)
        )),
        "nested_hyperparameter_selection": True,
        "public_test_used": False,
        **bootstrap,
    }
    summary["promotion_eligible"] = bool(
        summary["dynamic_accuracy"] >= 0.72
        and summary["folds_improved"] >= 3
        and summary["folds_worsened"] <= 1
        and summary["bootstrap_probability_positive"] >= 0.90
        and summary["bootstrap_p05_delta"] >= 0.0
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output / "folds.csv", index=False)
    pd.DataFrame({
        "id": ids, "gold": gold, "source_fold": folds, "source_group": sources,
        "incumbent": incumbent, "dynamic": nested,
        "selected_expert": [expert_names[index] for index in selected_experts],
    }).to_csv(output / "predictions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
