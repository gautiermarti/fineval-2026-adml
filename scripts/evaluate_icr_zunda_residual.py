#!/usr/bin/env python3
"""Nested source-held-out evaluation of query-conditioned Zunda residuals."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import probabilities, source_bootstrap_delta  # noqa: E402
from evaluate_icr_tsutsuji_semantics import (  # noqa: E402
    LEVELS, OOF_PATHS, nested_prediction, source_weights,
)
from fineval.icr_zunda import zunda_signature  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
SHRINKAGES = (0.25, 0.5, 0.75, 1.0)
ABSTENTION_THRESHOLDS = (0.0, 0.1, 0.25, 0.4, 0.6, 0.8)


def residual_prediction(
    design: np.ndarray, score: np.ndarray, gold: np.ndarray,
    incumbent: np.ndarray, rules: np.ndarray, sources: np.ndarray,
    train: np.ndarray, evaluation: np.ndarray,
    alpha: float, shrinkage: float, threshold: float,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(
        design[train], gold[train] - score[train],
        ridge__sample_weight=source_weights(sources, train),
    )
    residual = shrinkage * model.predict(design[evaluation])
    candidate = np.clip(np.rint(score[evaluation] + residual), -2, 2).astype(int)
    candidate = np.where(
        np.abs(residual) >= threshold, candidate, incumbent[evaluation],
    )
    return np.where(rules[evaluation] != 1, rules[evaluation], candidate)


def nested_residual(
    design: np.ndarray, score: np.ndarray, gold: np.ndarray,
    folds: np.ndarray, sources: np.ndarray, rules: np.ndarray,
    incumbent: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    output = incumbent.copy()
    fold_rows = []
    for outer_fold in sorted(np.unique(folds)):
        outer_train = folds != outer_fold
        outer_test = ~outer_train
        options = [{
            "parameters": None,
            "inner_accuracy": float(np.mean(incumbent[outer_train] == gold[outer_train])),
            "minimum_inner_fold_delta": 0.0,
            "inner_fold_balance": 0,
            "inner_changed_fraction": 0.0,
        }]
        for alpha, shrinkage, threshold in itertools.product(
            ALPHAS, SHRINKAGES, ABSTENTION_THRESHOLDS,
        ):
            predictions = incumbent.copy()
            deltas = []
            for inner_fold in sorted(np.unique(folds[outer_train])):
                inner_train = outer_train & (folds != inner_fold)
                inner_valid = outer_train & (folds == inner_fold)
                predictions[inner_valid] = residual_prediction(
                    design, score, gold, incumbent, rules, sources,
                    inner_train, inner_valid, alpha, shrinkage, threshold,
                )
                deltas.append(float(
                    np.mean(predictions[inner_valid] == gold[inner_valid])
                    - np.mean(incumbent[inner_valid] == gold[inner_valid])
                ))
            options.append({
                "parameters": (alpha, shrinkage, threshold),
                "inner_accuracy": float(np.mean(
                    predictions[outer_train] == gold[outer_train]
                )),
                "minimum_inner_fold_delta": min(deltas),
                "inner_fold_balance": (
                    sum(delta > 0 for delta in deltas)
                    - sum(delta < 0 for delta in deltas)
                ),
                "inner_changed_fraction": float(np.mean(
                    predictions[outer_train] != incumbent[outer_train]
                )),
            })
        # Accuracy is primary; robustness and abstention break ties.  The no-op
        # incumbent wins exact ties, preventing gratuitous overrides.
        selected = max(options, key=lambda row: (
            row["inner_accuracy"], row["inner_fold_balance"],
            row["minimum_inner_fold_delta"], -row["inner_changed_fraction"],
            row["parameters"] is None,
        ))
        if selected["parameters"] is not None:
            output[outer_test] = residual_prediction(
                design, score, gold, incumbent, rules, sources,
                outer_train, outer_test, *selected["parameters"],
            )
        fold_rows.append({
            "held_out_fold": int(outer_fold),
            **{key: value for key, value in selected.items() if key != "parameters"},
            "selected_parameters": selected["parameters"],
            "incumbent_accuracy": float(np.mean(
                incumbent[outer_test] == gold[outer_test]
            )),
            "accuracy": float(np.mean(output[outer_test] == gold[outer_test])),
            "changed_rows": int(np.sum(output[outer_test] != incumbent[outer_test])),
        })
    return output, fold_rows


def evaluate(
    predicted: np.ndarray, incumbent: np.ndarray, gold: np.ndarray,
    folds: np.ndarray, sources: np.ndarray,
) -> dict:
    fold_candidate = [
        np.mean(predicted[folds == fold] == gold[folds == fold])
        for fold in sorted(np.unique(folds))
    ]
    fold_incumbent = [
        np.mean(incumbent[folds == fold] == gold[folds == fold])
        for fold in sorted(np.unique(folds))
    ]
    result = {
        "accuracy": float(np.mean(predicted == gold)),
        "rescues": int(np.sum((predicted == gold) & (incumbent != gold))),
        "harms": int(np.sum((predicted != gold) & (incumbent == gold))),
        "changed_rows": int(np.sum(predicted != incumbent)),
        "folds_improved": int(np.sum(np.asarray(fold_candidate) > fold_incumbent)),
        "folds_tied": int(np.sum(np.asarray(fold_candidate) == fold_incumbent)),
        "folds_worsened": int(np.sum(np.asarray(fold_candidate) < fold_incumbent)),
        **source_bootstrap_delta(
            predicted.astype(str), incumbent.astype(str), gold.astype(str), sources,
        ),
    }
    result["promotion_eligible"] = bool(
        result["accuracy"] >= 0.72 and result["folds_improved"] >= 3
        and result["folds_worsened"] <= 1
        and result["bootstrap_probability_positive"] >= 0.90
        and result["bootstrap_p05_delta"] >= 0.0
    )
    return result


def main() -> None:
    input_path = ROOT / "artifacts/icr_zunda_semantic_events_v1/events.jsonl"
    output_dir = ROOT / "artifacts/icr_zunda_residual_v1"
    records = [json.loads(line) for line in input_path.read_text().splitlines()]
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    ids = frame.id.to_numpy()
    if [row["id"] for row in records] != ids.tolist():
        raise ValueError("Zunda artifact and JF-ICR row order differ")
    gold = frame.answer.astype(int).to_numpy()
    metadata = pd.read_csv(
        ROOT / "data/derived/icr_source_folds.csv"
    ).set_index("id").loc[ids]
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray([
        int(rule_prediction(query, "safe_rules")) for query in frame["query"]
    ])
    cube = np.stack([probabilities(path, ids) for path in OOF_PATHS], axis=1)
    incumbent_probabilities = cube.mean(axis=1)
    score = incumbent_probabilities @ LEVELS
    incumbent = decode_probabilities(
        incumbent_probabilities, ordinal_mean=True,
    ).astype(int)
    incumbent = np.where(rules != 1, rules, incumbent)
    base = np.concatenate([
        incumbent_probabilities, score[:, None],
        (-(incumbent_probabilities * np.log(
            incumbent_probabilities + 1e-12
        )).sum(axis=1))[:, None],
    ], axis=1)

    semantic_signatures = [zunda_signature(row["candidates"]) for row in records]
    raw_signatures = [zunda_signature([
        {key: value for key, value in candidate.items() if key != "ruri_similarity"}
        for candidate in row["candidates"]
    ]) for row in records]
    vectorizers = {
        "raw": DictVectorizer(sparse=False),
        "semantic": DictVectorizer(sparse=False),
    }
    designs = {
        "raw": vectorizers["raw"].fit_transform(raw_signatures),
        "semantic": vectorizers["semantic"].fit_transform(semantic_signatures),
    }
    variants = {}
    predictions = {
        "id": ids, "gold": gold, "source_fold": folds,
        "source_group": sources, "incumbent": incumbent,
    }
    all_folds = []
    for view, features in designs.items():
        for mode, design in (
            ("direct", np.concatenate([base, features], axis=1)),
            ("residual", np.concatenate([base, features], axis=1)),
        ):
            name = f"{view}_{mode}"
            if mode == "direct":
                predicted, fold_rows = nested_prediction(
                    design, gold, folds, sources, rules,
                )
            else:
                predicted, fold_rows = nested_residual(
                    design, score, gold, folds, sources, rules, incumbent,
                )
            variants[name] = evaluate(
                predicted, incumbent, gold, folds, sources,
            )
            predictions[name] = predicted
            all_folds.extend({"variant": name, **row} for row in fold_rows)

    result = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "evaluation": "fully nested five-fold reconstructed-source-disjoint",
        "semantic_retrieval": "Ruri-v3-310m query/document cosine; label-free",
        "raw_feature_count": int(designs["raw"].shape[1]),
        "semantic_feature_count": int(designs["semantic"].shape[1]),
        "current_or_hidden_test_used": False,
        "variants": variants,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(all_folds).to_csv(output_dir / "folds.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    for view, vectorizer in vectorizers.items():
        (output_dir / f"feature_names_{view}.txt").write_text(
            "\n".join(vectorizer.get_feature_names_out()) + "\n"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
