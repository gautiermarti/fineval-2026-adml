#!/usr/bin/env python3
"""Nested source-fold evaluation of a lexical/neural JF-ICR blend.

The lexical probabilities and Japanese-BERT probabilities are both generated
out of fold.  For each held-out source fold, the blend weight is chosen using
only the other four folds.  This prevents a favorable global weight from being
mistaken for evidence of generalization to unseen issuers/documents.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import probabilities, source_bootstrap_delta  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402
from train_icr_japanese_cv import LABELS  # noqa: E402


def lexical_estimator() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(2, 5), min_df=2,
                max_features=80_000, sublinear_tf=True,
            )),
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=40_000, sublinear_tf=True,
            )),
        ])),
        ("classifier", LogisticRegression(
            C=2.0, max_iter=3_000, random_state=2026,
        )),
    ])


def lexical_oof(texts: np.ndarray, gold: np.ndarray, folds: np.ndarray) -> np.ndarray:
    output = np.zeros((len(texts), len(LABELS)), dtype=float)
    for fold in sorted(np.unique(folds)):
        held_out = folds == fold
        fitted = lexical_estimator().fit(texts[~held_out], gold[~held_out])
        fold_probabilities = fitted.predict_proba(texts[held_out])
        for column, label in enumerate(fitted.classes_):
            output[held_out, LABELS.index(str(label))] = fold_probabilities[:, column]
    return output


def apply_rules(probability_matrix: np.ndarray, rules: np.ndarray) -> np.ndarray:
    raw = decode_probabilities(probability_matrix, ordinal_mean=True)
    return np.where(rules != "+1", rules, raw)


def choose_weight(
    neural: np.ndarray,
    lexical: np.ndarray,
    gold: np.ndarray,
    rules: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    scored = []
    for weight in weights:
        prediction = apply_rules((1.0 - weight) * neural + weight * lexical, rules)
        scored.append((float(np.mean(prediction[mask] == gold[mask])), float(weight)))
    best_score = max(score for score, _ in scored)
    # If calibration cannot distinguish weights, retain more of the stronger
    # neural incumbent.
    best_weight = min(weight for score, weight in scored if score == best_score)
    return best_weight, best_score


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--neural-oof", type=Path, nargs=2,
        default=[
            ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv",
            ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv",
        ],
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_nested_lexical_blend",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    ids = frame.id.to_numpy()
    gold = frame.answer.astype(str).to_numpy()
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id").loc[ids]
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    parsed = frame["query"].map(parse_icr_query)
    texts = np.asarray([
        f"質問: {item.question}\n回答: {item.response}" for item in parsed
    ])
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])

    neural = np.mean([probabilities(path, ids) for path in args.neural_oof], axis=0)
    lexical = lexical_oof(texts, gold, folds)
    incumbent = apply_rules(neural, rules)
    lexical_only = apply_rules(lexical, rules)
    weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)

    nested = np.empty(len(frame), dtype=object)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        held_out = folds == fold
        weight, calibration_accuracy = choose_weight(
            neural, lexical, gold, rules, ~held_out, weights,
        )
        prediction = apply_rules((1.0 - weight) * neural + weight * lexical, rules)
        nested[held_out] = prediction[held_out]
        fold_rows.append({
            "held_out_fold": int(fold),
            "selected_lexical_weight": weight,
            "calibration_accuracy": calibration_accuracy,
            "incumbent_accuracy": float(np.mean(incumbent[held_out] == gold[held_out])),
            "nested_accuracy": float(np.mean(prediction[held_out] == gold[held_out])),
            "rows": int(held_out.sum()),
        })

    deployment_weight, deployment_oof_accuracy = choose_weight(
        neural, lexical, gold, rules, np.ones(len(frame), dtype=bool), weights,
    )
    bootstrap = source_bootstrap_delta(
        nested, incumbent, gold, sources, draws=args.bootstrap_draws,
    )
    summary = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "lexical_only_accuracy": float(np.mean(lexical_only == gold)),
        "nested_accuracy": float(np.mean(nested == gold)),
        "folds_improved": sum(row["nested_accuracy"] > row["incumbent_accuracy"] for row in fold_rows),
        "folds_tied": sum(row["nested_accuracy"] == row["incumbent_accuracy"] for row in fold_rows),
        "folds_worsened": sum(row["nested_accuracy"] < row["incumbent_accuracy"] for row in fold_rows),
        "deployment_lexical_weight": deployment_weight,
        "deployment_oof_accuracy": deployment_oof_accuracy,
        **bootstrap,
    }
    summary["promotion_eligible"] = bool(
        summary["nested_accuracy"] > summary["incumbent_accuracy"]
        and summary["folds_improved"] >= 3
        and summary["folds_worsened"] <= 1
        and summary["bootstrap_probability_positive"] >= 0.90
        and summary["bootstrap_p05_delta"] >= 0.0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "folds.csv", index=False)
    prediction_frame = pd.DataFrame({
        "id": ids, "gold": gold, "source_fold": folds,
        "incumbent": incumbent, "lexical_only": lexical_only,
        "nested_blend": nested,
    })
    for column, label in enumerate(LABELS):
        prediction_frame[f"lexical_probability_{label}"] = lexical[:, column]
    prediction_frame.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(fold_rows).to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
