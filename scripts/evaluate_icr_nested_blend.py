#!/usr/bin/env python3
"""Nested source-fold calibration of the general/adapted JF-ICR blend.

For every held-out source fold, the adapted-model weight is selected using only
OOF predictions and labels from the other four folds.  This estimates the
generalization of the *weight-selection procedure*, rather than reporting the
optimistic score from choosing one weight on all public labels.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import probabilities, source_bootstrap_delta  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


def apply_rules(probability_matrix: np.ndarray, rules: np.ndarray) -> np.ndarray:
    raw = decode_probabilities(probability_matrix, ordinal_mean=True)
    return np.where(rules != "+1", rules, raw)


def choose_weight(
    general: np.ndarray,
    adapted: np.ndarray,
    gold: np.ndarray,
    rules: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    scored = []
    for weight in weights:
        blend = (1.0 - weight) * general + weight * adapted
        prediction = apply_rules(blend, rules)
        scored.append((float(np.mean(prediction[mask] == gold[mask])), float(weight)))
    best_score = max(score for score, _ in scored)
    # Conservative deterministic tie break: trust the adapted model as little
    # as possible when the calibration labels cannot distinguish candidates.
    best_weight = min(weight for score, weight in scored if score == best_score)
    return best_weight, best_score


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--general-oof", type=Path, nargs=2,
        default=[
            ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv",
            ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv",
        ],
    )
    parser.add_argument(
        "--adapted-oof", type=Path,
        default=ROOT / "artifacts/task2_edinet_cv/seed19/oof_predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_edinet_nested_blend",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    metadata = metadata.loc[frame.id]
    ids = frame.id.to_numpy()
    gold = frame.answer.astype(str).to_numpy()
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])

    general = np.mean([probabilities(path, ids) for path in args.general_oof], axis=0)
    adapted = probabilities(args.adapted_oof, ids)
    incumbent = apply_rules(general, rules)
    weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)

    nested = np.empty(len(frame), dtype=object)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        held_out = folds == fold
        selected_weight, calibration_accuracy = choose_weight(
            general, adapted, gold, rules, ~held_out, weights,
        )
        prediction = apply_rules(
            (1.0 - selected_weight) * general + selected_weight * adapted, rules,
        )
        nested[held_out] = prediction[held_out]
        fold_rows.append({
            "held_out_fold": int(fold),
            "selected_adapted_weight": selected_weight,
            "calibration_accuracy": calibration_accuracy,
            "incumbent_accuracy": float(np.mean(incumbent[held_out] == gold[held_out])),
            "nested_accuracy": float(np.mean(prediction[held_out] == gold[held_out])),
            "rows": int(held_out.sum()),
        })

    deployment_weight, deployment_oof_accuracy = choose_weight(
        general, adapted, gold, rules, np.ones(len(frame), dtype=bool), weights,
    )
    bootstrap = source_bootstrap_delta(
        nested, incumbent, gold, sources, draws=args.bootstrap_draws,
    )
    summary = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "nested_accuracy": float(np.mean(nested == gold)),
        "nested_core_accuracy": float(np.mean(nested[:94] == gold[:94])),
        "nested_expanded_accuracy": float(np.mean(nested[94:] == gold[94:])),
        "folds_improved": sum(row["nested_accuracy"] > row["incumbent_accuracy"] for row in fold_rows),
        "folds_tied": sum(row["nested_accuracy"] == row["incumbent_accuracy"] for row in fold_rows),
        "folds_worsened": sum(row["nested_accuracy"] < row["incumbent_accuracy"] for row in fold_rows),
        "deployment_adapted_weight": deployment_weight,
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
    pd.DataFrame({
        "id": ids, "gold": gold, "source_fold": folds,
        "incumbent": incumbent, "nested_blend": nested,
    }).to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
