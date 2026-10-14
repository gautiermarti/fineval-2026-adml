#!/usr/bin/env python3
"""Predeclared source-robust evaluation for EDINET-adapted JF-ICR models."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402
from train_icr_japanese_cv import LABELS  # noqa: E402


def probabilities(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"gold": str, "prediction": str})
    if not np.array_equal(frame.id.to_numpy(), expected_ids):
        raise ValueError(f"OOF row IDs are not aligned: {path}")
    logits = frame[[f"logit_{label}" for label in LABELS]].to_numpy()
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def source_bootstrap_delta(
    candidate: np.ndarray,
    incumbent: np.ndarray,
    gold: np.ndarray,
    sources: np.ndarray,
    *,
    draws: int = 100_000,
    seed: int = 20260804,
) -> dict:
    groups = np.unique(sources)
    indices = {group: np.flatnonzero(sources == group) for group in groups}
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for draw in range(draws):
        sampled = rng.choice(groups, len(groups), replace=True)
        rows = np.concatenate([indices[group] for group in sampled])
        deltas[draw] = (
            np.mean(candidate[rows] == gold[rows])
            - np.mean(incumbent[rows] == gold[rows])
        )
    return {
        "bootstrap_p05_delta": float(np.quantile(deltas, 0.05)),
        "bootstrap_median_delta": float(np.quantile(deltas, 0.50)),
        "bootstrap_p95_delta": float(np.quantile(deltas, 0.95)),
        "bootstrap_probability_positive": float(np.mean(deltas > 0)),
    }


def evaluate_profile(
    name: str,
    probability_matrix: np.ndarray,
    gold: np.ndarray,
    folds: np.ndarray,
    sources: np.ndarray,
    rules: np.ndarray,
    incumbent: np.ndarray,
    *,
    ordinal: bool,
    draws: int,
) -> tuple[dict, np.ndarray]:
    raw = decode_probabilities(probability_matrix, ordinal_mean=ordinal)
    prediction = np.where(rules != "+1", rules, raw)
    fold_values = sorted(np.unique(folds))
    fold_accuracies = {
        f"fold_{fold}_accuracy": float(np.mean(prediction[folds == fold] == gold[folds == fold]))
        for fold in fold_values
    }
    incumbent_fold = np.asarray([
        np.mean(incumbent[folds == fold] == gold[folds == fold]) for fold in fold_values
    ])
    candidate_fold = np.asarray([fold_accuracies[f"fold_{fold}_accuracy"] for fold in fold_values])
    bootstrap = source_bootstrap_delta(
        prediction, incumbent, gold, sources, draws=draws,
    )
    row = {
        "profile": name,
        "decoder": "ordinal_mean" if ordinal else "argmax",
        "all_accuracy": float(np.mean(prediction == gold)),
        "core_accuracy": float(np.mean(prediction[:94] == gold[:94])),
        "expanded_accuracy": float(np.mean(prediction[94:] == gold[94:])),
        "folds_improved": int(np.sum(candidate_fold > incumbent_fold)),
        "folds_tied": int(np.sum(candidate_fold == incumbent_fold)),
        "folds_worsened": int(np.sum(candidate_fold < incumbent_fold)),
        **fold_accuracies,
        **bootstrap,
    }
    row["promotion_eligible"] = bool(
        row["all_accuracy"] > np.mean(incumbent == gold)
        and row["folds_improved"] >= 3
        and row["folds_worsened"] <= 1
        and row["bootstrap_probability_positive"] >= 0.90
        and row["bootstrap_p05_delta"] >= 0.0
    )
    return row, prediction


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--adapted-oof", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--candidate-prefix", default="adapted",
        help="Prefix used to identify the candidate in result profiles.",
    )
    parser.add_argument(
        "--general-oof", type=Path, nargs=2,
        default=[
            ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv",
            ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv",
        ],
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_edinet_evaluation",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()
    if not args.candidate_prefix.replace("_", "").isalnum():
        raise ValueError("--candidate-prefix must contain only letters, digits, or underscores")

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    metadata = metadata.loc[frame.id]
    ids = frame.id.to_numpy()
    gold = frame.answer.astype(str).to_numpy()
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])
    general = [probabilities(path, ids) for path in args.general_oof]
    adapted = [probabilities(path, ids) for path in args.adapted_oof]

    incumbent_probabilities = np.mean(general, axis=0)
    incumbent_raw = decode_probabilities(incumbent_probabilities, ordinal_mean=True)
    incumbent = np.where(rules != "+1", rules, incumbent_raw)
    profiles = {"incumbent_general2": incumbent_probabilities}
    for index, matrix in enumerate(adapted, 1):
        profiles[f"{args.candidate_prefix}_seed_{index}"] = matrix
        profiles[f"general2_plus_{args.candidate_prefix}_seed_{index}"] = np.mean(
            [*general, matrix], axis=0,
        )
    if len(adapted) >= 2:
        profiles[f"{args.candidate_prefix}_ensemble"] = np.mean(adapted, axis=0)
        profiles[f"general2_plus_{args.candidate_prefix}_ensemble"] = np.mean(
            [*general, *adapted], axis=0,
        )

    rows = []
    predictions = {"id": ids, "gold": gold, "source_fold": folds}
    for name, matrix in profiles.items():
        for ordinal in (True, False):
            row, prediction = evaluate_profile(
                name, matrix, gold, folds, sources, rules, incumbent,
                ordinal=ordinal, draws=args.bootstrap_draws,
            )
            # The incumbent itself is the reference, not a promotion candidate.
            if name == "incumbent_general2":
                row["promotion_eligible"] = False
            rows.append(row)
            predictions[f"{name}__{row['decoder']}"] = prediction

    result = pd.DataFrame(rows).sort_values(
        ["promotion_eligible", "all_accuracy"], ascending=[False, False],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    eligible = result[result.promotion_eligible]
    decision = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "eligible_profiles": [
            {"profile": row.profile, "decoder": row.decoder}
            for row in eligible.itertuples()
        ],
        "decision": "promotion_candidate_found" if len(eligible) else "retain_incumbent",
        "gate": {
            "accuracy_strictly_higher": True,
            "minimum_folds_improved": 3,
            "maximum_folds_worsened": 1,
            "minimum_bootstrap_probability_positive": 0.90,
            "minimum_bootstrap_p05_delta": 0.0,
        },
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(result.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
