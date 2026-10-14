#!/usr/bin/env python3
"""Nested, label-free target-prior correction for the frozen JF-ICR ensemble."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_ordinal_decode import load_probabilities  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402
from train_icr_japanese_cv import LABELS  # noqa: E402


THREE_BIN = np.asarray([0, 0, 1, 2, 2], dtype=int)
FIVE_WAY = np.arange(5, dtype=int)


def aggregate_probabilities(probabilities: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    bins = int(mapping.max()) + 1
    output = np.zeros((len(probabilities), bins), dtype=np.float64)
    for label_index, bin_index in enumerate(mapping):
        output[:, bin_index] += probabilities[:, label_index]
    return output


def label_bins(labels: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    indices = np.asarray([LABELS.index(str(label)) for label in labels], dtype=int)
    return mapping[indices]


def normalized_counts(values: np.ndarray, bins: int) -> np.ndarray:
    counts = np.bincount(values, minlength=bins).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("source partition lacks a required label bin")
    return counts / counts.sum()


def estimate_soft_bbse_prior(
    source_probabilities: np.ndarray,
    source_labels: np.ndarray,
    target_probabilities: np.ndarray,
    mapping: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate target bin priors from a soft confusion matrix."""
    source_bins = aggregate_probabilities(source_probabilities, mapping)
    target_bins = aggregate_probabilities(target_probabilities, mapping)
    gold_bins = label_bins(source_labels, mapping)
    bins = source_bins.shape[1]
    source_prior = normalized_counts(gold_bins, bins)
    confusion = np.column_stack([
        source_bins[gold_bins == index].mean(axis=0) for index in range(bins)
    ])
    target_mean = target_bins.mean(axis=0)

    def objective(prior: np.ndarray) -> float:
        residual = confusion @ prior - target_mean
        return float(residual @ residual + ridge * np.sum((prior - source_prior) ** 2))

    result = minimize(
        objective,
        source_prior.copy(),
        method="SLSQP",
        bounds=[(1e-6, 1.0)] * bins,
        constraints={"type": "eq", "fun": lambda prior: float(prior.sum() - 1.0)},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"soft-BBSE optimization failed: {result.message}")
    target_prior = np.asarray(result.x, dtype=np.float64)
    target_prior /= target_prior.sum()
    return source_prior, target_prior, confusion


def estimate_em_prior(
    source_probabilities: np.ndarray,
    source_labels: np.ndarray,
    target_probabilities: np.ndarray,
    mapping: np.ndarray,
    *,
    shrink: float,
    iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate target priors with a shrunk Saerens-style fixed point."""
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("shrink must be in [0, 1]")
    target_bins = aggregate_probabilities(target_probabilities, mapping)
    gold_bins = label_bins(source_labels, mapping)
    source_prior = normalized_counts(gold_bins, target_bins.shape[1])
    target_prior = source_prior.copy()
    for _ in range(iterations):
        adjusted = target_bins * (target_prior / source_prior)[None, :]
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        update = shrink * source_prior + (1.0 - shrink) * adjusted.mean(axis=0)
        if np.max(np.abs(update - target_prior)) < 1e-12:
            target_prior = update
            break
        target_prior = update
    target_prior /= target_prior.sum()
    return source_prior, target_prior


def reweight_probabilities(
    probabilities: np.ndarray,
    source_prior: np.ndarray,
    target_prior: np.ndarray,
    mapping: np.ndarray,
) -> np.ndarray:
    ratios = target_prior / source_prior
    adjusted = probabilities * ratios[mapping][None, :]
    totals = adjusted.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(adjusted)) or np.any(totals <= 0):
        raise ValueError("non-finite label-shift adjustment")
    return adjusted / totals


def apply_rules(probabilities: np.ndarray, rules: np.ndarray) -> np.ndarray:
    decoded = decode_probabilities(probabilities, ordinal_mean=True)
    return np.where(rules != "+1", rules, decoded)


def recall(mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    return float(np.mean(predicted_mask[mask])) if np.any(mask) else 1.0


def bootstrap_delta(
    candidate: np.ndarray,
    incumbent: np.ndarray,
    gold: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        positions = np.concatenate([indices[group] for group in sampled])
        deltas[draw] = (
            np.mean(candidate[positions] == gold[positions])
            - np.mean(incumbent[positions] == gold[positions])
        )
    return {
        "p05": float(np.quantile(deltas, 0.05)),
        "median": float(np.quantile(deltas, 0.50)),
        "p95": float(np.quantile(deltas, 0.95)),
        "p_gain": float(np.mean(deltas > 0)),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_transductive_label_shift_v1",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    metadata = metadata.loc[frame.id]
    probabilities = (load_probabilities(7) + load_probabilities(19)) / 2
    gold = frame.answer.astype(str).to_numpy()
    folds = metadata.source_fold.to_numpy(dtype=int)
    groups = metadata.source_group.astype(str).to_numpy()
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])

    specifications = {
        "identity": {"kind": "identity"},
        "soft_bbse_3bin_ridge_0.1": {
            "kind": "bbse", "mapping": THREE_BIN, "ridge": 0.1,
        },
        "soft_bbse_3bin_ridge_1.0": {
            "kind": "bbse", "mapping": THREE_BIN, "ridge": 1.0,
        },
        "soft_bbse_5way_ridge_0.1": {
            "kind": "bbse", "mapping": FIVE_WAY, "ridge": 0.1,
        },
        "em_3bin_shrink_0.5": {
            "kind": "em", "mapping": THREE_BIN, "shrink": 0.5,
        },
    }
    predictions = {
        name: np.empty(len(frame), dtype="<U2") for name in specifications
    }
    fold_audit = []
    for fold in sorted(np.unique(folds)):
        target = folds == fold
        source = ~target
        for name, specification in specifications.items():
            if specification["kind"] == "identity":
                adjusted = probabilities[target]
                source_prior = target_prior = None
                confusion = None
            elif specification["kind"] == "bbse":
                source_prior, target_prior, confusion = estimate_soft_bbse_prior(
                    probabilities[source], gold[source], probabilities[target],
                    specification["mapping"], ridge=specification["ridge"],
                )
                adjusted = reweight_probabilities(
                    probabilities[target], source_prior, target_prior,
                    specification["mapping"],
                )
            elif specification["kind"] == "em":
                source_prior, target_prior = estimate_em_prior(
                    probabilities[source], gold[source], probabilities[target],
                    specification["mapping"], shrink=specification["shrink"],
                )
                confusion = None
                adjusted = reweight_probabilities(
                    probabilities[target], source_prior, target_prior,
                    specification["mapping"],
                )
            else:
                raise AssertionError(specification)
            if not np.allclose(adjusted.sum(axis=1), 1.0):
                raise ValueError(f"probability normalization failed for {name} fold {fold}")
            predictions[name][target] = apply_rules(adjusted, rules[target])
            fold_audit.append({
                "method": name,
                "fold": int(fold),
                "rows": int(target.sum()),
                "source_prior": None if source_prior is None else source_prior.tolist(),
                "estimated_target_prior": (
                    None if target_prior is None else target_prior.tolist()
                ),
                "soft_confusion": None if confusion is None else confusion.tolist(),
                "target_labels_used_for_estimation": False,
            })

    incumbent = predictions["identity"]
    fold_rows = []
    summaries = {}
    negative_gold = np.isin(gold, ["-2", "-1"])
    plus2_gold = gold == "+2"
    for offset, (name, prediction) in enumerate(predictions.items()):
        per_fold = []
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            candidate_accuracy = float(np.mean(prediction[mask] == gold[mask]))
            incumbent_accuracy = float(np.mean(incumbent[mask] == gold[mask]))
            per_fold.append(candidate_accuracy)
            fold_rows.append({
                "method": name,
                "fold": int(fold),
                "rows": int(mask.sum()),
                "accuracy": candidate_accuracy,
                "incumbent_accuracy": incumbent_accuracy,
                "delta": candidate_accuracy - incumbent_accuracy,
                "changed_rows": int(np.sum(prediction[mask] != incumbent[mask])),
            })
        delta = prediction != incumbent
        rescued = delta & (prediction == gold) & (incumbent != gold)
        harmed = delta & (prediction != gold) & (incumbent == gold)
        bootstrap = bootstrap_delta(
            prediction, incumbent, gold, groups,
            draws=args.bootstrap_draws, seed=20260805 + offset,
        )
        summaries[name] = {
            "accuracy": float(np.mean(prediction == gold)),
            "fold_accuracies": per_fold,
            "folds_improved": int(sum(
                row["delta"] > 1e-12 for row in fold_rows if row["method"] == name
            )),
            "folds_tied": int(sum(
                abs(row["delta"]) <= 1e-12 for row in fold_rows if row["method"] == name
            )),
            "folds_regressed": int(sum(
                row["delta"] < -1e-12 for row in fold_rows if row["method"] == name
            )),
            "changed_rows": int(delta.sum()),
            "rescued_rows": int(rescued.sum()),
            "harmed_rows": int(harmed.sum()),
            "negative_recall": recall(negative_gold, np.isin(prediction, ["-2", "-1"])),
            "plus2_recall": recall(plus2_gold, prediction == "+2"),
            "prediction_counts": {
                label: int(np.sum(prediction == label)) for label in LABELS
            },
            "source_cluster_bootstrap_vs_identity": bootstrap,
        }

    primary = summaries["soft_bbse_3bin_ridge_0.1"]
    base = summaries["identity"]
    gate = {
        "accuracy_above_incumbent": primary["accuracy"] > base["accuracy"],
        "at_least_three_folds_improved": primary["folds_improved"] >= 3,
        "at_most_one_fold_regressed": primary["folds_regressed"] <= 1,
        "bootstrap_p_gain_at_least_0.90": (
            primary["source_cluster_bootstrap_vs_identity"]["p_gain"] >= 0.90
        ),
        "bootstrap_p05_nonnegative": (
            primary["source_cluster_bootstrap_vs_identity"]["p05"] >= 0.0
        ),
        "negative_recall_non_decreasing": (
            primary["negative_recall"] >= base["negative_recall"]
        ),
        "plus2_recall_non_decreasing": primary["plus2_recall"] >= base["plus2_recall"],
    }
    gate["passed"] = all(gate.values())
    decision = "promote_primary" if gate["passed"] else "retain_ordinal_incumbent"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    (args.output_dir / "fold_prior_audit.json").write_text(
        json.dumps(fold_audit, indent=2) + "\n"
    )
    with (args.output_dir / "oof_predictions.jsonl").open("w") as handle:
        for position, row_id in enumerate(frame.id):
            handle.write(json.dumps({
                "id": int(row_id),
                "gold": str(gold[position]),
                "fold": int(folds[position]),
                **{name: str(value[position]) for name, value in predictions.items()},
            }, ensure_ascii=False) + "\n")
    result = {
        "evaluation": "five reconstructed-source-disjoint transductive folds",
        "rows": len(frame),
        "source_groups": int(len(np.unique(groups))),
        "bootstrap_draws": args.bootstrap_draws,
        "preregistered_primary": "soft_bbse_3bin_ridge_0.1",
        "target_labels_used_for_prior_estimation": False,
        "methods": summaries,
        "promotion_gate": gate,
        "decision": decision,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote={args.output_dir}")


if __name__ == "__main__":
    main()
