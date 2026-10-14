#!/usr/bin/env python3
"""Nested confidence gating for the frozen JF-ICR ordinal ensemble."""

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


def confidence_values(probabilities_: np.ndarray, method: str) -> np.ndarray:
    ordered = np.sort(probabilities_, axis=1)
    if method == "max_probability":
        return ordered[:, -1]
    if method == "top2_margin":
        return ordered[:, -1] - ordered[:, -2]
    raise ValueError(method)


def gated_prediction(
    probabilities_: np.ndarray,
    rules: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> np.ndarray:
    raw = decode_probabilities(probabilities_, ordinal_mean=True)
    prediction = np.where(rules != "+1", rules, raw)
    fallback = (rules == "+1") & (raw != "+1") & (confidence < threshold)
    return np.where(fallback, "+1", prediction)


def choose_threshold(
    probabilities_: np.ndarray,
    rules: np.ndarray,
    confidence: np.ndarray,
    gold: np.ndarray,
    mask: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    scores = []
    for threshold in thresholds:
        prediction = gated_prediction(probabilities_, rules, confidence, float(threshold))
        scores.append((float(np.mean(prediction[mask] == gold[mask])), float(threshold)))
    best = max(score for score, _ in scores)
    # Conservative tie break makes the smallest behavioral change.
    return min(threshold for score, threshold in scores if score == best), best


def evaluate_method(
    method: str,
    probabilities_: np.ndarray,
    rules: np.ndarray,
    gold: np.ndarray,
    folds: np.ndarray,
    sources: np.ndarray,
    incumbent: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[dict, list[dict], np.ndarray]:
    confidence = confidence_values(probabilities_, method)
    nested = np.empty(len(gold), dtype=object)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        held_out = folds == fold
        threshold, calibration_accuracy = choose_threshold(
            probabilities_, rules, confidence, gold, ~held_out, thresholds,
        )
        candidate = gated_prediction(probabilities_, rules, confidence, threshold)
        nested[held_out] = candidate[held_out]
        fold_rows.append({
            "method": method, "held_out_fold": int(fold),
            "selected_threshold": threshold,
            "calibration_accuracy": calibration_accuracy,
            "incumbent_accuracy": float(np.mean(incumbent[held_out] == gold[held_out])),
            "nested_accuracy": float(np.mean(candidate[held_out] == gold[held_out])),
            "changed_rows": int(np.sum(candidate[held_out] != incumbent[held_out])),
        })
    deployment_threshold, deployment_oof_accuracy = choose_threshold(
        probabilities_, rules, confidence, gold,
        np.ones(len(gold), dtype=bool), thresholds,
    )
    bootstrap = source_bootstrap_delta(nested, incumbent, gold, sources)
    result = {
        "method": method,
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "nested_accuracy": float(np.mean(nested == gold)),
        "core_accuracy": float(np.mean(nested[:94] == gold[:94])),
        "expanded_accuracy": float(np.mean(nested[94:] == gold[94:])),
        "folds_improved": sum(r["nested_accuracy"] > r["incumbent_accuracy"] for r in fold_rows),
        "folds_tied": sum(r["nested_accuracy"] == r["incumbent_accuracy"] for r in fold_rows),
        "folds_worsened": sum(r["nested_accuracy"] < r["incumbent_accuracy"] for r in fold_rows),
        "deployment_threshold": deployment_threshold,
        "deployment_oof_accuracy": deployment_oof_accuracy,
        **bootstrap,
    }
    # Two predeclared confidence definitions are tested, so require a slightly
    # stronger positive-probability gate than the earlier single-profile work.
    result["promotion_eligible"] = bool(
        result["nested_accuracy"] > result["incumbent_accuracy"]
        and result["folds_improved"] >= 3
        and result["folds_worsened"] <= 1
        and result["bootstrap_probability_positive"] >= 0.95
        and result["bootstrap_p05_delta"] >= 0.0
    )
    return result, fold_rows, nested


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id").loc[frame.id]
    ids = frame.id.to_numpy()
    gold = frame.answer.astype(str).to_numpy()
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])
    probs = np.mean([
        probabilities(ROOT / f"artifacts/task2_japanese_cv/seed{seed}/oof_predictions.csv", ids)
        for seed in (7, 19)
    ], axis=0)
    incumbent = gated_prediction(probs, rules, np.ones(len(frame)), 0.0)
    configurations = {
        "max_probability": np.round(np.arange(0.0, 0.901, 0.05), 2),
        "top2_margin": np.round(np.arange(0.0, 0.601, 0.05), 2),
    }
    results, fold_rows, predictions = [], [], {"id": ids, "gold": gold, "incumbent": incumbent}
    for method, thresholds in configurations.items():
        result, rows, nested = evaluate_method(
            method, probs, rules, gold, folds, sources, incumbent, thresholds,
        )
        results.append(result); fold_rows.extend(rows); predictions[method] = nested
    output = ROOT / "artifacts/icr_nested_confidence"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "folds.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    (output / "decision.json").write_text(json.dumps({
        "eligible_methods": [r["method"] for r in results if r["promotion_eligible"]],
        "decision": "promotion_candidate_found" if any(r["promotion_eligible"] for r in results) else "retain_incumbent",
    }, indent=2) + "\n")
    print(pd.DataFrame(results).to_string(index=False))
    print(pd.DataFrame(fold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
