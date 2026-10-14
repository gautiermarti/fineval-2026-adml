#!/usr/bin/env python3
"""Nested source-fold calibration of Task 2 ordinal decision thresholds."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_ordinal_decode import load_probabilities  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402


VALUES = np.arange(-2, 3)
LABEL_VALUE = {"-2": -2, "-1": -1, "0": 0, "+1": 1, "+2": 2}


def fit_thresholds(scores: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Find the minimum-error five-bin monotonic partition by dynamic programming."""
    order = np.argsort(scores)
    scores, gold = scores[order], gold[order]
    n = len(scores)
    cost = np.zeros((5, n + 1, n + 1), dtype=np.int16)
    for class_index, label in enumerate(VALUES):
        prefix = np.r_[0, np.cumsum(gold == label)]
        for left in range(n + 1):
            right = np.arange(left, n + 1)
            cost[class_index, left, left:] = (
                right - left - (prefix[left:] - prefix[left])
            )
    dp = np.full((6, n + 1), 10**9, dtype=np.int32)
    back = np.zeros((6, n + 1), dtype=np.int32)
    dp[0, 0] = 0
    for class_count in range(1, 6):
        for right in range(n + 1):
            choices = dp[class_count - 1, :right + 1] + cost[
                class_count - 1, :right + 1, right
            ]
            back[class_count, right] = int(np.argmin(choices))
            dp[class_count, right] = int(choices[back[class_count, right]])
    positions = []
    right = n
    for class_count in range(5, 0, -1):
        left = int(back[class_count, right])
        positions.append(left)
        right = left
    positions = list(reversed(positions))[1:]
    bounds = []
    for position in positions:
        if position == 0:
            bounds.append(-np.inf)
        elif position == n:
            bounds.append(np.inf)
        else:
            bounds.append((scores[position - 1] + scores[position]) / 2)
    return np.asarray(bounds)


def decode(scores: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return VALUES[np.searchsorted(bounds, scores, side="right")]


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    metadata = metadata.loc[frame.id]
    gold = frame.answer.astype(str).map(LABEL_VALUE).to_numpy()
    probabilities = (load_probabilities(7) + load_probabilities(19)) / 2
    scores = probabilities @ VALUES
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])
    overrides = np.asarray([LABEL_VALUE[value] for value in rules])
    active = rules == "+1"
    folds = metadata.source_fold.to_numpy()

    def apply_rules(raw: np.ndarray) -> np.ndarray:
        return np.where(active, raw, overrides)

    incumbent = apply_rules(np.clip(np.rint(scores), -2, 2).astype(int))
    nested = np.empty(len(frame), dtype=int)
    fold_results = []
    for fold in sorted(np.unique(folds)):
        train = (folds != fold) & active
        test = folds == fold
        bounds = fit_thresholds(scores[train], gold[train])
        nested[test] = apply_rules(decode(scores, bounds))[test]
        fold_results.append({
            "fold": int(fold), "thresholds": bounds.tolist(),
            "incumbent_accuracy": float(np.mean(incumbent[test] == gold[test])),
            "nested_accuracy": float(np.mean(nested[test] == gold[test])),
        })

    sources = metadata.source_group.to_numpy()
    groups = np.unique(sources)
    indices = {group: np.flatnonzero(sources == group) for group in groups}
    rng = np.random.default_rng(20260804)
    deltas = np.empty(100_000)
    for draw in range(len(deltas)):
        sampled = rng.choice(groups, len(groups), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        deltas[draw] = (
            np.mean(nested[chosen] == gold[chosen])
            - np.mean(incumbent[chosen] == gold[chosen])
        )
    result = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "nested_accuracy": float(np.mean(nested == gold)),
        "changed_rows": int(np.sum(nested != incumbent)),
        "bootstrap_p05_delta": float(np.quantile(deltas, .05)),
        "bootstrap_median_delta": float(np.quantile(deltas, .50)),
        "bootstrap_p95_delta": float(np.quantile(deltas, .95)),
        "bootstrap_probability_positive": float(np.mean(deltas > 0)),
        "folds": fold_results,
        "decision": "retain_incumbent",
    }
    output = ROOT / "artifacts/icr_nested_ordinal_thresholds/decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
