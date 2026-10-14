#!/usr/bin/env python3
"""Audit cost-sensitive versus prevalence-corrected boundary decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_anchors import (  # noqa: E402
    decode_cumulative_probabilities,
    undo_positive_weighted_bce_logits,
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def distribution(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "artifacts/icr_independent_boundaries_cv/seed7_cpu_fold0",
    )
    parser.add_argument(
        "--incumbent", type=Path,
        default=ROOT / "artifacts/icr_nested_lexical_blend/predictions.csv",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.run_dir / "oof_predictions_partial.csv")
    run_summary = json.loads((args.run_dir / "summary.json").read_text())
    logits = frame[[f"boundary_logit_{index}" for index in range(4)]].to_numpy()
    weights = np.asarray(run_summary["folds"][0]["positive_weights"], dtype=float)
    corrected_logits = undo_positive_weighted_bce_logits(logits, weights)
    raw_probabilities = sigmoid(logits)
    corrected_probabilities = sigmoid(corrected_logits)
    raw = np.asarray([int(value) for value in decode_cumulative_probabilities(
        raw_probabilities
    )])
    corrected = np.asarray([int(value) for value in decode_cumulative_probabilities(
        corrected_probabilities
    )])
    gold = frame.answer.astype(int).to_numpy()
    incumbent = (
        pd.read_csv(args.incumbent).set_index("id")
        .loc[frame.id, "incumbent"].astype(int).to_numpy()
    )
    summary = {
        "rows": len(frame),
        "raw_accuracy": float(np.mean(raw == gold)),
        "raw_prediction_distribution": distribution(raw),
        "analytic_unweight_accuracy": float(np.mean(corrected == gold)),
        "analytic_unweight_prediction_distribution": distribution(corrected),
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "analytic_unweight_rescues": int(np.sum((incumbent != gold) & (corrected == gold))),
        "analytic_unweight_harms": int(np.sum((incumbent == gold) & (corrected != gold))),
        "raw_mean_boundary_mass": float(raw_probabilities.sum(axis=1).mean()),
        "corrected_mean_boundary_mass": float(
            corrected_probabilities.sum(axis=1).mean()
        ),
        "raw_monotonic_violation_rate": float(np.mean(
            np.any(np.diff(raw_probabilities, axis=1) > 0, axis=1)
        )),
        "corrected_monotonic_violation_rate": float(np.mean(
            np.any(np.diff(corrected_probabilities, axis=1) > 0, axis=1)
        )),
        "correction": "subtract log(pos_weight) from each boundary logit",
        "correction_uses_evaluation_labels": False,
        "futility_gate_passed": False,
        "decision": "stop_after_fold_0",
        "current_or_hidden_test_used": False,
    }
    (args.run_dir / "decoder_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
