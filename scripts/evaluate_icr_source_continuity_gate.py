#!/usr/bin/env python3
"""Audit whether source-continuity diagnostics may replace Task 2 ordinal.

The neural probabilities are strictly source-OOF.  The source-prior candidate
is the existing leave-one-row-out continuity simulation: the target row is
removed, while other rows from its source may remain.  This deliberately gives
the continuity hypothesis its intended favorable evaluation condition.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_ordinal_decode import load_probabilities  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


def accuracy(prediction: np.ndarray, gold: np.ndarray) -> float:
    return float(np.mean(prediction == gold))


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
    deltas = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        positions = np.concatenate([indices[group] for group in sampled])
        deltas[draw] = (
            accuracy(candidate[positions], gold[positions])
            - accuracy(incumbent[positions], gold[positions])
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
        "--portfolio-dir", type=Path,
        default=ROOT / "artifacts/submission_portfolio_task2_public",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/icr_gates/source_continuity_gate_audit.json",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    metadata = metadata.loc[frame.id]
    gold = frame.answer.astype(str).to_numpy()
    probabilities = (load_probabilities(7) + load_probabilities(19)) / 2
    raw = decode_probabilities(probabilities, ordinal_mean=True)
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])
    ordinal = np.where(rules != "+1", rules, raw)

    prior_path = args.portfolio_dir / "task2_known_source.jsonl"
    prior = np.asarray([
        str(json.loads(line)["answer"])
        for line in prior_path.read_text().splitlines() if line.strip()
    ])
    diagnostics = pd.read_csv(args.portfolio_dir / "task2_source_diagnostics.csv")
    if not (len(frame) == len(prior) == len(diagnostics)):
        raise ValueError("Task 2 source-continuity artifacts are not row-aligned")
    if diagnostics.id.astype(str).tolist() != frame.id.astype(str).tolist():
        raise ValueError("Task 2 source-continuity diagnostic IDs are misaligned")

    agreement = diagnostics.source_agreement.to_numpy(dtype=int)
    similarity = diagnostics.nearest_similarity.to_numpy(dtype=float)
    policies = {
        "ordinal_incumbent": ordinal,
        "whole_packet_known_source": prior,
        "residual_agreement2_similarity016": np.where(
            (agreement >= 2) & (similarity >= 0.16), prior, ordinal,
        ),
        "residual_agreement3_similarity016": np.where(
            (agreement >= 3) & (similarity >= 0.16), prior, ordinal,
        ),
    }
    groups = metadata.source_group.astype(str).to_numpy()
    rows = {}
    for offset, (name, prediction) in enumerate(policies.items()):
        rescued = (prediction == gold) & (ordinal != gold)
        harmed = (prediction != gold) & (ordinal == gold)
        rows[name] = {
            "accuracy": accuracy(prediction, gold),
            "changed_rows": int(np.sum(prediction != ordinal)),
            "rescued_rows": int(np.sum(rescued)),
            "harmed_rows": int(np.sum(harmed)),
            "delta_correct": int(np.sum(rescued) - np.sum(harmed)),
            "source_cluster_bootstrap_vs_ordinal": bootstrap_delta(
                prediction, ordinal, gold, groups,
                draws=args.bootstrap_draws, seed=20260805 + offset,
            ),
        }

    result = {
        "evaluation": {
            "neural_predictions": "five reconstructed-source-disjoint OOF folds",
            "source_prior_predictions": (
                "leave-one-row-out continuity simulation; same-source peers may remain"
            ),
            "rows": len(frame),
            "source_groups": int(len(np.unique(groups))),
            "bootstrap_draws": args.bootstrap_draws,
        },
        "packet_diagnostics": {
            "median_nearest_similarity": float(np.median(similarity)),
            "agreement_ge_2_fraction": float(np.mean(agreement >= 2)),
            "legacy_packet_gate_fires": bool(
                np.median(similarity) >= 0.16 and np.mean(agreement >= 2) >= 0.65
            ),
        },
        "policies": rows,
        "decision": "retain_ordinal_primary_reject_automatic_source_prior_gate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
