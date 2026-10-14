#!/usr/bin/env python3
"""Apply the frozen Task 2 multi-task pilot and promotion gates."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import probabilities, source_bootstrap_delta  # noqa: E402
from evaluate_icr_tsutsuji_semantics import (  # noqa: E402
    LEVELS, OOF_PATHS as INCUMBENT_PATHS,
)
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


RARE = {-2, -1, 2}


def rare_recall(predicted: np.ndarray, gold: np.ndarray) -> float:
    mask = np.isin(gold, list(RARE))
    return float(np.mean(predicted[mask] == gold[mask])) if mask.any() else 0.0


def pilot_decision(
    candidate: np.ndarray, incumbent: np.ndarray, gold: np.ndarray,
) -> dict:
    candidate_accuracy = float(np.mean(candidate == gold))
    incumbent_accuracy = float(np.mean(incumbent == gold))
    harms = int(np.sum((candidate != gold) & (incumbent == gold)))
    rescues = int(np.sum((candidate == gold) & (incumbent != gold)))
    candidate_rare = rare_recall(candidate, gold)
    incumbent_rare = rare_recall(incumbent, gold)
    survives = (
        candidate_accuracy >= incumbent_accuracy
        and (candidate_rare > incumbent_rare or harms <= 2)
    )
    return {
        "candidate_accuracy": candidate_accuracy,
        "incumbent_accuracy": incumbent_accuracy,
        "rescues": rescues,
        "harms": harms,
        "candidate_rare_boundary_recall": candidate_rare,
        "incumbent_rare_boundary_recall": incumbent_rare,
        "survives_preregistered_futility_gate": bool(survives),
    }


def load_candidate(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", *(f"logit_{label}" for label in ("-2", "-1", "0", "+1", "+2"))}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"candidate OOF file lacks columns: {sorted(missing)}")
    if frame.id.duplicated().any():
        raise ValueError(f"duplicate candidate IDs in {path}")
    return frame


def is_smoke_artifact(path: Path) -> bool:
    metrics_path = path.parent / "metrics.json"
    if not metrics_path.is_file():
        return False
    metrics = json.loads(metrics_path.read_text())
    return metrics.get("max_train_steps") is not None


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--candidate-oof", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    candidates = [load_candidate(path).set_index("id") for path in args.candidate_oof]
    evaluated_ids = set(candidates[0].index)
    for candidate in candidates[1:]:
        evaluated_ids.intersection_update(candidate.index)
    ids = frame[frame.id.isin(evaluated_ids)].id.to_numpy()
    if not len(ids):
        raise ValueError("candidate files have no common benchmark rows")
    if len(ids) != len(evaluated_ids):
        raise ValueError("candidate OOF includes unknown benchmark IDs")
    selected = frame.set_index("id").loc[ids].reset_index()
    gold = selected.answer.astype(int).to_numpy()
    rules = np.asarray([
        int(rule_prediction(query, "safe_rules")) for query in selected["query"]
    ])

    logit_columns = [f"logit_{label}" for label in ("-2", "-1", "0", "+1", "+2")]
    candidate_probabilities = np.mean([
        softmax(candidate.loc[ids, logit_columns].to_numpy(dtype=float), axis=1)
        for candidate in candidates
    ], axis=0)
    candidate_prediction = decode_probabilities(
        candidate_probabilities, ordinal_mean=True,
    ).astype(int)
    candidate_prediction = np.where(rules != 1, rules, candidate_prediction)

    evaluated_mask = frame.id.isin(evaluated_ids).to_numpy()
    incumbent_probabilities = np.mean([
        probabilities(path, frame.id.to_numpy())[evaluated_mask]
        for path in INCUMBENT_PATHS
    ], axis=0)
    incumbent_prediction = decode_probabilities(
        incumbent_probabilities, ordinal_mean=True,
    ).astype(int)
    incumbent_prediction = np.where(rules != 1, rules, incumbent_prediction)
    summary = pilot_decision(candidate_prediction, incumbent_prediction, gold)
    smoke = any(is_smoke_artifact(path) for path in args.candidate_oof)
    if smoke:
        summary["survives_preregistered_futility_gate"] = None
    folds = metadata.loc[ids].source_fold.to_numpy()
    sources = metadata.loc[ids].source_group.to_numpy()
    fold_deltas = []
    for fold in sorted(set(folds)):
        mask = folds == fold
        fold_deltas.append({
            "fold": int(fold),
            "rows": int(mask.sum()),
            "candidate_accuracy": float(np.mean(candidate_prediction[mask] == gold[mask])),
            "incumbent_accuracy": float(np.mean(incumbent_prediction[mask] == gold[mask])),
        })
    summary.update({
        "candidate_files": [str(path) for path in args.candidate_oof],
        "rows_evaluated": len(ids),
        "folds_completed": len(fold_deltas),
        "folds": fold_deltas,
        "safe_rules_applied": True,
        "decoder": "equal_probability_mean_then_expected-score_rounding",
        "current_or_hidden_test_used": False,
        "evaluation_kind": "pipeline_smoke_only" if smoke else "source_disjoint_experiment",
        "gate_applied": not smoke,
    })
    if not smoke and len(ids) == len(frame) and len(fold_deltas) == 5:
        bootstrap = source_bootstrap_delta(
            candidate_prediction, incumbent_prediction, gold, sources,
        )
        improved = sum(
            row["candidate_accuracy"] > row["incumbent_accuracy"]
            for row in fold_deltas
        )
        tied = sum(
            row["candidate_accuracy"] == row["incumbent_accuracy"]
            for row in fold_deltas
        )
        worsened = len(fold_deltas) - improved - tied
        summary["full_gate"] = {
            "folds_improved": improved,
            "folds_tied": tied,
            "folds_worsened": worsened,
            **bootstrap,
            "promotion_eligible": bool(
                summary["candidate_accuracy"] >= 0.72
                and improved + tied >= 3
                and worsened <= 1
                and bootstrap["bootstrap_probability_positive"] >= 0.90
            ),
        }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
