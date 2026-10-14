#!/usr/bin/env python3
"""Apply the frozen Task 2 human-rationale pilot and promotion gates."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import probabilities, source_bootstrap_delta  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


GENERAL_SEED7 = ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv"
GENERAL_SEED19 = ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv"
POLICY_ALLOWED = {"eligible", "organizer_confirmed_eligible"}


def decode(matrix: np.ndarray, rules: np.ndarray) -> np.ndarray:
    raw = decode_probabilities(matrix, ordinal_mean=True)
    return np.where(rules != "+1", rules, raw)


def comparison(candidate: np.ndarray, incumbent: np.ndarray, gold: np.ndarray) -> dict:
    return {
        "accuracy": float(np.mean(candidate == gold)),
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "rescues": int(np.sum((candidate == gold) & (incumbent != gold))),
        "harms": int(np.sum((candidate != gold) & (incumbent == gold))),
        "changed": int(np.sum(candidate != incumbent)),
    }


def pilot_gate(
    replacement: dict,
    standalone_accuracy: float,
    general_seed7_accuracy: float,
    control_accuracy: float | None,
) -> dict:
    first_stage = {
        "replacement_at_least_34_of_56": replacement["accuracy"] >= 34 / 56,
        "standalone_at_least_34_of_56": standalone_accuracy >= 34 / 56,
        "replacement_harms_at_most_two": replacement["harms"] <= 2,
    }
    # Keep the historical value explicit as a separate audit even though it is
    # numerically the same 34/56 threshold on the frozen pilot fold.
    first_stage["standalone_not_below_general_seed7"] = (
        standalone_accuracy >= general_seed7_accuracy
    )
    first_stage_passed = all(first_stage.values())
    attribution = None if control_accuracy is None else (
        standalone_accuracy >= control_accuracy
    )
    return {
        "first_stage": first_stage,
        "first_stage_passed": first_stage_passed,
        "same_code_control_required": first_stage_passed,
        "standalone_not_below_same_code_control": attribution,
        "survives_preregistered_futility_gate": bool(
            first_stage_passed and attribution is True
        ),
    }


def load_metrics(oof: Path) -> dict:
    path = oof.parent / "metrics.json"
    if not path.is_file():
        raise ValueError(f"missing metrics beside candidate: {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument("--control-oof", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    all_ids = frame.id.to_numpy()
    candidate_table = pd.read_csv(args.candidate_oof)
    ids = candidate_table.id.to_numpy()
    if len(np.unique(ids)) != len(ids) or not set(ids) <= set(all_ids):
        raise ValueError("candidate IDs are duplicate or outside JF-ICR")
    # OOF writers preserve benchmark order; reject silent row reordering.
    expected_ids = frame[frame.id.isin(ids)].id.to_numpy()
    if not np.array_equal(ids, expected_ids):
        raise ValueError("candidate OOF IDs are not in benchmark order")
    selected = frame.set_index("id").loc[ids].reset_index()
    selected_meta = metadata.loc[ids]
    gold = selected.answer.astype(str).to_numpy()
    rules = np.asarray([
        rule_prediction(query, "safe_rules") for query in selected["query"]
    ])
    candidate = probabilities(args.candidate_oof, ids)
    general7 = probabilities(GENERAL_SEED7, all_ids)[frame.id.isin(ids)]
    general19 = probabilities(GENERAL_SEED19, all_ids)[frame.id.isin(ids)]

    matrices = {
        "incumbent_general2": (general7 + general19) / 2,
        "general_seed7": general7,
        "rationale_seed7": candidate,
        "primary_replace_seed7": (candidate + general19) / 2,
        "secondary_general2_plus_rationale": (general7 + general19 + candidate) / 3,
    }
    predictions = {name: decode(matrix, rules) for name, matrix in matrices.items()}
    incumbent = predictions["incumbent_general2"]
    profile_metrics = {
        name: comparison(prediction, incumbent, gold)
        for name, prediction in predictions.items()
    }

    control_accuracy = None
    if args.control_oof:
        control_table = pd.read_csv(args.control_oof)
        if not np.array_equal(control_table.id.to_numpy(), ids):
            raise ValueError("same-code control IDs do not align with candidate")
        control = probabilities(args.control_oof, ids)
        control_prediction = decode(control, rules)
        control_accuracy = float(np.mean(control_prediction == gold))
        profile_metrics["same_code_zero_weight_control"] = comparison(
            control_prediction, incumbent, gold,
        )

    folds = selected_meta.source_fold.to_numpy()
    sources = selected_meta.source_group.to_numpy()
    candidate_metrics = load_metrics(args.candidate_oof)
    smoke = candidate_metrics.get("max_train_steps") is not None
    result = {
        "candidate_oof": str(args.candidate_oof),
        "control_oof": str(args.control_oof) if args.control_oof else None,
        "rows_evaluated": len(ids),
        "folds_evaluated": sorted(int(value) for value in np.unique(folds)),
        "decoder": "equal-probability mean, expected-score rounding, safe rules",
        "profiles": profile_metrics,
        "current_or_hidden_test_used": False,
        "evaluation_kind": "pipeline_smoke_only" if smoke else "source_disjoint_experiment",
        "submission_policy_status": candidate_metrics["submission_policy_status"],
    }
    if not smoke and len(ids) == 56 and set(folds) == {2}:
        result["pilot_gate"] = pilot_gate(
            profile_metrics["primary_replace_seed7"],
            profile_metrics["rationale_seed7"]["accuracy"],
            profile_metrics["general_seed7"]["accuracy"],
            control_accuracy,
        )
    if not smoke and len(ids) == len(frame) and len(np.unique(folds)) == 5:
        primary = predictions["primary_replace_seed7"]
        fold_rows = []
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            fold_rows.append({
                "fold": int(fold),
                "rows": int(mask.sum()),
                "candidate_accuracy": float(np.mean(primary[mask] == gold[mask])),
                "incumbent_accuracy": float(np.mean(incumbent[mask] == gold[mask])),
            })
        improved = sum(
            row["candidate_accuracy"] > row["incumbent_accuracy"] for row in fold_rows
        )
        tied = sum(
            row["candidate_accuracy"] == row["incumbent_accuracy"] for row in fold_rows
        )
        worsened = len(fold_rows) - improved - tied
        bootstrap = source_bootstrap_delta(
            primary, incumbent, gold, sources, draws=args.bootstrap_draws,
        )
        model_gate = bool(
            profile_metrics["primary_replace_seed7"]["accuracy"]
            > profile_metrics["incumbent_general2"]["accuracy"]
            and improved >= 3
            and worsened <= 1
            and bootstrap["bootstrap_probability_positive"] >= 0.90
            and bootstrap["bootstrap_p05_delta"] >= 0.0
        )
        policy_gate = result["submission_policy_status"] in POLICY_ALLOWED
        result["full_gate"] = {
            "folds": fold_rows,
            "folds_improved": improved,
            "folds_tied": tied,
            "folds_worsened": worsened,
            **bootstrap,
            "model_promotion_gate_passed": model_gate,
            "submission_policy_gate_passed": policy_gate,
            "promotion_eligible": bool(model_gate and policy_gate),
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
