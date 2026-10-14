#!/usr/bin/env python3
"""Evaluate the discovery-separated rationale diversity confirmation."""

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
from evaluate_icr_human_rationale import comparison, decode, load_metrics  # noqa: E402
from train_icr_japanese_cv import LABELS  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402


GENERAL = {
    seed: ROOT / f"artifacts/task2_japanese_cv/seed{seed}/oof_predictions.csv"
    for seed in (7, 19, 31)
}
DEFAULT_DISCOVERY = (
    ROOT / "artifacts/task2_human_rationale_fold2_seed7_v1/oof_predictions.csv"
)
POLICY_ALLOWED = {"eligible", "organizer_confirmed_eligible"}


def merged_probabilities(paths: list[Path], benchmark_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tables = [pd.read_csv(path) for path in paths]
    table = pd.concat(tables, ignore_index=True)
    if table.id.duplicated().any():
        raise ValueError("candidate/control OOF paths contain duplicate IDs")
    positions = {value: index for index, value in enumerate(benchmark_ids)}
    if not set(table.id) <= set(benchmark_ids):
        raise ValueError("candidate/control OOF contains unknown IDs")
    table["_position"] = table.id.map(positions)
    table = table.sort_values("_position")
    logits = table[[f"logit_{label}" for label in LABELS]].to_numpy(dtype=float)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return table.id.to_numpy(), values / values.sum(axis=1, keepdims=True)


def initial_confirmation_gate(candidate: dict, generic_control_accuracy: float) -> dict:
    conditions = {
        "candidate_at_least_incumbent": (
            candidate["accuracy"] >= candidate["incumbent_accuracy"]
        ),
        "candidate_harms_at_most_two": candidate["harms"] <= 2,
        "candidate_at_least_generic_three_seed_control": (
            candidate["accuracy"] >= generic_control_accuracy
        ),
    }
    return {**conditions, "passed": all(conditions.values())}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--candidate-oof", type=Path, nargs="+", required=True)
    parser.add_argument("--control-oof", type=Path, nargs="+")
    parser.add_argument("--discovery-oof", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    all_ids = frame.id.to_numpy()
    ids, rationale_probability = merged_probabilities(args.candidate_oof, all_ids)
    selected = frame.set_index("id").loc[ids].reset_index()
    meta = metadata.loc[ids]
    folds = meta.source_fold.to_numpy()
    sources = meta.source_group.to_numpy()
    gold = selected.answer.astype(str).to_numpy()
    rules = np.asarray([
        rule_prediction(query, "safe_rules") for query in selected["query"]
    ])
    row_mask = frame.id.isin(ids).to_numpy()
    general = {
        seed: probabilities(path, all_ids)[row_mask]
        for seed, path in GENERAL.items()
    }
    matrices = {
        "incumbent_general2": (general[7] + general[19]) / 2,
        "generic_general3_control": (general[7] + general[19] + general[31]) / 3,
        "rationale_general3_candidate": (
            general[7] + general[19] + rationale_probability
        ) / 3,
    }
    predictions = {name: decode(matrix, rules) for name, matrix in matrices.items()}
    incumbent = predictions["incumbent_general2"]
    profiles = {
        name: comparison(prediction, incumbent, gold)
        for name, prediction in predictions.items()
    }
    candidate_metrics = [load_metrics(path) for path in args.candidate_oof]
    smoke = any(row.get("max_train_steps") is not None for row in candidate_metrics)
    policy_statuses = {row["submission_policy_status"] for row in candidate_metrics}
    if len(policy_statuses) != 1:
        raise ValueError("candidate artifacts disagree on submission policy")
    result = {
        "candidate_oof": [str(path) for path in args.candidate_oof],
        "control_oof": [str(path) for path in args.control_oof] if args.control_oof else None,
        "rows_evaluated": len(ids),
        "folds_evaluated": sorted(int(value) for value in np.unique(folds)),
        "discovery_fold_excluded_from_confirmation": 2 not in set(folds),
        "profiles": profiles,
        "decoder": "equal-probability mean, expected-score rounding, safe rules",
        "evaluation_kind": "pipeline_smoke_only" if smoke else "source_disjoint_confirmation",
        "submission_policy_status": next(iter(policy_statuses)),
        "current_or_hidden_test_used": False,
    }
    rationale_profile = profiles["rationale_general3_candidate"]
    generic_accuracy = profiles["generic_general3_control"]["accuracy"]
    initial = initial_confirmation_gate(rationale_profile, generic_accuracy)
    result["initial_confirmation_gate"] = initial if not smoke else None

    control_prediction = None
    if args.control_oof:
        control_ids, control_probability = merged_probabilities(args.control_oof, all_ids)
        if not np.array_equal(control_ids, ids):
            raise ValueError("zero-weight control IDs do not align with rationale IDs")
        control_prediction = decode(
            (general[7] + general[19] + control_probability) / 3, rules,
        )
        profiles["same_code_zero_weight_general3_control"] = comparison(
            control_prediction, incumbent, gold,
        )
        result["initial_attribution_gate"] = bool(
            rationale_profile["accuracy"]
            >= profiles["same_code_zero_weight_general3_control"]["accuracy"]
        ) if not smoke else None

    confirmatory_folds = {0, 1, 3, 4}
    if (
        not smoke and set(folds) == confirmatory_folds
        and len(ids) == int((metadata.source_fold != 2).sum())
        and control_prediction is not None
    ):
        primary = predictions["rationale_general3_candidate"]
        fold_rows = []
        for fold in sorted(confirmatory_folds):
            mask = folds == fold
            fold_rows.append({
                "fold": fold,
                "rows": int(mask.sum()),
                "candidate_accuracy": float(np.mean(primary[mask] == gold[mask])),
                "incumbent_accuracy": float(np.mean(incumbent[mask] == gold[mask])),
                "zero_weight_accuracy": float(np.mean(control_prediction[mask] == gold[mask])),
            })
        improved = sum(row["candidate_accuracy"] > row["incumbent_accuracy"] for row in fold_rows)
        tied = sum(row["candidate_accuracy"] == row["incumbent_accuracy"] for row in fold_rows)
        worsened = len(fold_rows) - improved - tied
        versus_incumbent = source_bootstrap_delta(
            primary, incumbent, gold, sources, draws=args.bootstrap_draws,
        )
        versus_control = source_bootstrap_delta(
            primary, control_prediction, gold, sources, draws=args.bootstrap_draws,
        )
        # Restore discovery fold only for the final sanity check, never for a
        # confirmatory condition or bootstrap above.
        discovery_ids, discovery_probability = merged_probabilities(
            [args.discovery_oof], all_ids,
        )
        discovery_frame = frame.set_index("id").loc[discovery_ids].reset_index()
        discovery_mask = frame.id.isin(discovery_ids).to_numpy()
        discovery_rules = np.asarray([
            rule_prediction(query, "safe_rules")
            for query in discovery_frame["query"]
        ])
        discovery_candidate = decode(
            (
                probabilities(GENERAL[7], all_ids)[discovery_mask]
                + probabilities(GENERAL[19], all_ids)[discovery_mask]
                + discovery_probability
            ) / 3,
            discovery_rules,
        )
        discovery_gold = discovery_frame.answer.astype(str).to_numpy()
        full_correct = int(np.sum(primary == gold) + np.sum(discovery_candidate == discovery_gold))
        full_accuracy = full_correct / len(frame)
        model_gate = bool(
            rationale_profile["accuracy"] > profiles["incumbent_general2"]["accuracy"]
            and rationale_profile["accuracy"]
            > profiles["same_code_zero_weight_general3_control"]["accuracy"]
            and rationale_profile["accuracy"] > generic_accuracy
            and improved >= 2
            and worsened <= 1
            and versus_incumbent["bootstrap_probability_positive"] >= 0.90
            and versus_incumbent["bootstrap_p05_delta"] >= 0.0
            and versus_control["bootstrap_probability_positive"] >= 0.80
            and full_accuracy > 0.691700
        )
        policy_gate = result["submission_policy_status"] in POLICY_ALLOWED
        result["full_confirmation_gate"] = {
            "folds": fold_rows,
            "folds_improved": improved,
            "folds_tied": tied,
            "folds_worsened": worsened,
            "versus_incumbent": versus_incumbent,
            "versus_zero_weight_control": versus_control,
            "full_253_accuracy_including_discovery_fold": full_accuracy,
            "model_confirmation_gate_passed": model_gate,
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
