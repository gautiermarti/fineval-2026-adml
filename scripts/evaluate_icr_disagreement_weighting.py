#!/usr/bin/env python3
"""Evaluate the preregistered Task 2 disagreement-weighting pilot."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402
from train_icr_japanese_cv import LABELS  # noqa: E402


PROTOCOL = ROOT / "docs/icr_disagreement_weighting_preregistration.md"
EXPECTED_PROTOCOL_SHA256 = (
    "c6d5199b28fe6a95ad17865b24b84a31ceadd71372ee4a0b8791a17269b23fb0"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    logits = frame[[f"logit_{label}" for label in LABELS]].to_numpy(float)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def profile(
    name: str,
    values: np.ndarray,
    gold: np.ndarray,
    rules: np.ndarray,
    incumbent: np.ndarray,
) -> tuple[dict, np.ndarray]:
    raw = decode_probabilities(values, ordinal_mean=True)
    prediction = np.where(rules != "+1", rules, raw)
    incumbent_correct = incumbent == gold
    candidate_correct = prediction == gold
    return ({
        "name": name,
        "accuracy": float(np.mean(candidate_correct)),
        "correct": int(candidate_correct.sum()),
        "rows": len(gold),
        "rescues": int(np.sum(candidate_correct & ~incumbent_correct)),
        "harms": int(np.sum(~candidate_correct & incumbent_correct)),
        "changed_predictions": int(np.sum(prediction != incumbent)),
    }, prediction)


def pilot_gate(profiles: dict[str, dict]) -> dict[str, bool]:
    standalone = profiles["weighted_seed7"]
    primary = profiles["primary_replace_seed7"]
    secondary = profiles["secondary_general2_plus_weighted"]
    checks = {
        "standalone_at_least_34_of_56": standalone["correct"] >= 34,
        "primary_at_least_35_of_56": primary["correct"] >= 35,
        "primary_rescues_exceed_harms": primary["rescues"] > primary["harms"],
        "primary_harms_at_most_two": primary["harms"] <= 2,
        "secondary_not_below_incumbent_34_of_56": secondary["correct"] >= 34,
    }
    return checks | {"pilot_passed": all(checks.values())}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/task2_disagreement_weighted_fold2_seed7_v1/evaluation.json",
    )
    args = parser.parse_args()
    if digest(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("preregistration file changed")

    benchmark = pd.read_parquet(
        ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    )
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    candidate_frame = pd.read_csv(
        args.candidate_oof, dtype={"gold": str, "prediction": str},
    )
    candidate_ids = candidate_frame.id.to_numpy()
    positions = benchmark.reset_index().set_index("id").loc[candidate_ids]["index"].to_numpy()
    if len(candidate_ids) != 56 or set(metadata.loc[candidate_ids].source_fold) != {2}:
        raise ValueError("pilot must contain exactly the 56 source-fold-2 rows")

    general = []
    for seed in (7, 19):
        frame = pd.read_csv(
            ROOT / f"artifacts/task2_japanese_cv/seed{seed}/oof_predictions.csv",
            dtype={"gold": str, "prediction": str},
        )
        if not np.array_equal(frame.id.to_numpy(), benchmark.id.to_numpy()):
            raise ValueError(f"general seed {seed} is misaligned")
        general.append(probabilities(frame)[positions])
    candidate = probabilities(candidate_frame)
    gold = benchmark.answer.astype(str).to_numpy()[positions]
    rules = np.asarray([
        rule_prediction(query, "safe_rules")
        for query in benchmark.iloc[positions]["query"]
    ])
    incumbent_raw = decode_probabilities((general[0] + general[1]) / 2, ordinal_mean=True)
    incumbent = np.where(rules != "+1", rules, incumbent_raw)

    matrices = {
        "incumbent_general2": (general[0] + general[1]) / 2,
        "general_seed7": general[0],
        "weighted_seed7": candidate,
        "primary_replace_seed7": (candidate + general[1]) / 2,
        "secondary_general2_plus_weighted": (
            general[0] + general[1] + candidate
        ) / 3,
    }
    profile_rows = {}
    predictions = {}
    for name, matrix in matrices.items():
        row, prediction = profile(name, matrix, gold, rules, incumbent)
        profile_rows[name] = row
        predictions[name] = prediction
    gate = pilot_gate(profile_rows)
    posthoc_diagnostics = {}
    incumbent_correct = incumbent == gold
    for name in ("weighted_seed7", "primary_replace_seed7", "secondary_general2_plus_weighted"):
        prediction = predictions[name]
        candidate_correct = prediction == gold
        rescue_mask = candidate_correct & ~incumbent_correct
        harm_mask = ~candidate_correct & incumbent_correct
        posthoc_diagnostics[name] = {
            "rescue_gold_labels": {
                label: int(np.sum(rescue_mask & (gold == label))) for label in LABELS
            },
            "harm_gold_labels": {
                label: int(np.sum(harm_mask & (gold == label))) for label in LABELS
            },
            "changed_rows": [
                {
                    "id": int(candidate_ids[index]),
                    "gold": str(gold[index]),
                    "incumbent": str(incumbent[index]),
                    "candidate": str(prediction[index]),
                }
                for index in np.flatnonzero(prediction != incumbent)
            ],
        }
    result = {
        "experiment": "icr_disagreement_weighted_hard_gold_fold2_pilot",
        "protocol_sha256": digest(PROTOCOL),
        "candidate_oof": str(args.candidate_oof),
        "candidate_oof_sha256": digest(args.candidate_oof),
        "rows": len(gold),
        "folds": [2],
        "profiles": profile_rows,
        "pilot_gate": gate,
        "posthoc_diagnostics_not_used_for_selection": posthoc_diagnostics,
        "decision": (
            "advance_to_full_confirmation" if gate["pilot_passed"]
            else "reject_and_stop"
        ),
        "current_or_hidden_test_used": False,
        "submission_policy_status": "policy_gated_pending_license_clarification",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
