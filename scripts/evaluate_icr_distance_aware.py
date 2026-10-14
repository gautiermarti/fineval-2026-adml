#!/usr/bin/env python3
"""Evaluate the preregistered full-fold Task 2 distance-aware seed-7 pilot."""

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

from evaluate_icr_adapted import source_bootstrap_delta  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402
from train_icr_japanese_cv import LABELS  # noqa: E402


PROTOCOL = ROOT / "docs/icr_distance_aware_loss_preregistration.md"
EXPECTED_PROTOCOL_SHA256 = "24f5d725a0ddbad5ed4443883bca2a6240162a51dbf1fd2c01aa98087bc3433f"
EXPECTED_REVISION = "65243d6e5629b969c77309f217bd7b1a79d43c7e"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    logits = frame[[f"logit_{label}" for label in LABELS]].to_numpy(float)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def profile(
    name: str, matrix: np.ndarray, gold: np.ndarray, rules: np.ndarray,
    incumbent: np.ndarray, folds: np.ndarray, sources: np.ndarray,
) -> tuple[dict, np.ndarray]:
    raw = decode_probabilities(matrix, ordinal_mean=True)
    prediction = np.where(rules != "+1", rules, raw)
    correct = prediction == gold
    incumbent_correct = incumbent == gold
    fold_values = sorted(np.unique(folds))
    incumbent_fold = np.asarray([
        np.mean(incumbent[folds == fold] == gold[folds == fold]) for fold in fold_values
    ])
    candidate_fold = np.asarray([
        np.mean(prediction[folds == fold] == gold[folds == fold]) for fold in fold_values
    ])
    return ({
        "name": name, "accuracy": float(np.mean(correct)),
        "correct": int(correct.sum()), "rows": len(gold),
        "rescues": int(np.sum(correct & ~incumbent_correct)),
        "harms": int(np.sum(~correct & incumbent_correct)),
        "changed_predictions": int(np.sum(prediction != incumbent)),
        "folds_improved": int(np.sum(candidate_fold > incumbent_fold)),
        "folds_tied": int(np.sum(candidate_fold == incumbent_fold)),
        "folds_worsened": int(np.sum(candidate_fold < incumbent_fold)),
        "fold_accuracy": {
            str(fold): float(value) for fold, value in zip(fold_values, candidate_fold, strict=True)
        },
        **source_bootstrap_delta(prediction, incumbent, gold, sources),
    }, prediction)


def advance_gate(profiles: dict[str, dict]) -> dict[str, bool]:
    incumbent = profiles["incumbent_general2"]
    primary = profiles["primary_replace_general_seed7"]
    secondary = profiles["secondary_general2_plus_distance_seed7"]
    checks = {
        "primary_at_least_one_more_correct": primary["correct"] >= incumbent["correct"] + 1,
        "primary_rescues_exceed_harms": primary["rescues"] > primary["harms"],
        "primary_at_least_two_folds_improved": primary["folds_improved"] >= 2,
        "primary_at_most_one_fold_worsened": primary["folds_worsened"] <= 1,
        "secondary_not_below_incumbent": secondary["correct"] >= incumbent["correct"],
    }
    return checks | {"advance_seed19": all(checks.values())}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--candidate-dir", type=Path,
        default=ROOT / "artifacts/task2_distance_aware_seed7_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/task2_distance_aware_seed7_v1/evaluation.json",
    )
    args = parser.parse_args()
    if digest(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("distance-aware preregistration changed")
    summary = json.loads((args.candidate_dir / "summary.json").read_text())
    required = {
        "model": "cl-tohoku/bert-base-japanese-v3", "revision": EXPECTED_REVISION,
        "seed": 7, "epochs": 4, "batch_size": 8, "max_length": 512,
        "learning_rate": 2e-5, "weight_decay": 0.01, "distance_weight": 0.5,
        "dtype": "float32", "folds_completed": [0, 1, 2, 3, 4],
        "rows_evaluated": 253, "current_or_hidden_test_used": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(f"candidate summary changed: {key}={summary.get(key)!r}")

    benchmark = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id").loc[benchmark.id]
    ids = benchmark.id.to_numpy()
    gold = benchmark.answer.astype(str).to_numpy()
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in benchmark["query"]])

    matrices = {}
    for name, path in (
        ("general_seed7", ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv"),
        ("general_seed19", ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv"),
        ("distance_seed7", args.candidate_dir / "oof_predictions.csv"),
    ):
        frame = pd.read_csv(path, dtype={"gold": str, "prediction": str})
        if not np.array_equal(frame.id.to_numpy(), ids):
            raise ValueError(f"misaligned OOF file: {path}")
        matrices[name] = probabilities(frame)
    incumbent_matrix = (matrices["general_seed7"] + matrices["general_seed19"]) / 2
    incumbent_raw = decode_probabilities(incumbent_matrix, ordinal_mean=True)
    incumbent = np.where(rules != "+1", rules, incumbent_raw)
    profile_matrices = {
        "incumbent_general2": incumbent_matrix,
        "distance_seed7": matrices["distance_seed7"],
        "primary_replace_general_seed7": (
            matrices["distance_seed7"] + matrices["general_seed19"]
        ) / 2,
        "secondary_general2_plus_distance_seed7": (
            matrices["general_seed7"] + matrices["general_seed19"]
            + matrices["distance_seed7"]
        ) / 3,
    }
    profiles, predictions = {}, {}
    for name, matrix in profile_matrices.items():
        profiles[name], predictions[name] = profile(
            name, matrix, gold, rules, incumbent, folds, sources,
        )
    gate = advance_gate(profiles)
    changed_rows = {}
    for name in profile_matrices:
        if name == "incumbent_general2":
            continue
        prediction = predictions[name]
        changed_rows[name] = [
            {
                "id": int(ids[index]), "gold": str(gold[index]),
                "incumbent": str(incumbent[index]), "candidate": str(prediction[index]),
            }
            for index in np.flatnonzero(prediction != incumbent)
        ]
    result = {
        "experiment": "task2_distance_aware_seed7_full_source_cv",
        "protocol_sha256": digest(PROTOCOL),
        "candidate_oof_sha256": digest(args.candidate_dir / "oof_predictions.csv"),
        "profiles": profiles, "advance_gate": gate,
        "decision": "train_seed19" if gate["advance_seed19"] else "reject_and_stop",
        "posthoc_changed_rows_not_used_for_selection": changed_rows,
        "current_or_hidden_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
