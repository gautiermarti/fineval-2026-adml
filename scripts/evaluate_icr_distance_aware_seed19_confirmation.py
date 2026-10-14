#!/usr/bin/env python3
"""Evaluate the frozen seed-19 distance-aware diversity confirmation."""

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

from evaluate_icr_distance_aware import probabilities, profile  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


PROTOCOL = (
    ROOT
    / "docs/icr_distance_aware_seed19_diversity_confirmation_preregistration.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "c6b4ae27a662f74a1cac3f4c12cb9874fdb2a28cef2b8fca2f7ce1c86b9dc030"
)
EXPECTED_REVISION = "65243d6e5629b969c77309f217bd7b1a79d43c7e"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_distance_summary(path: Path, seed: int) -> dict:
    summary = json.loads(path.read_text())
    required = {
        "model": "cl-tohoku/bert-base-japanese-v3",
        "revision": EXPECTED_REVISION,
        "seed": seed,
        "epochs": 4,
        "batch_size": 8,
        "max_length": 512,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "distance_weight": 0.5,
        "dtype": "float32",
        "folds_completed": [0, 1, 2, 3, 4],
        "rows_evaluated": 253,
        "current_or_hidden_test_used": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"distance seed {seed} summary changed: "
                f"{key}={summary.get(key)!r}"
            )
    return summary


def confirmation_gate(profiles: dict[str, dict]) -> dict[str, bool]:
    incumbent = profiles["incumbent_general2"]
    replication = profiles["replication_general2_plus_distance19"]
    final = profiles["final_general2_plus_distance2"]
    checks = {
        "replication_strictly_above_incumbent": (
            replication["correct"] > incumbent["correct"]
        ),
        "replication_rescues_exceed_harms": (
            replication["rescues"] > replication["harms"]
        ),
        "replication_at_least_two_folds_improved": (
            replication["folds_improved"] >= 2
        ),
        "replication_at_most_one_fold_worsened": (
            replication["folds_worsened"] <= 1
        ),
        "replication_bootstrap_probability_at_least_0_80": (
            replication["bootstrap_probability_positive"] >= 0.80
        ),
        "replication_bootstrap_p05_nonnegative": (
            replication["bootstrap_p05_delta"] >= 0.0
        ),
        "final_strictly_above_incumbent": final["correct"] > incumbent["correct"],
        "final_at_least_three_folds_improved": final["folds_improved"] >= 3,
        "final_at_most_one_fold_worsened": final["folds_worsened"] <= 1,
        "final_at_least_two_more_rescues_than_harms": (
            final["rescues"] >= final["harms"] + 2
        ),
        "final_bootstrap_probability_at_least_0_90": (
            final["bootstrap_probability_positive"] >= 0.90
        ),
        "final_bootstrap_p05_nonnegative": final["bootstrap_p05_delta"] >= 0.0,
        "no_current_or_hidden_test_used": True,
    }
    return checks | {"passed": all(checks.values())}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--seed7-dir",
        type=Path,
        default=ROOT / "artifacts/task2_distance_aware_seed7_v1",
    )
    parser.add_argument(
        "--seed19-dir",
        type=Path,
        default=ROOT / "artifacts/task2_distance_aware_seed19_confirmation_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "artifacts/task2_distance_aware_seed19_confirmation_v1/evaluation.json"
        ),
    )
    args = parser.parse_args()
    if digest(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("seed-19 confirmation preregistration changed")
    verify_distance_summary(args.seed7_dir / "summary.json", 7)
    verify_distance_summary(args.seed19_dir / "summary.json", 19)

    benchmark = pd.read_parquet(
        ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    )
    metadata = (
        pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")
        .set_index("id")
        .loc[benchmark.id]
    )
    ids = benchmark.id.to_numpy()
    gold = benchmark.answer.astype(str).to_numpy()
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray(
        [rule_prediction(query, "safe_rules") for query in benchmark["query"]]
    )

    paths = {
        "general_seed7": (
            ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv"
        ),
        "general_seed19": (
            ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv"
        ),
        "distance_seed7": args.seed7_dir / "oof_predictions.csv",
        "distance_seed19": args.seed19_dir / "oof_predictions.csv",
    }
    matrices = {}
    for name, path in paths.items():
        frame = pd.read_csv(path, dtype={"gold": str, "prediction": str})
        if not np.array_equal(frame.id.to_numpy(), ids):
            raise ValueError(f"misaligned OOF file: {path}")
        matrices[name] = probabilities(frame)

    incumbent_matrix = (matrices["general_seed7"] + matrices["general_seed19"]) / 2
    incumbent_raw = decode_probabilities(incumbent_matrix, ordinal_mean=True)
    incumbent = np.where(rules != "+1", rules, incumbent_raw)
    profile_matrices = {
        "incumbent_general2": incumbent_matrix,
        "replication_general2_plus_distance19": (
            matrices["general_seed7"]
            + matrices["general_seed19"]
            + matrices["distance_seed19"]
        ) / 3,
        "distance_two_seed": (
            matrices["distance_seed7"] + matrices["distance_seed19"]
        ) / 2,
        "final_general2_plus_distance2": sum(matrices.values()) / 4,
    }
    profiles, predictions = {}, {}
    for name, matrix in profile_matrices.items():
        profiles[name], predictions[name] = profile(
            name, matrix, gold, rules, incumbent, folds, sources,
        )

    gate = confirmation_gate(profiles)
    changed_rows = {}
    for name, prediction in predictions.items():
        if name == "incumbent_general2":
            continue
        changed_rows[name] = [
            {
                "id": int(ids[index]),
                "gold": str(gold[index]),
                "incumbent": str(incumbent[index]),
                "candidate": str(prediction[index]),
            }
            for index in np.flatnonzero(prediction != incumbent)
        ]

    result = {
        "experiment": "task2_distance_aware_seed19_diversity_confirmation",
        "protocol_sha256": digest(PROTOCOL),
        "oof_sha256": {name: digest(path) for name, path in paths.items()},
        "profiles": profiles,
        "promotion_gate": gate,
        "decision": (
            "advance_to_full_refits_and_secondary_portfolio_lane"
            if gate["passed"]
            else "stop"
        ),
        "original_seed7_stop_decision_overridden": False,
        "posthoc_motivation_disclosed": True,
        "posthoc_changed_rows_not_used_for_selection": changed_rows,
        "current_or_hidden_test_used": False,
        "post_hoc_tuning_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
