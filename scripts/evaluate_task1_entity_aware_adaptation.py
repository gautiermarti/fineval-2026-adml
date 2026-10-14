#!/usr/bin/env python3
"""Evaluate the frozen Task 1 entity-aware fold-0 futility gate."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/task1_entity_aware_adaptation_preregistration.md"
EXPECTED_PROTOCOL_SHA256 = "59fdb499fff4f83771deacb028bd347b8514dbf7852826304e4ef5666fa14153"
BASE_REVISION = "e06fba0eb7a107f401055bb174f9e1ba1a29ba17"
ARMS = ("base", "matched_random_mlm", "whole_entity_mlm")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_downstream(summary: dict, arm: str) -> None:
    required = {
        "arm": arm, "fold": 0, "seed": 7, "epochs": 7,
        "batch_size": 8, "max_length": 512, "learning_rate": 3e-5,
        "weight_decay": 0.01, "dtype": "float32", "device": "cuda",
        "held_rows": 64, "fixed_epoch_no_held_fold_selection": True,
        "validation_used": False, "public_test_used": False,
        "hidden_test_used": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(f"{arm} downstream drift: {key}={summary.get(key)!r}")


def verify_adaptation(summary: dict, mode: str, corpus_hash: str) -> None:
    required = {
        "mode": mode, "model": "spyrosbriakos/greek_legal_bert_v2",
        "revision": BASE_REVISION, "seed": 7, "max_length": 256,
        "batch_size": 16, "learning_rate": 5e-5, "weight_decay": 0.01,
        "max_steps": 256, "mask_probability": 0.15,
        "dtype": "float32", "device": "cuda", "corpus_sha256": corpus_hash,
        "validation_used": False, "public_test_used": False,
        "hidden_test_used": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(f"{mode} adaptation drift: {key}={summary.get(key)!r}")


def read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=ROOT / "artifacts/task1_entity_aware_fold0_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/task1_entity_aware_fold0_v1/evaluation.json",
    )
    args = parser.parse_args()
    if digest(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("entity-aware preregistration changed")

    silver_summary = load_json(args.experiment_dir / "silver/summary.json")
    if silver_summary.get("silver_chunks", 0) < 512 or silver_summary.get("silver_spans", 0) < 1000:
        raise ValueError("silver corpus failed frozen minimum")
    if silver_summary.get("validation_answer_column_loaded") is not False:
        raise ValueError("validation answer access was not excluded")
    corpus_path = args.experiment_dir / "silver/corpus.jsonl"
    corpus_hash = digest(corpus_path)
    if silver_summary.get("corpus_sha256") != corpus_hash:
        raise ValueError("silver corpus hash drift")

    for mode in ("matched_random_mlm", "whole_entity_mlm"):
        verify_adaptation(
            load_json(args.experiment_dir / mode / "adaptation_summary.json"),
            mode, corpus_hash,
        )

    summaries = {
        arm: load_json(args.experiment_dir / "downstream" / arm / "summary.json")
        for arm in ARMS
    }
    predictions = {
        arm: read_predictions(args.experiment_dir / "downstream" / arm / "predictions.jsonl")
        for arm in ARMS
    }
    for arm in ARMS:
        verify_downstream(summaries[arm], arm)
        if summaries[arm]["predictions_sha256"] != digest(
            args.experiment_dir / "downstream" / arm / "predictions.jsonl"
        ):
            raise ValueError(f"{arm} prediction hash drift")
    expected_indices = [row["row_index"] for row in predictions["base"]]
    expected_gold = [row["gold"] for row in predictions["base"]]
    for arm in ARMS[1:]:
        if [row["row_index"] for row in predictions[arm]] != expected_indices:
            raise ValueError(f"{arm} row alignment drift")
        if [row["gold"] for row in predictions[arm]] != expected_gold:
            raise ValueError(f"{arm} gold alignment drift")

    control_arm = max(
        ("matched_random_mlm", "base"),
        key=lambda arm: summaries[arm]["macro_entity_f1"],
    )
    treatment = summaries["whole_entity_mlm"]
    control = summaries[control_arm]
    row_deltas = [
        treatment_row["row_entity_f1"] - control_row["row_entity_f1"]
        for treatment_row, control_row in zip(
            predictions["whole_entity_mlm"], predictions[control_arm], strict=True,
        )
    ]
    type_deltas = {
        kind: treatment["entity_type_f1"][kind] - control["entity_type_f1"][kind]
        for kind in treatment["entity_type_f1"]
    }
    rescues = sum(delta > 1e-12 for delta in row_deltas)
    harms = sum(delta < -1e-12 for delta in row_deltas)
    macro_deltas = {
        arm: treatment["macro_entity_f1"] - summaries[arm]["macro_entity_f1"]
        for arm in ("base", "matched_random_mlm")
    }
    gate = {
        "macro_at_least_0_005_above_base": macro_deltas["base"] >= 0.005,
        "macro_at_least_0_005_above_matched_random_mlm": (
            macro_deltas["matched_random_mlm"] >= 0.005
        ),
        "rescues_strictly_exceed_harms": rescues > harms,
        "at_least_two_types_nonnegative": sum(
            delta >= 0.0 for delta in type_deltas.values()
        ) >= 2,
        "no_type_regresses_more_than_0_01": min(type_deltas.values()) >= -0.01,
        "corpus_and_provenance_checks_passed": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "experiment": "task1_entity_aware_adaptation_fold0",
        "protocol_sha256": digest(PROTOCOL),
        "silver_summary_sha256": digest(args.experiment_dir / "silver/summary.json"),
        "arms": {
            arm: {
                "macro_entity_f1": summaries[arm]["macro_entity_f1"],
                "entity_type_f1": summaries[arm]["entity_type_f1"],
                "predictions_sha256": summaries[arm]["predictions_sha256"],
            } for arm in ARMS
        },
        "stronger_control": control_arm,
        "treatment_macro_deltas": macro_deltas,
        "treatment_type_deltas_vs_stronger_control": type_deltas,
        "rescues": rescues, "harms": harms,
        "ties": len(row_deltas) - rescues - harms,
        "futility_gate": gate,
        "decision": "advance_to_full_seed7_oof" if gate["passed"] else "stop",
        "post_hoc_tuning_allowed": False,
        "validation_used": False, "public_test_used": False,
        "hidden_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
