#!/usr/bin/env python3
"""Evaluate the preregistered public-label transfer pilot for GRFinNER."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import entity_type_f1  # noqa: E402


BASELINE_SCORE = 0.857348901098901


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def promotion_gate(
    pilot_score: float,
    baseline_types: dict[str, float],
    pilot_types: dict[str, float],
    improved_rows: int,
    provenance_ok: bool,
) -> dict[str, bool]:
    type_deltas = {
        kind: pilot_types[kind] - baseline_types[kind]
        for kind in baseline_types
    }
    checks = {
        "macro_strictly_above_frozen_baseline": pilot_score > BASELINE_SCORE,
        "at_least_two_types_nonnegative": sum(
            delta >= 0.0 for delta in type_deltas.values()
        ) >= 2,
        "no_type_regresses_more_than_0.01": min(type_deltas.values()) >= -0.01,
        "gain_not_from_single_row": improved_rows >= 2,
        "public_test_provenance_explicit": provenance_ok,
    }
    return checks | {"passed": all(checks.values())}


def attribution_gate(
    pilot_score: float,
    control_score: float,
    pilot_types: dict[str, float],
    control_types: dict[str, float],
) -> dict[str, bool]:
    type_deltas = {
        kind: pilot_types[kind] - control_types[kind]
        for kind in control_types
    }
    checks = {
        "macro_strictly_above_same_code_control": pilot_score > control_score,
        "at_least_two_types_nonnegative_vs_control": sum(
            delta >= 0.0 for delta in type_deltas.values()
        ) >= 2,
        "no_type_regresses_more_than_0.01_vs_control": min(type_deltas.values()) >= -0.01,
    }
    return checks | {"passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir", type=Path,
        default=ROOT / "artifacts/task1_textual_train_plus_public_test_seed7_v1",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "artifacts/task1_textual/validation_legal_base_seed7.jsonl",
    )
    parser.add_argument(
        "--same-code-control-dir", type=Path,
        help="Current-code train-only run required by the fairness amendment.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_all_released_labels_pilot_v1",
    )
    args = parser.parse_args()

    validation = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    gold = [parse_answer(row.answer) for row in validation.itertuples()]

    baseline_records = load_jsonl(args.baseline)
    pilot_records = load_jsonl(args.pilot_dir / "evaluation_predictions.jsonl")
    if len(baseline_records) != len(validation) or len(pilot_records) != len(validation):
        raise ValueError("prediction rows must exactly match the 80-row validation split")
    if [int(row["source_row_index"]) for row in pilot_records] != list(range(len(validation))):
        raise ValueError("pilot prediction row indices are missing, duplicated, or reordered")
    baseline = [parse_answer(str(record["answer"])) for record in baseline_records]
    pilot = [
        [(str(entity), str(kind)) for entity, kind in record["entities"]]
        for record in pilot_records
    ]

    baseline_rows = np.array([
        score(target, prediction)[2]
        for target, prediction in zip(gold, baseline, strict=True)
    ])
    pilot_rows = np.array([
        score(target, prediction)[2]
        for target, prediction in zip(gold, pilot, strict=True)
    ])
    deltas = pilot_rows - baseline_rows
    baseline_score = float(baseline_rows.mean())
    pilot_score = float(pilot_rows.mean())
    if abs(baseline_score - BASELINE_SCORE) > 1e-12:
        raise ValueError(
            f"baseline drift: expected {BASELINE_SCORE:.12f}, got {baseline_score:.12f}"
        )

    baseline_types = entity_type_f1(gold, baseline)
    pilot_types = entity_type_f1(gold, pilot)
    metrics = json.loads((args.pilot_dir / "metrics.json").read_text())
    provenance_ok = bool(
        metrics.get("test_split_used_for_training") is True
        and metrics.get("evaluation_split") == "validation"
        and metrics.get("train_splits") == ["train", "test"]
        and any(row.get("split") == "test" and row.get("sha256")
                for row in metrics.get("train_provenance", []))
    )
    gate = promotion_gate(
        pilot_score, baseline_types, pilot_types,
        int((deltas > 1e-12).sum()), provenance_ok,
    )

    control_summary = None
    attribution = {"passed": False, "status": "same_code_control_pending"}
    if args.same_code_control_dir is not None:
        control_records = load_jsonl(
            args.same_code_control_dir / "evaluation_predictions.jsonl"
        )
        if len(control_records) != len(validation):
            raise ValueError("same-code control must match the validation row count")
        control = [
            [(str(entity), str(kind)) for entity, kind in record["entities"]]
            for record in control_records
        ]
        control_rows = np.array([
            score(target, prediction)[2]
            for target, prediction in zip(gold, control, strict=True)
        ])
        control_score = float(control_rows.mean())
        control_types = entity_type_f1(gold, control)
        attribution = attribution_gate(
            pilot_score, control_score, pilot_types, control_types
        )
        control_summary = {
            "macro_entity_f1": control_score,
            "pilot_delta": pilot_score - control_score,
            "type_f1": control_types,
            "pilot_type_deltas": {
                kind: pilot_types[kind] - control_types[kind]
                for kind in control_types
            },
            "improved_rows": int((pilot_rows - control_rows > 1e-12).sum()),
            "harmed_rows": int((pilot_rows - control_rows < -1e-12).sum()),
        }

    rng = np.random.default_rng(260805)
    draws = deltas[rng.integers(0, len(deltas), size=(10_000, len(deltas)))].mean(axis=1)
    summary = {
        "evaluation": "train+published-test labels to held-out validation",
        "current_or_hidden_test_used": False,
        "external_public_test_labels_used_for_training": True,
        "baseline_macro_entity_f1": baseline_score,
        "pilot_macro_entity_f1": pilot_score,
        "delta": pilot_score - baseline_score,
        "improved_rows": int((deltas > 1e-12).sum()),
        "harmed_rows": int((deltas < -1e-12).sum()),
        "tied_rows": int((abs(deltas) <= 1e-12).sum()),
        "baseline_type_f1": baseline_types,
        "pilot_type_f1": pilot_types,
        "type_deltas": {
            kind: pilot_types[kind] - baseline_types[kind]
            for kind in baseline_types
        },
        "row_bootstrap": {
            "resamples": 10_000,
            "p_gain": float((draws > 0).mean()),
            "p_nonnegative": float((draws >= 0).mean()),
            "p05": float(np.quantile(draws, 0.05)),
            "median": float(np.quantile(draws, 0.50)),
            "p95": float(np.quantile(draws, 0.95)),
        },
        "promotion_gate": gate,
        "same_code_control": control_summary,
        "attribution_gate": attribution,
        "decision": (
            "advance_policy_gated_all_release_refit"
            if gate["passed"] and attribution["passed"]
            else "control_pending" if args.same_code_control_dir is None
            else "stop"
        ),
        "pilot_metrics": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output_dir / "row_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "row_index", "baseline_f1", "pilot_f1", "delta",
        ])
        writer.writeheader()
        for index, (baseline_f1, pilot_f1, delta) in enumerate(
            zip(baseline_rows, pilot_rows, deltas, strict=True)
        ):
            writer.writerow({
                "row_index": index,
                "baseline_f1": float(baseline_f1),
                "pilot_f1": float(pilot_f1),
                "delta": float(delta),
            })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
