#!/usr/bin/env python3
"""Evaluate the preregistered second-encoder public-label replication."""

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


BASELINE_SCORE = 0.8580975274725274


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replication_gate(
    pilot_score: float,
    baseline_types: dict[str, float],
    pilot_types: dict[str, float],
    improved_rows: int,
    bootstrap_p_gain: float,
    provenance_ok: bool,
) -> dict[str, bool]:
    type_deltas = {
        kind: pilot_types[kind] - baseline_types[kind]
        for kind in baseline_types
    }
    checks = {
        "macro_strictly_above_frozen_baseline": pilot_score > BASELINE_SCORE,
        "at_least_two_types_nonnegative": sum(
            delta >= 0 for delta in type_deltas.values()
        ) >= 2,
        "no_type_regresses_more_than_0.01": min(type_deltas.values()) >= -0.01,
        "at_least_two_rows_improve": improved_rows >= 2,
        "bootstrap_p_gain_at_least_0.80": bootstrap_p_gain >= 0.80,
        "public_test_provenance_explicit": provenance_ok,
    }
    return checks | {"passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir", type=Path,
        default=ROOT / "artifacts/task1_textual_train_plus_public_test_legal_ner_seed7_v1",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "artifacts/task1_textual/validation_legal_ner_init_seed7.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_all_released_labels_second_encoder_v1",
    )
    args = parser.parse_args()

    validation = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    gold = [parse_answer(row.answer) for row in validation.itertuples()]
    baseline_records = load_jsonl(args.baseline)
    pilot_records = load_jsonl(args.pilot_dir / "evaluation_predictions.jsonl")
    if len(baseline_records) != 80 or len(pilot_records) != 80:
        raise ValueError("both prediction files must contain all 80 validation rows")
    if [int(row["source_row_index"]) for row in pilot_records] != list(range(80)):
        raise ValueError("pilot rows are missing, duplicated, or reordered")

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
    baseline_score = float(baseline_rows.mean())
    if abs(baseline_score - BASELINE_SCORE) > 1e-12:
        raise ValueError("frozen baseline prediction artifact has drifted")
    pilot_score = float(pilot_rows.mean())
    deltas = pilot_rows - baseline_rows
    baseline_types = entity_type_f1(gold, baseline)
    pilot_types = entity_type_f1(gold, pilot)

    rng = np.random.default_rng(260806)
    draws = deltas[rng.integers(0, len(deltas), size=(10_000, len(deltas)))].mean(axis=1)
    p_gain = float((draws > 0).mean())
    metrics = json.loads((args.pilot_dir / "metrics.json").read_text())
    provenance_ok = bool(
        metrics.get("train_splits") == ["train", "test"]
        and metrics.get("evaluation_split") == "validation"
        and metrics.get("test_split_used_for_training") is True
        and any(row.get("split") == "test" and row.get("sha256")
                for row in metrics.get("train_provenance", []))
    )
    gate = replication_gate(
        pilot_score, baseline_types, pilot_types,
        int((deltas > 1e-12).sum()), p_gain, provenance_ok,
    )
    summary = {
        "evaluation": "second encoder train+published-test to held-out validation",
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
            "p_gain": p_gain,
            "p_nonnegative": float((draws >= 0).mean()),
            "p05": float(np.quantile(draws, 0.05)),
            "median": float(np.quantile(draws, 0.50)),
            "p95": float(np.quantile(draws, 0.95)),
        },
        "replication_gate": gate,
        "decision": "retain_second_all_label_encoder" if gate["passed"] else "remove",
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
        for index, (base, candidate, delta) in enumerate(
            zip(baseline_rows, pilot_rows, deltas, strict=True)
        ):
            writer.writerow({
                "row_index": index, "baseline_f1": float(base),
                "pilot_f1": float(candidate), "delta": float(delta),
            })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
