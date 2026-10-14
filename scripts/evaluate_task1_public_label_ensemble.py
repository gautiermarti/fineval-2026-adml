#!/usr/bin/env python3
"""Evaluate the preregistered public-label member replacement in Task 1."""

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


BASELINE_SCORE = 0.9167834249084249


def load_answers(path: Path) -> list[list[tuple[str, str]]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 80 or [int(row.get("row_index", -1)) for row in records] != list(range(80)):
        raise ValueError(f"{path}: expected exactly 80 aligned prediction rows")
    return [parse_answer(str(row["answer"])) for row in records]


def ensemble_gate(
    candidate_score: float,
    baseline_types: dict[str, float],
    candidate_types: dict[str, float],
    improved_rows: int,
    bootstrap_p_gain: float,
    provenance_ok: bool,
) -> dict[str, bool]:
    type_deltas = {
        kind: candidate_types[kind] - baseline_types[kind]
        for kind in baseline_types
    }
    checks = {
        "macro_strictly_above_frozen_baseline": candidate_score > BASELINE_SCORE,
        "at_least_two_types_nonnegative": sum(
            delta >= 0 for delta in type_deltas.values()
        ) >= 2,
        "no_type_regresses_more_than_0.01": min(type_deltas.values()) >= -0.01,
        "at_least_two_rows_improve": improved_rows >= 2,
        "bootstrap_p_gain_at_least_0.80": bootstrap_p_gain >= 0.80,
        "public_label_provenance_explicit": provenance_ok,
    }
    return checks | {"passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = ROOT / "artifacts/task1_all_released_ensemble_validation_v1"
    parser.add_argument(
        "--baseline", type=Path, default=base_dir / "baseline_train_only.jsonl",
    )
    parser.add_argument(
        "--candidate", type=Path,
        default=base_dir / "candidate_single_verified_external.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=base_dir)
    args = parser.parse_args()

    validation = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    gold = [parse_answer(row.answer) for row in validation.itertuples()]
    baseline = load_answers(args.baseline)
    candidate = load_answers(args.candidate)
    baseline_rows = np.array([
        score(target, prediction)[2]
        for target, prediction in zip(gold, baseline, strict=True)
    ])
    candidate_rows = np.array([
        score(target, prediction)[2]
        for target, prediction in zip(gold, candidate, strict=True)
    ])
    baseline_score = float(baseline_rows.mean())
    if abs(baseline_score - BASELINE_SCORE) > 1e-12:
        raise ValueError("frozen ensemble baseline has drifted")
    candidate_score = float(candidate_rows.mean())
    deltas = candidate_rows - baseline_rows
    baseline_types = entity_type_f1(gold, baseline)
    candidate_types = entity_type_f1(gold, candidate)
    rng = np.random.default_rng(260807)
    draws = deltas[rng.integers(0, 80, size=(10_000, 80))].mean(axis=1)
    p_gain = float((draws > 0).mean())
    member_metrics = json.loads((
        ROOT / "artifacts/task1_textual_train_plus_public_test_seed7_v1/metrics.json"
    ).read_text())
    provenance_ok = bool(
        member_metrics.get("train_splits") == ["train", "test"]
        and member_metrics.get("evaluation_split") == "validation"
        and member_metrics.get("test_split_used_for_training") is True
    )
    gate = ensemble_gate(
        candidate_score, baseline_types, candidate_types,
        int((deltas > 1e-12).sum()), p_gain, provenance_ok,
    )
    summary = {
        "evaluation": "one-for-one public-label ensemble member replacement",
        "current_or_hidden_test_used": False,
        "baseline_macro_entity_f1": baseline_score,
        "candidate_macro_entity_f1": candidate_score,
        "delta": candidate_score - baseline_score,
        "improved_rows": int((deltas > 1e-12).sum()),
        "harmed_rows": int((deltas < -1e-12).sum()),
        "tied_rows": int((abs(deltas) <= 1e-12).sum()),
        "baseline_type_f1": baseline_types,
        "candidate_type_f1": candidate_types,
        "type_deltas": {
            kind: candidate_types[kind] - baseline_types[kind]
            for kind in baseline_types
        },
        "row_bootstrap": {
            "resamples": 10_000,
            "p_gain": p_gain,
            "p_nonnegative": float((draws >= 0).mean()),
            "p05": float(np.quantile(draws, 0.05)),
            "median": float(np.quantile(draws, 0.5)),
            "p95": float(np.quantile(draws, 0.95)),
        },
        "ensemble_gate": gate,
        "decision": "confirm_external_ensemble_lane" if gate["passed"] else "demote",
        "external_member_metrics": member_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output_dir / "row_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "row_index", "baseline_f1", "candidate_f1", "delta",
        ])
        writer.writeheader()
        for index, (base, candidate_value, delta) in enumerate(
            zip(baseline_rows, candidate_rows, deltas, strict=True)
        ):
            writer.writerow({
                "row_index": index, "baseline_f1": float(base),
                "candidate_f1": float(candidate_value), "delta": float(delta),
            })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
