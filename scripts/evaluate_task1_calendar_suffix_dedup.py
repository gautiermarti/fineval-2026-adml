#!/usr/bin/env python3
"""Audit the narrow GRFinNUM inflected-calendar-day normalization.

The public split is intentionally loaded only after all train/validation gates
and the externally reported unit-test gate pass.  Its labels were already
known when the hypothesis was written, so the resulting public score is a
transparent diagnostic rather than a blind estimate.
"""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.task1_numeric_baseline import parse_answer, score  # noqa: E402
from scripts.task1_postprocess import (  # noqa: E402
    INFLECTED_DATE_DAY,
    normalize_calendar_day_entities,
)


FROZEN_HASHES = {
    "train": "8195f2a8430fbe2f5ddd105ae022a2a8af80ef353c28664e2c97d4c91a8a2f24",
    "validation": "1dd14163fb81782e8e82fcb6437509956432cb4d2c8b06bcdf132fccbce19d84",
    "public_test": "fdfc51bad2bbe84f40a9061ddbb476971120038905c3cf03873839b35b183206",
    "validation_predictions": "ce0c474c36a8b7b1430beb9387fb7132fefd38452abf6272728eb54c15c01457",
    "public_test_predictions": "2685f95e3d25a9fed8c072d28b6c14337cd913446a00a781c45025207a939aa9",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def answer(entities: list[tuple[str, str]]) -> str:
    return "\n".join(f"{surface}, {label}" for surface, label in entities)


def load_predictions(path: Path, expected_rows: int) -> list[list[tuple[str, str]]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    indices = [int(record["row_index"]) for record in records]
    if indices != list(range(expected_rows)):
        raise ValueError(
            f"{path} row_index sequence is not exactly 0..{expected_rows - 1}"
        )
    return [parse_answer(record.get("answer", "")) for record in records]


def convention_audit(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    occurrences: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        gold = set(parse_answer(row["answer"]))
        for match in INFLECTED_DATE_DAY.finditer(row["text"]):
            surface = match.group("surface")
            day = match.group("day")
            occurrences.append({
                "split": split,
                "row_index": int(row_index),
                "matched_text": match.group(0),
                "surface": surface,
                "day": day,
                "core_temporal_in_gold": (day, "ΧΡΟΝΙΚΑ") in gold,
                "surface_temporal_in_gold": (surface, "ΧΡΟΝΙΚΑ") in gold,
            })
    return {
        "split": split,
        "matching_occurrences": len(occurrences),
        "matching_rows": len({item["row_index"] for item in occurrences}),
        "core_temporal_support": sum(
            item["core_temporal_in_gold"] for item in occurrences
        ),
        "surface_temporal_counterexamples": sum(
            item["surface_temporal_in_gold"] for item in occurrences
        ),
        "occurrences": occurrences,
    }


def transform_and_score(
    frame: pd.DataFrame,
    predictions: list[list[tuple[str, str]]],
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[tuple[str, str]]]]:
    transformed = [
        normalize_calendar_day_entities(row.text, predicted)
        for row, predicted in zip(frame.itertuples(), predictions, strict=True)
    ]
    old_scores = [
        score(parse_answer(row.answer), predicted)[2]
        for row, predicted in zip(frame.itertuples(), predictions, strict=True)
    ]
    new_scores = [
        score(parse_answer(row.answer), predicted)[2]
        for row, predicted in zip(frame.itertuples(), transformed, strict=True)
    ]
    changed: list[dict[str, Any]] = []
    for row_index, (row, old, new, old_f1, new_f1) in enumerate(zip(
        frame.itertuples(), predictions, transformed, old_scores, new_scores,
        strict=True,
    )):
        if old != new:
            changed.append({
                "split": split,
                "row_index": row_index,
                "text": row.text,
                "gold_answer": row.answer,
                "old_answer": answer(old),
                "new_answer": answer(new),
                "removed": [list(item) for item in old if item not in new],
                "added": [list(item) for item in new if item not in old],
                "old_f1": old_f1,
                "new_f1": new_f1,
                "delta_f1": new_f1 - old_f1,
            })
    old_macro = sum(old_scores) / len(old_scores)
    new_macro = sum(new_scores) / len(new_scores)
    return ({
        "rows": len(frame),
        "changed_rows": len(changed),
        "improved_rows": sum(item["delta_f1"] > 0 for item in changed),
        "harmed_rows": sum(item["delta_f1"] < 0 for item in changed),
        "unchanged_score_rows": sum(item["delta_f1"] == 0 for item in changed),
        "old_macro_entity_f1": old_macro,
        "new_macro_entity_f1": new_macro,
        "delta_macro_entity_f1": new_macro - old_macro,
    }, changed, transformed)


def write_predictions(
    path: Path, predictions: list[list[tuple[str, str]]],
) -> None:
    with path.open("w") as handle:
        for row_index, entities in enumerate(predictions):
            handle.write(json.dumps(
                {"row_index": row_index, "answer": answer(entities)},
                ensure_ascii=False,
            ) + "\n")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--train", type=Path,
        default=ROOT / "data/raw/subtask1/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--validation", type=Path,
        default=ROOT / "data/raw/subtask1/validation-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--public-test", type=Path,
        default=ROOT / "data/external/plutus_finner_numeric/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--validation-predictions", type=Path,
        default=ROOT / "artifacts/task1_ensemble/validation_submission.jsonl",
    )
    parser.add_argument(
        "--public-test-predictions", type=Path,
        default=ROOT / "artifacts/task1_numeric_test_full_refit.jsonl",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "docs/task1_calendar_suffix_dedup_audit_protocol.md",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_calendar_suffix_dedup_v1",
    )
    parser.add_argument(
        "--unit-tests-passed", action="store_true",
        help="Confirm the protocol's targeted unit tests passed in this run.",
    )
    args = parser.parse_args()

    pre_public_paths = {
        "train": args.train,
        "validation": args.validation,
        "validation_predictions": args.validation_predictions,
    }
    observed_hashes = {
        name: digest(path) for name, path in pre_public_paths.items()
    }
    hash_gate = all(
        observed_hashes[name] == FROZEN_HASHES[name]
        for name in pre_public_paths
    )

    # Phase one: released train/validation only.
    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    train_audit = convention_audit(train, "train")
    validation_audit = convention_audit(validation, "validation")
    validation_predictions = load_predictions(
        args.validation_predictions, len(validation)
    )
    validation_result, validation_changes, validation_transformed = (
        transform_and_score(
            validation, validation_predictions, "validation",
        )
    )

    convention_gate = all([
        train_audit["matching_occurrences"] > 0,
        validation_audit["matching_occurrences"] > 0,
        train_audit["core_temporal_support"]
            == train_audit["matching_occurrences"],
        validation_audit["core_temporal_support"]
            == validation_audit["matching_occurrences"],
        train_audit["surface_temporal_counterexamples"] == 0,
        validation_audit["surface_temporal_counterexamples"] == 0,
        train_audit["matching_occurrences"]
            + validation_audit["matching_occurrences"] >= 10,
    ])
    validation_gate = all([
        validation_result["changed_rows"] == 0,
        validation_result["old_macro_entity_f1"]
            == validation_result["new_macro_entity_f1"],
    ])
    pre_public_gates = {
        "frozen_input_hashes_match": hash_gate,
        "train_validation_annotation_convention": convention_gate,
        "validation_zero_changes_and_exact_score_equality": validation_gate,
        "targeted_unit_tests_reported_passed": args.unit_tests_passed,
    }
    if not all(pre_public_gates.values()):
        raise RuntimeError(
            "Pre-public promotion gates failed; public split was not loaded: "
            + json.dumps(pre_public_gates, sort_keys=True)
        )

    # Phase two: transparent diagnostic on the already-opened public split.
    public_paths = {
        "public_test": args.public_test,
        "public_test_predictions": args.public_test_predictions,
    }
    public_hashes = {name: digest(path) for name, path in public_paths.items()}
    if not all(
        public_hashes[name] == FROZEN_HASHES[name] for name in public_paths
    ):
        raise RuntimeError("Released-public frozen input hashes do not match")
    observed_hashes.update(public_hashes)
    public_test = pd.read_parquet(args.public_test)
    public_audit = convention_audit(public_test, "released_public_test")
    public_predictions = load_predictions(
        args.public_test_predictions, len(public_test)
    )
    public_result, public_changes, public_transformed = transform_and_score(
        public_test, public_predictions, "released_public_test",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_output = args.output_dir / "validation_predictions.jsonl"
    public_output = args.output_dir / "public_test_predictions.jsonl"
    changes_output = args.output_dir / "changed_rows.jsonl"
    write_predictions(validation_output, validation_transformed)
    write_predictions(public_output, public_transformed)
    with changes_output.open("w") as handle:
        for item in validation_changes + public_changes:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "experiment": "task1_calendar_suffix_dedup_v1",
        "status": "promote" if all(pre_public_gates.values()) else "reject",
        "disclosure": {
            "post_public_test_hypothesis": True,
            "released_public_test_answers_used_for_diagnostic": True,
            "hidden_competition_test_used": False,
            "answer_lookup_or_reference_access_at_inference": False,
        },
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": digest(args.protocol),
        },
        "frozen_hashes_expected": FROZEN_HASHES,
        "frozen_hashes_observed": observed_hashes,
        "pre_public_gates": pre_public_gates,
        "annotation_convention": {
            "train": train_audit,
            "validation": validation_audit,
            "released_public_test": public_audit,
        },
        "validation": validation_result,
        "released_public_test": public_result,
        "outputs": {
            "validation_predictions": {
                "path": str(validation_output.relative_to(ROOT)),
                "sha256": digest(validation_output),
                "bytes": validation_output.stat().st_size,
                "rows": len(validation_transformed),
            },
            "released_public_test_predictions": {
                "path": str(public_output.relative_to(ROOT)),
                "sha256": digest(public_output),
                "bytes": public_output.stat().st_size,
                "rows": len(public_transformed),
            },
            "changed_rows": {
                "path": str(changes_output.relative_to(ROOT)),
                "sha256": digest(changes_output),
                "bytes": changes_output.stat().st_size,
                "rows": len(validation_changes) + len(public_changes),
            },
        },
        "decision": (
            "Promote the input-only normalization for future GRFinNUM inference; "
            "treat the released-public score only as a non-blind diagnostic."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "status": summary["status"],
        "train_occurrences": train_audit["matching_occurrences"],
        "validation_occurrences": validation_audit["matching_occurrences"],
        "validation": validation_result,
        "released_public_test": public_result,
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
