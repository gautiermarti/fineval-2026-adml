#!/usr/bin/env python3
"""Apply idempotent input-only Task 1 invariants to a cached prediction file."""

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

from scripts.predict_task1_lookup_safe import task_kind  # noqa: E402
from scripts.task1_numeric_baseline import parse_answer  # noqa: E402
from scripts.task1_postprocess import normalize_calendar_day_entities  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(entities: list[tuple[str, str]]) -> str:
    return "\n".join(f"{surface}, {label}" for surface, label in entities)


def normalize_records(
    source: list[dict[str, Any]], predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(source) != len(predictions):
        raise ValueError(
            f"row-count mismatch: input={len(source)} predictions={len(predictions)}"
        )
    output: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for position, (row, prediction) in enumerate(zip(source, predictions, strict=True)):
        expected = row.get("row_index", position)
        if prediction.get("row_index") != expected:
            raise ValueError(
                f"row {position}: alignment mismatch "
                f"{prediction.get('row_index')!r} != {expected!r}"
            )
        kind = task_kind(row.get("query", ""))
        if kind == "unknown":
            raise ValueError(f"row {position}: unrecognized Task 1 prompt")
        old = parse_answer(prediction.get("answer", ""))
        new = (
            normalize_calendar_day_entities(str(row.get("text", "")), old)
            if kind == "numeric" else old
        )
        output.append({"row_index": expected, "answer": render(new)})
        if old != new:
            changes.append({
                "row_index": expected,
                "task_kind": kind,
                "removed": [list(item) for item in old if item not in new],
                "added": [list(item) for item in new if item not in old],
                "old_answer": render(old),
                "new_answer": render(new),
            })
    return output, changes


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    source = pd.read_parquet(args.input).to_dict("records")
    predictions = [
        json.loads(line) for line in args.predictions.read_text().splitlines()
        if line.strip()
    ]
    output, changes = normalize_records(source, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "input": str(args.input),
        "input_sha256": digest(args.input),
        "predictions": str(args.predictions),
        "predictions_sha256": digest(args.predictions),
        "output": str(args.output),
        "output_sha256": digest(args.output),
        "rows": len(output),
        "changed_rows": len(changes),
        "changes": changes,
        "uses_answers_or_labels": False,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "rows": len(output),
        "changed_rows": len(changes),
        "output": str(args.output),
        "output_sha256": report["output_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()

