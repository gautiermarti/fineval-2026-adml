#!/usr/bin/env python3
"""Convert validated ADML JSONL predictions to official FinNLP CSV uploads."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK1_LABELS = {
    "ΠΡΟΣΩΠΟ", "ΟΡΓΑΝΙΣΜΟΣ", "ΤΟΠΟΘΕΣΙΑ",
    "ΧΡΗΜΑΤΑ", "ΠΟΣΟΣΤΑ", "ΧΡΟΝΙΚΑ", "ΠΟΣΟΤΗΤΕΣ", "ΑΛΛΑ",
}
TASK2_LABELS = {"+2", "+1", "0", "-1", "-2"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def validate_task1_prediction(value: str) -> str:
    if not value.strip() or value.strip() == "None":
        return "None"
    for line in value.splitlines():
        if ", " not in line:
            raise ValueError(f"invalid Task 1 entity line: {line!r}")
        entity, label = line.rsplit(", ", 1)
        if not entity or label not in TASK1_LABELS:
            raise ValueError(f"invalid Task 1 entity line: {line!r}")
    return value


def build_rows(task: int, input_path: Path, predictions_path: Path) -> list[dict[str, str]]:
    frame = pd.read_parquet(input_path)
    predictions = read_jsonl(predictions_path)
    if len(frame) != len(predictions):
        raise ValueError(
            f"Task {task} input/prediction row mismatch; "
            f"input={len(frame)} predictions={len(predictions)}"
        )
    if "id" not in frame.columns or not frame["id"].is_unique:
        raise ValueError(f"Task {task} input requires unique id values")
    if "answer" in frame.columns:
        raise ValueError(f"Task {task} canonical test input unexpectedly contains answers")

    output = []
    for index, (source, prediction) in enumerate(zip(frame.to_dict("records"), predictions)):
        value = str(prediction.get("answer", ""))
        if task == 1:
            if prediction.get("row_index") != index:
                raise ValueError(f"Task 1 row_index mismatch at row {index}")
            value = validate_task1_prediction(value)
        elif task == 2:
            if str(prediction.get("id")) != str(source["id"]):
                raise ValueError(f"Task 2 id mismatch at row {index}")
            if value not in TASK2_LABELS:
                raise ValueError(f"invalid Task 2 label at row {index}: {value!r}")
        else:
            if prediction.get("task_id") != source.get("task_id"):
                raise ValueError(f"Task 3 task_id mismatch at row {index}")
            if not value.strip():
                raise ValueError(f"empty Task 3 answer at row {index}")
        output.append({"id": str(source["id"]), "prediction": value})
    return output


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_rows(args.task, args.input, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prediction"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "task": args.task,
        "rows": len(rows),
        "columns": ["id", "prediction"],
        "output": str(args.output),
        "answer_column_accessed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
