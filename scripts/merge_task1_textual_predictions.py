#!/usr/bin/env python3
"""Safely overlay local GRFinNER predictions onto a full Task 1 submission."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_task1_lookup_safe import task_kind
from prepare_task1_textual_subset import INDEX_COLUMN


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def native_scalar(value):
    return value.item() if hasattr(value, "item") else value


def merge(
    source: pd.DataFrame,
    base: list[dict],
    textual_source: pd.DataFrame,
    textual: list[dict],
) -> list[dict]:
    if len(base) != len(source):
        raise ValueError(f"base row-count mismatch: {len(base)} != {len(source)}")
    source_ids = (
        [native_scalar(value) for value in source["row_index"].tolist()]
        if "row_index" in source.columns
        else list(range(len(source)))
    )
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Task 1 source row indices are not unique")
    for position, (expected, record) in enumerate(zip(source_ids, base)):
        if record.get("row_index") != expected or "answer" not in record:
            raise ValueError(f"base prediction misalignment at position {position}")

    if INDEX_COLUMN not in textual_source.columns:
        raise ValueError(
            f"textual input lacks {INDEX_COLUMN!r}; create it with "
            "prepare_task1_textual_subset.py"
        )
    textual_ids = [
        native_scalar(value) for value in textual_source[INDEX_COLUMN].tolist()
    ]
    if len(textual) != len(textual_ids):
        raise ValueError(
            f"textual row-count mismatch: {len(textual)} != {len(textual_ids)}"
        )
    if len(set(textual_ids)) != len(textual_ids):
        raise ValueError("textual submission row indices are not unique")
    source_id_set = set(source_ids)
    if not set(textual_ids).issubset(source_id_set):
        raise ValueError("textual subset contains indices absent from full Task 1 input")

    if "query" in source.columns:
        kinds = source["query"].map(task_kind).tolist()
        if "unknown" in kinds:
            raise ValueError("full Task 1 input contains unrecognized prompts")
        expected_textual_ids = [
            source_ids[position]
            for position, kind in enumerate(kinds)
            if kind == "textual"
        ]
        if textual_ids != expected_textual_ids:
            raise ValueError(
                "textual subset does not exactly match the full input's GRFinNER rows"
            )

    overlay = {}
    for local_index, (source_id, record) in enumerate(zip(textual_ids, textual)):
        if record.get("row_index") != local_index or "answer" not in record:
            raise ValueError(
                f"textual prediction misalignment at local position {local_index}"
            )
        overlay[source_id] = str(record["answer"])

    output = []
    for source_id, record in zip(source_ids, base):
        output.append({
            "row_index": source_id,
            "answer": overlay.get(source_id, str(record["answer"])),
        })
    return output


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--textual-input", type=Path, required=True)
    parser.add_argument("--textual-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = merge(
        pd.read_parquet(args.input), load_jsonl(args.base),
        pd.read_parquet(args.textual_input), load_jsonl(args.textual_predictions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in output
    ))
    print(
        f"wrote rows={len(output)} textual_overlays="
        f"{len(load_jsonl(args.textual_predictions))} output={args.output}"
    )


if __name__ == "__main__":
    main()
