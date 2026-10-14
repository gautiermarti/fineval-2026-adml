#!/usr/bin/env python3
"""Extract GRFinNER rows while preserving their Task 1 submission indices."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_task1_lookup_safe import task_kind


INDEX_COLUMN = "submission_row_index"


def native_scalar(value):
    return value.item() if hasattr(value, "item") else value


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"query", "text"}.issubset(frame.columns):
        raise ValueError("Task 1 input must contain query and text columns")
    if INDEX_COLUMN in frame.columns:
        raise ValueError(f"input already contains reserved column {INDEX_COLUMN!r}")
    kinds = frame["query"].map(task_kind)
    if (kinds == "unknown").any():
        bad = kinds[kinds == "unknown"].index.tolist()[:10]
        raise ValueError(f"unrecognized Task 1 prompts at input positions {bad}")
    source_ids = (
        [native_scalar(value) for value in frame["row_index"].tolist()]
        if "row_index" in frame.columns
        else list(range(len(frame)))
    )
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Task 1 source row indices are not unique")
    subset = frame.loc[kinds == "textual"].copy()
    subset.insert(
        0, INDEX_COLUMN,
        [source_ids[position] for position, kind in enumerate(kinds) if kind == "textual"],
    )
    return subset.reset_index(drop=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    subset = prepare(pd.read_parquet(args.input))
    if subset.empty:
        raise SystemExit("Task 1 input contains no GRFinNER/textual rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(args.output, index=False)
    print(
        f"wrote textual_rows={len(subset)} output={args.output} "
        f"source_indices_preserved=true"
    )


if __name__ == "__main__":
    main()
