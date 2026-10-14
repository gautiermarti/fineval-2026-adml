#!/usr/bin/env python3
"""Fail-fast schema and alignment checks for final FinEval JSONL files."""

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd

from predict_task1_lookup_safe import task_kind
from task1_numeric_baseline import parse_answer
from task1_postprocess import INFLECTED_DATE_DAY


def load(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--allow-extra-fields", action="store_true",
        help="accept diagnostic rather than final-minimal records",
    )
    args = parser.parse_args()
    source, predictions = load(args.input), load(args.predictions)
    if len(source) != len(predictions):
        raise SystemExit(f"row-count mismatch: input={len(source)} predictions={len(predictions)}")

    key = "row_index" if args.task == 1 else ("id" if args.task == 2 else "task_id")
    required = {key, "answer"}
    allowed_labels = {"-2", "-1", "0", "+1", "+2"}
    allowed_entity_types = {
        "ΧΡΗΜΑΤΑ", "ΠΟΣΟΣΤΑ", "ΧΡΟΝΙΚΑ", "ΠΟΣΟΤΗΤΕΣ", "ΑΛΛΑ",
        "ΠΡΟΣΩΠΟ", "ΟΡΓΑΝΙΣΜΟΣ", "ΤΟΠΟΘΕΣΙΑ",
    }
    numeric_entity_types = {
        "ΧΡΗΜΑΤΑ", "ΠΟΣΟΣΤΑ", "ΧΡΟΝΙΚΑ", "ΠΟΣΟΤΗΤΕΣ", "ΑΛΛΑ",
    }
    textual_entity_types = {"ΠΡΟΣΩΠΟ", "ΟΡΓΑΝΙΣΜΟΣ", "ΤΟΠΟΘΕΣΙΑ"}
    seen = set()
    for position, (row, prediction) in enumerate(zip(source, predictions)):
        if not required.issubset(prediction):
            raise SystemExit(f"row {position}: missing {required - prediction.keys()}")
        if not args.allow_extra_fields and set(prediction) != required:
            raise SystemExit(f"row {position}: unexpected fields {set(prediction) - required}")
        expected = row.get(key, position)
        if prediction[key] != expected:
            raise SystemExit(
                f"row {position}: alignment mismatch {prediction[key]!r} != {expected!r}"
            )
        # PolyFiQA task_id identifies the source financial packet and is
        # intentionally reused across multiple questions. JF-ICR ids remain
        # row-unique.
        if args.task in {1, 2} and prediction[key] in seen:
            raise SystemExit(f"row {position}: duplicate {key} {prediction[key]!r}")
        seen.add(prediction[key])
        answer = prediction["answer"]
        if not isinstance(answer, str) or (args.task != 1 and not answer.strip()):
            raise SystemExit(f"row {position}: empty/non-string answer")
        if args.task == 1 and answer.strip():
            parsed = parse_answer(answer)
            kind = task_kind(row.get("query", ""))
            track_types = (
                numeric_entity_types if kind == "numeric"
                else textual_entity_types if kind == "textual"
                else allowed_entity_types
            )
            for line in answer.splitlines():
                if "," not in line:
                    raise SystemExit(f"row {position}: malformed Task 1 line {line!r}")
                entity, entity_type = line.rsplit(",", 1)
                if not entity.strip() or entity_type.strip() not in allowed_entity_types:
                    raise SystemExit(f"row {position}: malformed Task 1 line {line!r}")
                if entity_type.strip() not in track_types:
                    raise SystemExit(
                        f"row {position}: {entity_type.strip()} is invalid for "
                        f"Task 1 {kind} track"
                    )
            if kind == "numeric":
                predicted = set(parsed)
                for match in INFLECTED_DATE_DAY.finditer(str(row.get("text", ""))):
                    surface = (match.group("surface"), "ΧΡΟΝΙΚΑ")
                    if surface in predicted:
                        raise SystemExit(
                            f"row {position}: redundant inflected calendar-day "
                            f"surface {surface[0]!r}; emit numeric core "
                            f"{match.group('day')!r}"
                        )
        if args.task == 2 and answer not in allowed_labels:
            raise SystemExit(f"row {position}: invalid JF-ICR label {answer!r}")
        if args.task == 3 and len(answer.split()) > 100:
            print(f"warning: row {position} exceeds 100 whitespace words")
    uniqueness = len(seen) == len(predictions)
    print(f"valid task={args.task} rows={len(predictions)} unique_{key}={str(uniqueness).lower()} aligned=true")


if __name__ == "__main__":
    main()
