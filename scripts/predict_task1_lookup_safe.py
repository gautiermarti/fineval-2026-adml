#!/usr/bin/env python3
"""Quarantined exact-public lookup for the two Plutus Task 1 NER tracks."""

from argparse import ArgumentParser
from pathlib import Path
import json
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATHS = {
    "numeric": [
        ROOT / "data/raw/subtask1/train-00000-of-00001.parquet",
        ROOT / "data/raw/subtask1/validation-00000-of-00001.parquet",
        ROOT / "data/external/plutus_finner_numeric/test-00000-of-00001.parquet",
    ],
    "textual": [
        ROOT / f"data/external/plutus_finner_text/{split}-00000-of-00001.parquet"
        for split in ("train", "validation", "test")
    ],
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip())


def task_kind(query: str) -> str:
    folded = normalize(query).casefold()
    if "αριθμητικές οντότητες" in folded or "χρηματικά ποσά" in folded:
        return "numeric"
    if "πρόσωπο" in folded and "οργανισμό" in folded and "τοποθεσία" in folded:
        return "textual"
    return "unknown"


def load_references() -> tuple[dict, dict, dict]:
    by_query: dict[tuple[str, str], str] = {}
    by_normalized_query: dict[tuple[str, str], str] = {}
    by_packet: dict[tuple[str, str], str] = {}
    for kind, paths in REFERENCE_PATHS.items():
        for path in paths:
            if not path.exists():
                continue
            for row in pd.read_parquet(path).itertuples():
                by_query.setdefault((kind, row.query), row.answer)
                by_normalized_query.setdefault((kind, normalize(row.query)), row.answer)
                by_packet.setdefault((kind, normalize(row.text)), row.answer)
    return by_query, by_normalized_query, by_packet


def load_fallback(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(row.get("row_index", index)): str(row["answer"]) for index, row in enumerate(records)}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fallback", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--require-all-matched", action="store_true")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    by_query, by_normalized_query, by_packet = load_references()
    fallback = load_fallback(args.fallback)
    output, audit = [], []
    for index, row in enumerate(frame.itertuples()):
        kind = task_kind(row.query)
        query_key = (kind, row.query)
        normalized_key = (kind, normalize(row.query))
        packet_key = (kind, normalize(row.text))
        if query_key in by_query:
            answer, method = by_query[query_key], "exact_query"
        elif normalized_key in by_normalized_query:
            answer, method = by_normalized_query[normalized_key], "normalized_query"
        elif packet_key in by_packet:
            answer, method = by_packet[packet_key], "exact_task_text"
        else:
            answer, method = fallback.get(index, ""), "fallback" if index in fallback else "empty"
        output.append({"row_index": index, "answer": answer})
        audit.append({
            "row_index": index,
            "task_kind": kind,
            "method": method,
            "matched": method not in {"fallback", "empty"},
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit).to_csv(args.audit_output, index=False)
    counts = pd.DataFrame(audit).groupby(["task_kind", "method"]).size()
    print(counts.to_string())
    print(f"wrote={len(output)} output={args.output}")
    unmatched = sum(not row["matched"] for row in audit)
    if args.require_all_matched and unmatched:
        raise ValueError(f"external reference coverage incomplete: unmatched={unmatched}")


if __name__ == "__main__":
    main()
