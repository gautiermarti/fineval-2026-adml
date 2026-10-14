#!/usr/bin/env python3
"""Audit exact and normalized query overlap with the public JF-ICR release."""

from argparse import ArgumentParser
from pathlib import Path
import json
import re
import unicodedata
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    return re.sub(r"\s+", " ", value)


def semantic_key(value: str) -> tuple[str, str]:
    parsed = parse_icr_query(value)
    return normalize(parsed.question), normalize(parsed.response)


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame(
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    hidden = load(args.input)
    exact = dict(zip(public["query"], public["answer"].astype(str)))
    normalized = dict(zip(public["query"].map(normalize), public["answer"].astype(str)))
    semantic = dict(zip(public["query"].map(semantic_key), public["answer"].astype(str)))
    rows = []
    for position, row in hidden.iterrows():
        query = row["query"]
        exact_label = exact.get(query)
        normalized_label = normalized.get(normalize(query))
        semantic_label = semantic.get(semantic_key(query))
        rows.append({
            "row_index": position,
            "id": row.get("id", position),
            "exact_query_match": exact_label is not None,
            "normalized_query_match": normalized_label is not None,
            "semantic_packet_match": semantic_label is not None,
            "public_label": exact_label or normalized_label or semantic_label or "",
        })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"rows={len(result)} exact={result.exact_query_match.sum()} "
        f"normalized={result.normalized_query_match.sum()} "
        f"semantic={result.semantic_packet_match.sum()} wrote={args.output}"
    )


if __name__ == "__main__":
    main()
