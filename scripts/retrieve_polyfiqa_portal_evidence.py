#!/usr/bin/env python3
"""CPU multilingual evidence retrieval for unmatched prior hidden inputs."""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from retrieve_polyfiqa_revenue_evidence import LANGUAGES, MODEL, sentences  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    known_queries = set(pd.read_parquet(
        ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet"
    )["query"])
    rows = [
        row for row in rows
        if row.get("tier") == "easy" and row["question"].startswith("What trends")
        and row["query"] not in known_queries
    ]
    model = SentenceTransformer(MODEL, device="cpu")
    records = []
    for row in rows:
        context = parse_polyfiqa_query(row["query"])
        candidates = [
            {"language": language, "text": sentence}
            for language in LANGUAGES
            for sentence in sentences(getattr(context, f"{language}_news"))[:20]
        ]
        query = (
            "company revenue sales growth increased decreased trend year-over-year "
            + row["question"]
        )
        query_vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        vectors = model.encode(
            [candidate["text"] for candidate in candidates], batch_size=32,
            normalize_embeddings=True, show_progress_bar=False,
        )
        for candidate, score in zip(candidates, vectors @ query_vector, strict=True):
            candidate["semantic_score"] = float(score)
        candidates.sort(key=lambda item: item["semantic_score"], reverse=True)
        records.append({"task_id": row["task_id"], "candidates": candidates[:15]})
        print(row["task_id"], candidates[0]["language"], round(candidates[0]["semantic_score"], 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
