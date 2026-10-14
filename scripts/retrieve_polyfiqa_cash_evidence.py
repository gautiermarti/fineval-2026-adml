#!/usr/bin/env python3
"""CPU-only multilingual evidence retrieval for cash-flow questions."""

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

from fineval.finance import extract_cash_flows  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from retrieve_polyfiqa_revenue_evidence import LANGUAGES, MODEL, numeric_tokens, sentences  # noqa: E402


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    frame = frame[frame["question"].str.startswith("Are there any significant")]
    model = SentenceTransformer(MODEL, device="cpu")
    records = []
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        result = extract_cash_flows(context.financial_statements)
        values = result.operating + result.investing + result.financing
        query = (
            f"cash flow operating investing financing acquisitions capital expenditures "
            f"debt dividends repurchases {' '.join(f'{abs(value):,.0f}' for value in values)} "
            f"{result.unit} {row['question']}"
        )
        candidates = [
            {"language": language, "text": sentence}
            for language in LANGUAGES
            for sentence in sentences(getattr(context, f"{language}_news"))
        ]
        query_embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        embeddings = model.encode(
            [candidate["text"] for candidate in candidates], batch_size=32,
            normalize_embeddings=True, show_progress_bar=False,
        )
        expected = numeric_tokens(query)
        for candidate, semantic_score in zip(candidates, embeddings @ query_embedding, strict=True):
            overlap = len(expected & numeric_tokens(candidate["text"]))
            candidate["semantic_score"] = float(semantic_score)
            candidate["numeric_overlap"] = overlap
            candidate["hybrid_score"] = float(semantic_score + min(overlap, 3) * 0.10)
        candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
        records.append({"task_id": row["task_id"], "query": query, "candidates": candidates[:15]})
        print(row["task_id"], candidates[0]["language"], round(candidates[0]["hybrid_score"], 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
