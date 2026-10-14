#!/usr/bin/env python3
"""CPU-only multilingual evidence retrieval for revenue questions."""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path
from argparse import ArgumentParser
import json
import re
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.finance import extract_revenue  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402


MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LANGUAGES = ["english", "chinese", "japanese", "spanish", "greek"]


def sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", text)
    cleaned = []
    for piece in pieces:
        piece = piece.strip().strip('"')
        if 25 <= len(piece) <= 700:
            cleaned.append(piece)
    return cleaned


def numeric_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"\d[\d,.]*", text):
        token = token.rstrip(".,")
        tokens.add(token.replace(",", ""))
        # Spanish and Greek commonly use decimal commas. Keep both the
        # thousands interpretation and a decimal variant for short suffixes.
        if re.fullmatch(r"\d+,\d{1,2}", token):
            tokens.add(token.replace(",", "."))
    return tokens


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/subtask3/public-00000-of-00001.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/polyfiqa_retrieval/revenue_candidates.jsonl")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    frame = frame[frame["question"].str.startswith("What trends")]
    model = SentenceTransformer(MODEL, device="cpu")
    records = []
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        result = extract_revenue(context.financial_statements)
        company = row["task_id"].split("_")[0]
        query = (
            f"{company} company revenue sales increased decreased growth "
            f"{result.current:,.0f} {result.previous:,.0f} {result.unit} {row['question']}"
        )
        candidates = []
        for language in LANGUAGES:
            for sentence in sentences(getattr(context, f"{language}_news")):
                candidates.append({"language": language, "text": sentence})
        query_embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        candidate_embeddings = model.encode(
            [candidate["text"] for candidate in candidates],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        semantic = candidate_embeddings @ query_embedding
        expected_numbers = numeric_tokens(query)
        expected_numbers |= numeric_tokens(f"{abs(result.change_percent):.1f}")
        for candidate, semantic_score in zip(candidates, semantic, strict=True):
            overlap = len(expected_numbers & numeric_tokens(candidate["text"]))
            candidate["semantic_score"] = float(semantic_score)
            candidate["numeric_overlap"] = overlap
            candidate["hybrid_score"] = float(semantic_score + min(overlap, 3) * 0.08)
        candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
        records.append({"task_id": row["task_id"], "query": query, "candidates": candidates[:15]})
        print(row["task_id"], round(candidates[0]["hybrid_score"], 3), candidates[0]["language"], candidates[0]["text"][:100])
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
