#!/usr/bin/env python3
"""CPU-only multilingual evidence retrieval for cash-flow questions."""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.finance import extract_cash_flows  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LANGUAGES = ["english", "chinese", "japanese", "spanish", "greek"]


def sentences(text: str) -> list[str]:
    return [
        piece.strip().strip('"')
        for piece in re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", text)
        if 25 <= len(piece.strip()) <= 700
    ]


def numeric_tokens(text: str) -> set[str]:
    return {token.replace(",", "") for token in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask3/public-00000-of-00001.parquet")
    frame = frame[frame["question"].str.startswith("Are there any significant")]
    model = SentenceTransformer(MODEL, device="cpu")
    records = []
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        result = extract_cash_flows(context.financial_statements)
        values = [result.operating[0], result.investing[0], result.financing[0]]
        query = (
            f"{row['task_id'].split('_')[0]} operating cash flow investing cash flow financing cash flow "
            + " ".join(f"{abs(value):,.0f}" for value in values)
            + " significant change irregularity"
        )
        candidates = [
            {"language": language, "text": sentence}
            for language in LANGUAGES
            for sentence in sentences(getattr(context, f"{language}_news"))
        ]
        query_embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        candidate_embeddings = model.encode(
            [candidate["text"] for candidate in candidates], batch_size=32,
            normalize_embeddings=True, show_progress_bar=False,
        )
        expected = numeric_tokens(query)
        for candidate, score in zip(candidates, candidate_embeddings @ query_embedding, strict=True):
            overlap = len(expected & numeric_tokens(candidate["text"]))
            candidate.update(
                semantic_score=float(score), numeric_overlap=overlap,
                hybrid_score=float(score + min(overlap, 3) * 0.08),
            )
        candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
        records.append({"task_id": row["task_id"], "candidates": candidates[:15]})
        print(row["task_id"], round(candidates[0]["hybrid_score"], 3), candidates[0]["language"], candidates[0]["text"][:100])
    output = ROOT / "artifacts/polyfiqa_retrieval/cashflow_candidates.jsonl"
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

