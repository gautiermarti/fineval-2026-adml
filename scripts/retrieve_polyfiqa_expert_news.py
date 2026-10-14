#!/usr/bin/env python3
"""Build CPU-only multilingual news packets for PolyFiQA-Expert generation."""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path
import json
import re
import sys

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_polyfiqa_query  # noqa: E402

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LANGUAGES = ("english", "chinese", "japanese", "spanish", "greek")
QUERIES = {
    "revenue_focus": "top revenue business segments products growth drivers sales demand",
    "capital_allocation": "capital allocation investments share repurchases dividends acquisitions debt liquidity outlook",
    "margin": "strategy maintain profit margins cost control productivity efficiency high-margin products services",
    "capex": "capital expenditures strategic significance infrastructure capacity technology expansion investment",
}


def sentences(text: str) -> list[str]:
    return [
        part.strip().strip('"')
        for part in re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", text)
        if 30 <= len(part.strip()) <= 700
    ]


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/polyfiqa_expert/test-00000-of-00001.parquet")
    frame = frame.drop_duplicates("task_id")
    model = SentenceTransformer(MODEL, device="cpu")
    records = []
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        candidates = [
            {"language": language, "text": sentence}
            for language in LANGUAGES
            for sentence in sentences(getattr(context, f"{language}_news"))
        ]
        embeddings = model.encode(
            [item["text"] for item in candidates], batch_size=32,
            normalize_embeddings=True, show_progress_bar=False,
        )
        company = row["task_id"].split("_")[0]
        family_candidates = {}
        for family, query in QUERIES.items():
            query_embedding = model.encode(
                [f"{company} {query}"], normalize_embeddings=True, show_progress_bar=False,
            )[0]
            scored = []
            for item, score in zip(candidates, embeddings @ query_embedding, strict=True):
                scored.append({**item, "score": float(score)})
            scored.sort(key=lambda item: item["score"], reverse=True)
            family_candidates[family] = scored[:10]
        records.append({"task_id": row["task_id"], "families": family_candidates})
        print(row["task_id"], " ".join(
            f"{family}={items[0]['score']:.3f}/{items[0]['language']}"
            for family, items in family_candidates.items()
        ))
    output = ROOT / "artifacts/polyfiqa_expert/news_candidates.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
