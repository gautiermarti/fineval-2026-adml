#!/usr/bin/env python3
"""Rank Task 2 event candidates with label-free Japanese semantic retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def encode(
    texts: list[str], tokenizer, model, *, batch_size: int, max_length: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(texts), batch_size):
        features = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**features).last_hidden_state
            mask = features["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            batches.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.concatenate(batches, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "artifacts/icr_zunda_events_v1/events.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_zunda_semantic_events_v1",
    )
    parser.add_argument("--model", default="cl-nagoya/ruri-v3-310m")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    # The GPU may be serving an unrelated workload.  This finite 2,224-text
    # extraction is deliberately CPU-only and never contends for GPU memory.
    torch.set_num_threads(args.threads)
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    questions = ["検索クエリ: " + row["question"] for row in records]
    candidates = [
        candidate for row in records for candidate in row["candidates"]
    ]
    documents = ["検索文書: " + candidate["text"] for candidate in candidates]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).eval().cpu()
    question_embeddings = encode(
        questions, tokenizer, model, batch_size=args.batch_size,
        max_length=args.max_length,
    )
    document_embeddings = encode(
        documents, tokenizer, model, batch_size=args.batch_size,
        max_length=args.max_length,
    )

    position = 0
    similarities = []
    for row_index, row in enumerate(records):
        count = len(row["candidates"])
        scores = document_embeddings[position:position + count] @ question_embeddings[row_index]
        for candidate, score in zip(row["candidates"], scores, strict=True):
            candidate["ruri_similarity"] = float(score)
            similarities.append(float(score))
        position += count
    if position != len(candidates):
        raise AssertionError("candidate embedding alignment failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "events.jsonl").write_text("".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    ))
    manifest = {
        "rows": len(records), "candidates": len(candidates),
        "model": args.model, "prefix_scheme": "検索クエリ/検索文書",
        "pooling": "attention-mask mean then L2 normalization",
        "device": "cpu", "max_length": args.max_length,
        "similarity_min": min(similarities),
        "similarity_mean": float(np.mean(similarities)),
        "similarity_max": max(similarities),
        "jf_icr_labels_read": False,
        "current_or_hidden_test_used": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
