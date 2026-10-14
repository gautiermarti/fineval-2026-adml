#!/usr/bin/env python3
"""Generate low-threshold raw GLiNER proposals from a persisted model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu for intentional CPU inference")
    from gliner import GLiNER

    frame = pd.read_parquet(args.input, columns=["text"])
    model = GLiNER.from_pretrained(args.model, map_location="cpu")
    if not args.cpu:
        model = model.to("cuda")
    raw = model.batch_predict_entities(
        frame.text.astype(str).tolist(), ["person", "organization", "location"],
        flat_ner=False, multi_label=False, threshold=args.threshold,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps({"row_index": index, "entities": entities}, ensure_ascii=False) + "\n"
        for index, entities in enumerate(raw)
    ))
    print(json.dumps({
        "rows": len(raw), "entities": sum(len(row) for row in raw),
        "model": args.model, "threshold": args.threshold,
        "device": "cpu" if args.cpu else "cuda", "answer_column_accessed": False,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
