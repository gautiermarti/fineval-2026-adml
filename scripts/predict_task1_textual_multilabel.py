#!/usr/bin/env python3
"""Predict overlapping GRFinNER entities with a saved multi-head model."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_task1_textual_multilabel import (  # noqa: E402
    MultiHeadNER, NERDataset, collate, decode, make_examples,
)
from train_task1_textual import encode_release_entity  # noqa: E402


def predict_frame(frame, model_dir: Path, device, batch_size: int = 8):
    metadata = json.loads((model_dir / "metrics.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    examples, _ = make_examples(frame.assign(answer=""), tokenizer)
    model = MultiHeadNER(metadata["model"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.to(device).eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            rows = [NERDataset(examples)[index] for index in range(
                start, min(start + batch_size, len(examples))
            )]
            batch = collate(rows)
            values, _ = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch.get("token_type_ids", None).to(device)
                if "token_type_ids" in batch else None,
            )
            logits.extend(values.cpu())
    return decode(logits, examples)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "artifacts/task1_textual/multilabel_legal_seed7",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = predict_frame(frame, args.model_dir, device, args.batch_size)
    predictions = [
        [(encode_release_entity(entity, kind), kind) for entity, kind in entities]
        for entities in predictions
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for index, entities in enumerate(predictions):
            answer = "\n".join(f"{entity}, {kind}" for entity, kind in entities)
            handle.write(json.dumps({"row_index": index, "answer": answer}, ensure_ascii=False) + "\n")
    print(f"wrote={len(predictions)} output={args.output}")


if __name__ == "__main__":
    main()
