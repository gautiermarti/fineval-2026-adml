#!/usr/bin/env python3
"""Predict GRFinNER entities with one or more compatible token classifiers."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_task1_textual import (  # noqa: E402
    LABELS, decode, encode_release_entity, make_examples,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+",
        default=[str(ROOT / "artifacts/task1_textual/legal_base_seed7")],
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_logits = []
    reference_examples = None
    reference_tokenizer = None
    for model_path in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        examples, _ = make_examples(frame.assign(answer=""), tokenizer)
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device).eval()
        logits = []
        with torch.no_grad():
            for encoded, _, _, _ in examples:
                input_ids = torch.tensor(encoded["input_ids"])[None].to(device)
                attention = torch.tensor(encoded["attention_mask"])[None].to(device)
                logits.append(model(input_ids=input_ids, attention_mask=attention).logits[0].cpu())
        if reference_examples is not None:
            offsets = [row[0].offset_mapping for row in examples]
            reference_offsets = [row[0].offset_mapping for row in reference_examples]
            if offsets != reference_offsets:
                raise ValueError("textual ensemble tokenizers produce incompatible offsets")
        else:
            reference_examples = examples
            reference_tokenizer = tokenizer
        all_logits.append(logits)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    averaged = [
        sum((model_rows[index] for model_rows in all_logits), torch.zeros_like(all_logits[0][index]))
        / len(all_logits)
        for index in range(len(frame))
    ]
    predictions = decode(averaged, reference_examples)
    predictions = [
        [(encode_release_entity(entity, kind), kind) for entity, kind in entities]
        for entities in predictions
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for index, entities in enumerate(predictions):
            answer = "\n".join(f"{entity}, {kind}" for entity, kind in entities)
            handle.write(json.dumps({"row_index": index, "answer": answer}, ensure_ascii=False) + "\n")
    print(f"wrote={len(predictions)} models={len(args.models)} labels={LABELS} output={args.output}")


if __name__ == "__main__":
    main()
