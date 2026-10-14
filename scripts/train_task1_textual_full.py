#!/usr/bin/env python3
"""Refit a frozen GRFinNER architecture on all released train/validation rows."""

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_task1_textual import (  # noqa: E402
    LABELS, LABEL_TO_ID, TextualDataset, collate, load_release_split,
    make_examples,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--training-splits", nargs="+", choices=("train", "validation", "test"),
        default=["train", "validation"],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(set(args.training_splits)) != len(args.training_splits):
        raise ValueError("--training-splits must not contain duplicates")
    loaded = [load_release_split(split) for split in args.training_splits]
    frame = pd.concat([item[0] for item in loaded], ignore_index=True)
    provenance = [
        {"split": split, "path": str(path.relative_to(ROOT)), "sha256": digest,
         "rows": len(split_frame)}
        for split, (split_frame, path, digest) in zip(
            args.training_splits, loaded, strict=True
        )
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    examples, dropped = make_examples(frame, tokenizer, drop_fully_unaligned=True)
    print(f"full_rows={len(examples)} dropped={dropped}", flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID, ignore_mismatched_sizes=True,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device_name = (
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader = DataLoader(
        TextualDataset(examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach())
        print(f"epoch={epoch} mean_train_loss={total_loss / len(loader):.12f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "released_rows_used": len(examples),
        "dropped_fully_unaligned_rows": dropped,
        "training_splits": args.training_splits,
        "training_provenance": provenance,
        "validation_used_for_training": "validation" in args.training_splits,
        "public_test_used_for_training": "test" in args.training_splits,
        "validation_score_claimed": False,
        "device": device_name,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
