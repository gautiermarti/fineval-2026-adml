#!/usr/bin/env python3
"""Refit the selected overlapping GRFinNER model on all released rows."""

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_task1_textual_multilabel import (  # noqa: E402
    MultiHeadNER, NERDataset, collate, make_examples,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="spyrosbriakos/greek_legal_bert_v2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_textual_full/multilabel_legal_seed7",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = pd.concat([
        pd.read_parquet(
            ROOT / f"data/external/plutus_finner_text/{split}-00000-of-00001.parquet"
        )
        for split in ("train", "validation")
    ], ignore_index=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    examples, dropped = make_examples(frame, tokenizer, drop_fully_unaligned=True)
    print(f"full_rows={len(examples)} dropped={dropped}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadNER(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader = DataLoader(
        NERDataset(examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            _, loss = model(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach())
        print(f"epoch={epoch} mean_train_loss={total_loss / len(loader):.12f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "released_rows_used": len(examples),
        "dropped_fully_unaligned_rows": dropped,
        "validation_used_for_training": True,
        "public_test_used_for_training": False,
        "validation_score_claimed": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
