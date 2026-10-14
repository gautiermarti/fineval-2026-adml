#!/usr/bin/env python3
"""Refit a selected Task 1 encoder on all usable released labels."""

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_task1_transformer import (  # noqa: E402
    DS, LABELS, L2I, align_annotations, collate, make_examples, parse_answer,
)


def completely_alignable(row: pd.Series) -> bool:
    spans = parse_answer(row.answer)
    return len(align_annotations(str(row.text), spans)) == len(spans)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = pd.read_parquet(ROOT / "data/raw/subtask1/train-00000-of-00001.parquet")
    validation = pd.read_parquet(
        ROOT / "data/raw/subtask1/validation-00000-of-00001.parquet"
    )
    frame = pd.concat([train, validation], ignore_index=True)
    usable = frame.apply(completely_alignable, axis=1)
    excluded = frame.index[~usable].tolist()
    frame = frame[usable].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    examples = make_examples(frame, tokenizer)
    loader = DataLoader(
        DS(examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=L2I,
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01,
    )
    epoch_losses = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))
        print(f"epoch={epoch + 1} mean_loss={epoch_losses[-1]:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "base_model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "released_rows": len(train) + len(validation),
        "usable_rows": len(frame),
        "excluded_combined_row_indices": excluded,
        "epoch_mean_losses": epoch_losses,
        "selection_evidence": (
            "four full-data epochs preserve the update count of the predeclared "
            "five-epoch 320-row development protocol"
        ),
    }
    (args.output_dir / "training.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
