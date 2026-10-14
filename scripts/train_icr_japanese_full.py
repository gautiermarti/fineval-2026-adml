#!/usr/bin/env python3
"""Train a full-data Japanese JF-ICR classifier for one reproducible seed."""

from argparse import ArgumentParser
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_icr_japanese_cv import (  # noqa: E402
    DEFAULT_MODEL, ICRDataset, LABELS, collate,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        help="Optional immutable Hugging Face model revision/commit.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--truncation-profile",
        choices=("longest_first", "response_head_tail"),
        default="longest_first",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--bf16", action="store_true",
        help="Use CUDA bfloat16 autocast while retaining FP32 optimizer state.",
    )
    parser.add_argument(
        "--rinna-roberta-compat", action="store_true",
        help="Use the tokenizer/position convention required by rinna Japanese RoBERTa.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16 and device.type != "cuda":
        raise ValueError("--bf16 requires a CUDA device")
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=False if args.rinna_roberta_compat else None,
    )
    if args.rinna_roberta_compat:
        tokenizer.do_lower_case = True
    dataset = ICRDataset(
        frame, tokenizer, args.max_length, args.truncation_profile,
        args.rinna_roberta_compat,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda items: collate(
            items, tokenizer,
            zero_based_position_ids=args.rinna_roberta_compat,
        ),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01,
    )
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=args.bf16,
            ):
                loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} mean_loss={np.mean(losses):.6f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training.txt").write_text(
        f"base_model={args.model}\nrevision={args.revision or 'default'}\n"
        f"seed={args.seed}\nepochs={args.epochs}\n"
        f"bf16={args.bf16}\n"
        f"rows={len(frame)}\ntruncation_profile={args.truncation_profile}\n"
        f"rinna_roberta_compat={args.rinna_roberta_compat}\n"
    )


if __name__ == "__main__":
    main()
