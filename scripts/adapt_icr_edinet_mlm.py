#!/usr/bin/env python3
"""Domain-adapt Japanese BERT on unrelated-company EDINET filing blocks."""

from argparse import ArgumentParser
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForPreTraining, AutoTokenizer, DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "cl-tohoku/bert-base-japanese-v3"


class BlockDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        # PyArrow list columns round-trip as numpy integers, but the
        # Transformers tokenizer pad path accepts native Python integers.
        return {"input_ids": [int(token) for token in self.blocks[index]]}


def masked_lm_loss(model, batch):
    labels = batch.pop("labels")
    logits = model(**batch).prediction_logits
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100,
    )


def evaluate_mlm(model, blocks, collator, batch_size, device, seed):
    # Re-seeding makes the pre/post comparison use identical masking patterns.
    torch.manual_seed(seed)
    loader = DataLoader(
        BlockDataset(blocks), batch_size=batch_size, shuffle=False,
        collate_fn=collator,
    )
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(masked_lm_loss(model, batch).cpu()))
    return float(np.mean(losses))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/derived/icr_edinet2023_mlm.parquet",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--max-train-blocks", type=int, default=50000)
    parser.add_argument(
        "--sampling-profile", choices=("broad", "mixed_outlook"),
        default="mixed_outlook",
    )
    parser.add_argument(
        "--outlook-index", type=Path,
        default=ROOT / "data/derived/icr_edinet2023_outlook_index.parquet",
    )
    parser.add_argument("--max-validation-blocks", type=int, default=1000)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    frame = pd.read_parquet(args.input)
    train = frame[frame.split == "train"]
    if args.sampling_profile == "mixed_outlook":
        outlook = pd.read_parquet(args.outlook_index).set_index("sha256")
        train = train.copy()
        train["outlook_rich"] = outlook.loc[train.sha256, "outlook_rich"].to_numpy()
        rich_target = args.max_train_blocks // 2
        rich = train[train.outlook_rich].sample(
            min(rich_target, int(train.outlook_rich.sum())), random_state=args.seed,
        )
        broad_target = args.max_train_blocks - len(rich)
        broad = train[~train.outlook_rich].sample(
            min(broad_target, int((~train.outlook_rich).sum())), random_state=args.seed,
        )
        train = pd.concat([rich, broad]).sample(frac=1, random_state=args.seed)
    elif len(train) > args.max_train_blocks:
        train = train.sample(args.max_train_blocks, random_state=args.seed)
    validation = frame[frame.split == "validation"]
    if len(validation) > args.max_validation_blocks:
        validation = validation.sample(args.max_validation_blocks, random_state=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Retain the pretrained pooler for downstream sequence classification.
    # BertForMaskedLM drops it and would force a random pooler on only 253 rows.
    model = AutoModelForPreTraining.from_pretrained(args.model).to(device)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        BlockDataset(train.input_ids.tolist()), batch_size=args.batch_size,
        shuffle=True, generator=generator, collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01,
    )
    total_updates = int(np.ceil(len(loader) / args.gradient_accumulation)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_updates * args.warmup_fraction)),
        num_training_steps=total_updates,
    )

    validation_blocks = validation.input_ids.tolist()
    baseline_validation_loss = evaluate_mlm(
        model, validation_blocks, collator, args.batch_size, device, args.seed + 1,
    )
    print(f"baseline_validation_mlm_loss={baseline_validation_loss:.6f}", flush=True)

    optimizer.zero_grad()
    updates = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = masked_lm_loss(model, batch) / args.gradient_accumulation
            loss.backward()
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                updates += 1
                if updates % 100 == 0:
                    print(
                        f"epoch={epoch + 1} update={updates} "
                        f"recent_loss={np.mean(losses[-100 * args.gradient_accumulation:]):.6f}",
                        flush=True,
                    )

    validation_loss = evaluate_mlm(
        model, validation_blocks, collator, args.batch_size, device, args.seed + 1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "adaptation.txt").write_text(
        f"base_model={args.model}\nseed={args.seed}\nepochs={args.epochs}\n"
        f"sampling_profile={args.sampling_profile}\n"
        f"train_blocks={len(train)}\nvalidation_blocks={len(validation)}\n"
        f"optimizer_updates={updates}\n"
        f"baseline_validation_mlm_loss={baseline_validation_loss:.12f}\n"
        f"validation_mlm_loss={validation_loss:.12f}\n"
    )
    print(
        f"train_blocks={len(train)} validation_blocks={len(validation)} "
        f"updates={updates} validation_mlm_loss={validation_loss:.6f} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
