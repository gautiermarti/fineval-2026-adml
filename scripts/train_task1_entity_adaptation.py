#!/usr/bin/env python3
"""Matched random-token versus whole-entity MLM adaptation for Task 1."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "spyrosbriakos/greek_legal_bert_v2"
DEFAULT_REVISION = "e06fba0eb7a107f401055bb174f9e1ba1a29ba17"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SilverDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            encoded = tokenizer(
                row["text"], truncation=True, max_length=max_length,
                return_offsets_mapping=True,
            )
            entity_groups = []
            for span in row["spans"]:
                group = [
                    index for index, (left, right) in enumerate(encoded["offset_mapping"])
                    if right > left and left >= span["start"] and right <= span["end"]
                ]
                if group:
                    entity_groups.append(group)
            if entity_groups:
                encoded.pop("offset_mapping")
                self.rows.append((encoded, entity_groups))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class MatchedMLMCollator:
    def __init__(self, tokenizer, mode: str, seed: int, probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mode = mode
        self.probability = probability
        self.generator = torch.Generator().manual_seed(seed)
        if tokenizer.mask_token_id is None:
            raise ValueError("tokenizer has no mask token")

    def __call__(self, rows):
        encodings, entity_groups = zip(*rows, strict=True)
        batch = self.tokenizer.pad(encodings, padding=True, return_tensors="pt")
        inputs = batch["input_ids"].clone()
        masked = torch.zeros_like(inputs, dtype=torch.bool)
        special_ids = set(self.tokenizer.all_special_ids)
        for row_index, groups in enumerate(entity_groups):
            eligible = torch.tensor([
                token_id not in special_ids and bool(batch["attention_mask"][row_index, index])
                for index, token_id in enumerate(inputs[row_index].tolist())
            ])
            eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
            target = max(1, int(round(self.probability * len(eligible_indices))))
            selected = set()
            if self.mode == "whole_entity_mlm":
                group_index = int(torch.randint(
                    len(groups), (1,), generator=self.generator,
                ).item())
                selected.update(index for index in groups[group_index] if eligible[index])
            remaining = [index.item() for index in eligible_indices if index.item() not in selected]
            needed = max(0, target - len(selected))
            if needed and remaining:
                order = torch.randperm(len(remaining), generator=self.generator)[:needed]
                selected.update(remaining[index] for index in order.tolist())
            if selected:
                masked[row_index, list(selected)] = True

        labels = inputs.clone()
        labels[~masked] = -100
        replacement_draw = torch.rand(inputs.shape, generator=self.generator)
        replace_mask = masked & (replacement_draw < 0.8)
        random_mask = masked & (replacement_draw >= 0.8) & (replacement_draw < 0.9)
        inputs[replace_mask] = self.tokenizer.mask_token_id
        random_tokens = torch.randint(
            len(self.tokenizer), inputs.shape, generator=self.generator,
        )
        inputs[random_mask] = random_tokens[random_mask]
        batch["input_ids"] = inputs
        batch["labels"] = labels
        return batch


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=("matched_random_mlm", "whole_entity_mlm"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "artifacts/task1_entity_aware_fold0_v1/silver/corpus.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen adaptation requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, use_fast=True,
    )
    dataset = SilverDataset(args.corpus, tokenizer, args.max_length)
    if len(dataset) < 512:
        raise ValueError(f"tokenized silver corpus below frozen minimum: {len(dataset)}")
    collator = MatchedMLMCollator(tokenizer, args.mode, args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
        generator=loader_generator, num_workers=0,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, revision=args.revision,
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01,
    )
    model.train()
    iterator = iter(loader)
    losses = []
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to("cuda") for key, value in batch.items()}
        loss = model(**batch).loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite MLM loss")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % 32 == 0:
            print(json.dumps({
                "mode": args.mode, "step": step, "loss": losses[-1],
                "mean_loss": float(np.mean(losses)),
            }), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    summary = {
        "experiment": "task1_entity_aware_adaptation",
        "mode": args.mode, "model": args.model, "revision": args.revision,
        "seed": args.seed, "max_length": args.max_length,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": 0.01, "max_steps": args.max_steps,
        "mask_probability": 0.15,
        "replacement_policy": "80_percent_mask_10_percent_random_10_percent_unchanged",
        "dtype": "float32", "device": "cuda", "tokenized_rows": len(dataset),
        "mean_loss": float(np.mean(losses)), "final_loss": losses[-1],
        "corpus": str(args.corpus), "corpus_sha256": digest(args.corpus),
        "validation_used": False, "public_test_used": False, "hidden_test_used": False,
    }
    (args.output_dir / "adaptation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
