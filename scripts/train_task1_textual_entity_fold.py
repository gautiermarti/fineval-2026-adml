#!/usr/bin/env python3
"""Fixed-epoch entity-disjoint GRFinNER fold training for adaptation ablations."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.task1_folds import (  # noqa: E402
    assign_entity_disjoint_folds,
    audit_entity_disjoint_folds,
)
from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import (  # noqa: E402
    LABELS, LABEL_TO_ID, TextualDataset, collate, decode,
    entity_type_f1, make_examples,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--arm", choices=("base", "matched_random_mlm", "whole_entity_mlm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen downstream test requires CUDA")

    train_path = ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
    frame = pd.read_parquet(train_path)
    entity_rows = [parse_answer(answer) for answer in frame.answer]
    assignments = assign_entity_disjoint_folds(entity_rows, n_splits=5, seed=args.seed)
    split_audit = audit_entity_disjoint_folds(entity_rows, assignments)
    if split_audit["leaking_entity_keys"]:
        raise ValueError("entity-disjoint fold leakage")
    train_indices = [index for index, value in enumerate(assignments) if value != args.fold]
    held_indices = [index for index, value in enumerate(assignments) if value == args.fold]
    train_frame = frame.iloc[train_indices]
    held_frame = frame.iloc[held_indices]

    load_kwargs = {"use_fast": True}
    if args.revision:
        load_kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    train_examples, dropped = make_examples(
        train_frame, tokenizer, drop_fully_unaligned=True,
    )
    held_examples, _ = make_examples(held_frame, tokenizer)
    model_kwargs = {
        "num_labels": len(LABELS), "id2label": dict(enumerate(LABELS)),
        "label2id": LABEL_TO_ID, "ignore_mismatched_sizes": True,
    }
    if args.revision:
        model_kwargs["revision"] = args.revision
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, **model_kwargs,
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01,
    )
    loader = DataLoader(
        TextualDataset(train_examples), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss = model(**batch).loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite NER loss")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))
        print(json.dumps({
            "arm": args.arm, "fold": args.fold, "epoch": epoch,
            "mean_train_loss": epoch_losses[-1],
        }), flush=True)

    model.eval()
    predictions = []
    dataset = TextualDataset(held_examples)
    with torch.no_grad():
        for start in range(0, len(held_examples), args.batch_size):
            stop = min(start + args.batch_size, len(held_examples))
            batch = collate([dataset[index] for index in range(start, stop)])
            logits = model(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
            ).logits.cpu()
            predictions.extend(decode(logits, held_examples[start:stop]))
    gold = [parse_answer(answer) for answer in held_frame.answer]
    row_scores = [
        score(target, prediction)[2]
        for target, prediction in zip(gold, predictions, strict=True)
    ]
    macro = float(np.mean(row_scores))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "model")
    tokenizer.save_pretrained(args.output_dir / "model")
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text("".join(
        json.dumps({
            "row_index": int(row_index), "gold": gold_row,
            "prediction": prediction, "row_entity_f1": row_score,
        }, ensure_ascii=False) + "\n"
        for row_index, gold_row, prediction, row_score in zip(
            held_indices, gold, predictions, row_scores, strict=True,
        )
    ))
    summary = {
        "experiment": "task1_entity_aware_downstream_fold",
        "arm": args.arm, "model": args.model, "revision": args.revision,
        "fold": args.fold, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "learning_rate": args.learning_rate, "weight_decay": 0.01,
        "dtype": "float32", "device": "cuda",
        "train_rows": len(train_frame), "train_examples": len(train_examples),
        "held_rows": len(held_frame), "held_indices": held_indices,
        "dropped_fully_unaligned_train_rows": dropped,
        "epoch_train_losses": epoch_losses,
        "macro_entity_f1": macro,
        "entity_type_f1": entity_type_f1(gold, predictions),
        "predictions_sha256": digest(prediction_path),
        "input_sha256": {"train": digest(train_path)},
        "fixed_epoch_no_held_fold_selection": True,
        "validation_used": False, "public_test_used": False, "hidden_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
