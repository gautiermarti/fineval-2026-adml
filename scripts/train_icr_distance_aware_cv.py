#!/usr/bin/env python3
"""Preregistered source-disjoint JF-ICR training with CE + ordered-CDF loss."""

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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.icr_ordinal_loss import distance_aware_cross_entropy  # noqa: E402
from train_icr_japanese_cv import (  # noqa: E402
    ICRDataset, LABELS, LABEL_TO_ID, build_oof_output, collate,
)


DEFAULT_MODEL = "cl-tohoku/bert-base-japanese-v3"
DEFAULT_REVISION = "65243d6e5629b969c77309f217bd7b1a79d43c7e"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--distance-weight", type=float, default=0.5)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task2_distance_aware_seed7_v1",
    )
    args = parser.parse_args()
    if args.distance_weight < 0:
        raise ValueError("--distance-weight must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    fold_path = ROOT / "data/derived/icr_source_folds.csv"
    frame = pd.read_parquet(data_path)
    fold_table = pd.read_csv(fold_path).set_index("id")
    frame["fold"] = fold_table.loc[frame.id].source_fold.to_numpy()
    available_folds = sorted(int(value) for value in frame.fold.unique())
    fold_values = available_folds if args.folds is None else args.folds
    if len(set(fold_values)) != len(fold_values) or not set(fold_values) <= set(available_folds):
        raise ValueError("--folds contains duplicates or unavailable source folds")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision,
    )
    all_logits = np.full((len(frame), len(LABELS)), np.nan, dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_audits = []

    for fold in fold_values:
        train_frame = frame[frame.fold != fold]
        valid_frame = frame[frame.fold == fold]
        train = ICRDataset(train_frame, tokenizer, args.max_length)
        valid = ICRDataset(valid_frame, tokenizer, args.max_length)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, revision=args.revision, num_labels=len(LABELS),
            ignore_mismatched_sizes=True,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.01,
        )
        train_loader = DataLoader(
            train, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda items: collate(items, tokenizer),
        )
        epoch_audits = []
        for epoch in range(args.epochs):
            model.train()
            totals = {"total": [], "cross_entropy": [], "ordered_cdf": []}
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch.pop("labels")
                logits = model(**batch).logits
                loss, cross_entropy, ordered_cdf = distance_aware_cross_entropy(
                    logits, labels, distance_weight=args.distance_weight,
                )
                if not all(bool(torch.isfinite(value)) for value in (
                    loss, cross_entropy, ordered_cdf,
                )):
                    raise ValueError("non-finite distance-aware loss component")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                totals["total"].append(float(loss.detach().cpu()))
                totals["cross_entropy"].append(float(cross_entropy.detach().cpu()))
                totals["ordered_cdf"].append(float(ordered_cdf.detach().cpu()))
            epoch_audit = {
                "epoch": epoch + 1,
                **{f"mean_{key}": float(np.mean(values)) for key, values in totals.items()},
            }
            epoch_audits.append(epoch_audit)
            print(json.dumps({"fold": fold, **epoch_audit}), flush=True)

        model.eval()
        fold_logits = []
        with torch.no_grad():
            valid_loader = DataLoader(
                valid, batch_size=args.batch_size,
                collate_fn=lambda items: collate(items, tokenizer),
            )
            for batch in valid_loader:
                batch.pop("labels")
                batch = {key: value.to(device) for key, value in batch.items()}
                fold_logits.append(model(**batch).logits.float().cpu().numpy())
        values = np.concatenate(fold_logits)
        all_logits[valid_frame.index] = values
        raw_accuracy = float(np.mean(
            values.argmax(axis=1) == np.asarray(valid.labels),
        ))
        fold_audit = {
            "fold": int(fold), "train_rows": len(train), "validation_rows": len(valid),
            "raw_argmax_accuracy": raw_accuracy, "epochs": epoch_audits,
        }
        fold_audits.append(fold_audit)
        print(json.dumps(fold_audit), flush=True)
        build_oof_output(frame, all_logits).to_csv(
            args.output_dir / "oof_predictions.partial.csv", index=False,
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    evaluated = ~np.isnan(all_logits).any(axis=1)
    gold = np.asarray([LABEL_TO_ID[str(value)] for value in frame.answer])
    raw_accuracy = float(np.mean(all_logits[evaluated].argmax(axis=1) == gold[evaluated]))
    output = build_oof_output(frame, all_logits)
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    summary = {
        "experiment": "task2_distance_aware_five_class_cv",
        "model": args.model, "revision": args.revision, "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_length": args.max_length, "learning_rate": args.learning_rate,
        "weight_decay": 0.01, "distance_weight": args.distance_weight,
        "distance_definition": "mean squared ordered CDF discrepancy over K-1 thresholds",
        "device": str(device), "dtype": "float32", "folds_completed": fold_values,
        "rows_evaluated": int(evaluated.sum()), "raw_argmax_accuracy": raw_accuracy,
        "fold_audits": fold_audits,
        "input_sha256": {"benchmark": digest(data_path), "source_folds": digest(fold_path)},
        "current_or_hidden_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
