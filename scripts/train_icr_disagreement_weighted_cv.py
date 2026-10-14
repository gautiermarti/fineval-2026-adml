#!/usr/bin/env python3
"""Source-disjoint JF-ICR training with adjudication-disagreement weighting."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_icr_japanese_cv import (  # noqa: E402
    ICRDataset,
    LABELS,
    build_oof_output,
    collate,
)


CURATED = ROOT / "data/external/ebisu_human_rationales/rationales.jsonl"
EXPECTED_CURATED_SHA256 = (
    "b0023a2e3fa7b6ec505ca19667ec831105a138107fa1b1a8948a2f153caac452"
)
DEFAULT_MODEL = "cl-tohoku/bert-base-japanese-v3"
DEFAULT_REVISION = "65243d6e5629b969c77309f217bd7b1a79d43c7e"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_disagreement_ids(path: Path = CURATED) -> set[int]:
    observed = digest(path)
    if observed != EXPECTED_CURATED_SHA256:
        raise ValueError(f"curated annotation hash changed: {observed}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    values = {
        int(row["benchmark_id"])
        for row in rows
        if len(set(row["annotation_labels"])) > 1
    }
    if len(values) != 14:
        raise ValueError(f"expected 14 disagreement rows, observed {len(values)}")
    return values


class WeightedDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, tokenizer, max_length: int,
        disagreement_ids: set[int], disagreement_weight: float,
    ) -> None:
        self.base = ICRDataset(frame, tokenizer, max_length)
        self.weights = [
            disagreement_weight if int(row_id) in disagreement_ids else 1.0
            for row_id in frame.id
        ]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        return self.base[index] | {"sample_weight": self.weights[index]}


def collate_weighted(batch: list[dict], tokenizer) -> dict[str, torch.Tensor]:
    weights = torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float)
    ordinary = [
        {key: value for key, value in item.items() if key != "sample_weight"}
        for item in batch
    ]
    encoded = collate(ordinary, tokenizer)
    encoded["sample_weight"] = weights
    return encoded


def weighted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--disagreement-weight", type=float, default=2.0)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task2_disagreement_weighted_cv_seed7_v1",
    )
    args = parser.parse_args()
    if args.disagreement_weight != 2.0:
        raise ValueError("preregistered disagreement weight is exactly 2.0")
    if args.seed != 7 or args.epochs != 4 or args.learning_rate != 2e-5:
        raise ValueError("seed, epochs, and learning rate are preregistered")
    if len(set(args.folds)) != len(args.folds):
        raise ValueError("duplicate fold requested")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.bf16 and device.type != "cuda":
        raise ValueError("--bf16 requires CUDA")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    source_folds = pd.read_csv(
        ROOT / "data/derived/icr_source_folds.csv"
    ).set_index("id")
    frame["fold"] = source_folds.loc[frame.id].source_fold.to_numpy()
    valid_folds = set(int(value) for value in frame.fold.unique())
    if not set(args.folds).issubset(valid_folds):
        raise ValueError(f"invalid folds: {set(args.folds) - valid_folds}")
    disagreement_ids = load_disagreement_ids()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision,
    )
    all_logits = np.full((len(frame), len(LABELS)), np.nan, dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_records = []

    for fold in args.folds:
        train_frame = frame[frame.fold != fold]
        valid_frame = frame[frame.fold == fold]
        train_disagreement_ids = disagreement_ids & set(
            train_frame.id.astype(int)
        )
        held_disagreement_ids = disagreement_ids & set(
            valid_frame.id.astype(int)
        )
        train = WeightedDataset(
            train_frame, tokenizer, args.max_length,
            train_disagreement_ids, args.disagreement_weight,
        )
        valid = ICRDataset(valid_frame, tokenizer, args.max_length)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            revision=args.revision,
            num_labels=len(LABELS),
            ignore_mismatched_sizes=True,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.01,
        )
        train_loader = DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda items: collate_weighted(items, tokenizer),
        )
        epoch_losses = []
        for _ in range(args.epochs):
            model.train()
            losses = []
            for batch in train_loader:
                weights = batch.pop("sample_weight").to(device)
                labels = batch.pop("labels").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    logits = model(**batch).logits
                    loss = weighted_cross_entropy(logits, labels, weights)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(float(loss.detach().cpu()))
            epoch_losses.append(float(np.mean(losses)))

        model.eval()
        fold_logits = []
        valid_loader = DataLoader(
            valid,
            batch_size=args.batch_size,
            collate_fn=lambda items: collate(items, tokenizer),
        )
        with torch.no_grad():
            for batch in valid_loader:
                batch.pop("labels")
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    logits = model(**batch).logits
                fold_logits.append(logits.float().cpu().numpy())
        values = np.concatenate(fold_logits)
        all_logits[valid_frame.index] = values
        accuracy = float(np.mean(
            values.argmax(axis=1) == np.asarray(valid.labels)
        ))
        fold_dir = args.output_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)
        fold_record = {
            "fold": int(fold),
            "train_rows": len(train_frame),
            "validation_rows": len(valid_frame),
            "train_disagreement_rows": len(train_disagreement_ids),
            "held_disagreement_rows_not_used": len(held_disagreement_ids),
            "epoch_losses": epoch_losses,
            "argmax_accuracy": accuracy,
        }
        fold_records.append(fold_record)
        print(json.dumps(fold_record), flush=True)
        build_oof_output(frame, all_logits).to_csv(
            args.output_dir / "oof_predictions.partial.csv", index=False,
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = build_oof_output(frame, all_logits)
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    metrics = {
        "experiment": "icr_disagreement_weighted_hard_gold",
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "disagreement_weight": args.disagreement_weight,
        "folds_completed": args.folds,
        "rows_evaluated": len(output),
        "disagreement_rows_total": len(disagreement_ids),
        "folds": fold_records,
        "curated_annotations_sha256": digest(CURATED),
        "current_or_hidden_test_used": False,
        "held_fold_annotations_enter_training_dataset": False,
        "submission_policy_status": "policy_gated_pending_license_clarification",
        "device": str(device),
        "bf16": args.bf16,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

