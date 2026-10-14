#!/usr/bin/env python3
"""Source-disjoint CV with a monotonic cumulative-ordinal Japanese BERT head."""

from argparse import ArgumentParser
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_icr import rule_prediction  # noqa: E402
from train_icr_japanese_cv import (  # noqa: E402
    DEFAULT_MODEL, ICRDataset, LABELS, collate,
)


def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


class MonotonicOrdinalClassifier(nn.Module):
    """Predict P(y > k) using one score and four strictly ordered cutpoints."""

    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        dropout = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden, 1)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(
            torch.full((3,), inverse_softplus(1.0)),
        )

    def cutpoints(self) -> torch.Tensor:
        increments = F.softplus(self.cutpoint_increments)
        return torch.cat((
            self.first_cutpoint[None],
            self.first_cutpoint + torch.cumsum(increments, dim=0),
        ))

    def forward(self, **features) -> torch.Tensor:
        encoded = self.encoder(**features)
        pooled = getattr(encoded, "pooler_output", None)
        if pooled is None:
            pooled = encoded.last_hidden_state[:, 0]
        score = self.score(self.dropout(pooled))
        return score - self.cutpoints()[None, :]


def cumulative_targets(labels: torch.Tensor) -> torch.Tensor:
    boundaries = torch.arange(4, device=labels.device)
    return (labels[:, None] > boundaries[None, :]).float()


def decode_logits(logits: torch.Tensor) -> torch.Tensor:
    """Ordered class index in [0, 4]."""
    return (logits > 0).sum(dim=1)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task2_japanese_ordinal_cv/seed7",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    folds = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    frame["fold"] = folds.loc[frame.id].source_fold.to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_logits = np.full((len(frame), 4), np.nan, dtype=np.float32)

    for fold in sorted(frame["fold"].unique()):
        # Reset per fold so fold order cannot alter initialization or shuffling.
        fold_seed = args.seed * 100 + int(fold)
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        train_frame = frame[frame.fold != fold]
        valid_frame = frame[frame.fold == fold]
        train = ICRDataset(train_frame, tokenizer, args.max_length)
        valid = ICRDataset(valid_frame, tokenizer, args.max_length)
        model = MonotonicOrdinalClassifier(args.model).to(device)
        head_parameters = [
            *model.score.parameters(), model.first_cutpoint, model.cutpoint_increments,
        ]
        head_ids = {id(parameter) for parameter in head_parameters}
        encoder_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in head_ids
        ]
        optimizer = torch.optim.AdamW([
            {"params": encoder_parameters, "lr": args.learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ], weight_decay=0.01)
        generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(
            train, batch_size=args.batch_size, shuffle=True, generator=generator,
            collate_fn=lambda items: collate(items, tokenizer),
        )

        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in train_loader:
                labels = batch.pop("labels").to(device)
                features = {key: value.to(device) for key, value in batch.items()}
                logits = model(**features)
                loss = F.binary_cross_entropy_with_logits(
                    logits, cumulative_targets(labels),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                losses.append(float(loss.detach().cpu()))
            print(
                f"fold={fold} epoch={epoch + 1} loss={np.mean(losses):.6f} "
                f"cutpoints={model.cutpoints().detach().cpu().tolist()}",
                flush=True,
            )

        model.eval()
        fold_logits = []
        with torch.no_grad():
            valid_loader = DataLoader(
                valid, batch_size=args.batch_size,
                collate_fn=lambda items: collate(items, tokenizer),
            )
            for batch in valid_loader:
                batch.pop("labels")
                features = {key: value.to(device) for key, value in batch.items()}
                fold_logits.append(model(**features).cpu())
        fold_logits = torch.cat(fold_logits)
        all_logits[valid_frame.index] = fold_logits.numpy()
        predictions = decode_logits(fold_logits).numpy()
        gold = np.asarray(valid.labels)
        print(
            f"fold={fold} n={len(valid)} raw_accuracy={np.mean(predictions == gold):.6f}",
            flush=True,
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gold = np.asarray([LABELS.index(str(value)) for value in frame.answer])
    raw_predictions = (all_logits > 0).sum(axis=1)
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])
    rule_indices = np.asarray([LABELS.index(label) for label in rules])
    predictions = np.where(rules != "+1", rule_indices, raw_predictions)
    raw_accuracy = float(np.mean(raw_predictions == gold))
    ruled_accuracy = float(np.mean(predictions == gold))
    print(f"source_cv_raw_accuracy={raw_accuracy:.6f}")
    print(f"source_cv_safe_rules_accuracy={ruled_accuracy:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({
        "id": frame.id,
        "gold": frame.answer.astype(str),
        "prediction": [LABELS[index] for index in predictions],
        "raw_prediction": [LABELS[index] for index in raw_predictions],
        "fold": frame.fold,
        **{f"cumulative_logit_{index}": all_logits[:, index] for index in range(4)},
    })
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    (args.output_dir / "metrics.txt").write_text(
        f"model={args.model}\nseed={args.seed}\nepochs={args.epochs}\n"
        f"source_cv_raw_accuracy={raw_accuracy:.12f}\n"
        f"source_cv_safe_rules_accuracy={ruled_accuracy:.12f}\n"
    )


if __name__ == "__main__":
    main()
