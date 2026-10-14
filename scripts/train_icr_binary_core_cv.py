#!/usr/bin/env python3
"""Source-disjoint JF-ICR model specialized for the 0 versus +1 boundary."""

from argparse import ArgumentParser
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.parsing import parse_icr_query  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402


LABELS = ["0", "+1"]


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int, labeled: bool):
        self.features = []
        self.labels = []
        for row in frame.itertuples():
            parsed = parse_icr_query(row.query)
            self.features.append(tokenizer(
                parsed.question, parsed.response,
                truncation=True, max_length=max_length,
            ))
            if labeled:
                self.labels.append(LABELS.index(str(row.answer)))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.features[index])
        if self.labels:
            item["labels"] = self.labels[index]
        return item


def collate(batch: list[dict], tokenizer) -> dict[str, torch.Tensor]:
    labels = [item.get("labels") for item in batch]
    features = [{key: value for key, value in item.items() if key != "labels"} for item in batch]
    encoded = tokenizer.pad(features, return_tensors="pt")
    if labels[0] is not None:
        encoded["labels"] = torch.tensor(labels)
    return encoded


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--model", default="cl-tohoku/bert-base-japanese-v3")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    fold_table = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    frame["fold"] = fold_table.loc[frame.id].source_fold.to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_logits = np.full((len(frame), 2), np.nan, dtype=np.float32)

    for fold in sorted(frame.fold.unique()):
        train_frame = frame[(frame.fold != fold) & frame.answer.isin(LABELS)]
        valid_frame = frame[frame.fold == fold]
        train = PairDataset(train_frame, tokenizer, 512, labeled=True)
        valid = PairDataset(valid_frame, tokenizer, 512, labeled=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2, ignore_mismatched_sizes=True,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        loader = DataLoader(
            train, batch_size=8, shuffle=True,
            collate_fn=lambda items: collate(items, tokenizer),
        )
        for _ in range(args.epochs):
            model.train()
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        fold_logits = []
        with torch.no_grad():
            loader = DataLoader(
                valid, batch_size=8,
                collate_fn=lambda items: collate(items, tokenizer),
            )
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                fold_logits.append(model(**batch).logits.cpu().numpy())
        fold_logits = np.concatenate(fold_logits)
        all_logits[valid_frame.index] = fold_logits
        raw = np.asarray(LABELS)[fold_logits.argmax(axis=1)]
        rules = valid_frame["query"].map(
            lambda query: rule_prediction(query, "safe_rules")
        ).to_numpy()
        hybrid = np.where(rules != "+1", rules, raw)
        gold = valid_frame.answer.astype(str).to_numpy()
        print(
            f"fold={fold} n={len(valid_frame)} raw={np.mean(raw == gold):.6f} "
            f"hybrid={np.mean(hybrid == gold):.6f}", flush=True,
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw = np.asarray(LABELS)[all_logits.argmax(axis=1)]
    rules = frame["query"].map(
        lambda query: rule_prediction(query, "safe_rules")
    ).to_numpy()
    hybrid = np.where(rules != "+1", rules, raw)
    gold = frame.answer.astype(str).to_numpy()
    print(f"source_cv_raw={np.mean(raw == gold):.6f}")
    print(f"source_cv_hybrid={np.mean(hybrid == gold):.6f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id": frame.id, "gold": gold, "prediction": raw,
        "hybrid_prediction": hybrid, "fold": frame.fold,
        "logit_0": all_logits[:, 0], "logit_+1": all_logits[:, 1],
    }).to_csv(args.output_dir / "oof_predictions.csv", index=False)


if __name__ == "__main__":
    main()
