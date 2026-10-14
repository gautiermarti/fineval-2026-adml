#!/usr/bin/env python3
"""Source-group-held-out supervised adaptation of Ruri for JF-ICR."""

from argparse import ArgumentParser
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402

LABELS = ["-2", "-1", "0", "+1", "+2"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
MODEL = "cl-nagoya/ruri-v3-310m"


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], int(self.labels[index])


class Classifier(nn.Module):
    def __init__(self, encoder: SentenceTransformer):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.get_sentence_embedding_dimension(), len(LABELS))

    def forward(self, texts: list[str]) -> torch.Tensor:
        features = self.encoder.tokenize(texts)
        features = {
            key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
            for key, value in features.items()
        }
        return self.head(self.encoder(features)["sentence_embedding"])


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--max-id", type=int)
    args = parser.parse_args()

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    folds_frame = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")
    if args.max_id is not None:
        frame = frame[frame["id"] <= args.max_id].reset_index(drop=True)
        folds_frame = folds_frame[folds_frame["id"] <= args.max_id].reset_index(drop=True)
    if frame["id"].tolist() != folds_frame["id"].tolist():
        raise ValueError("fold rows do not align with input rows")
    parsed = frame["query"].map(parse_icr_query)
    texts = [
        f"トピック: 質問: {item.question}\n会社回答: {item.response}"
        for item in parsed
    ]
    labels = frame["answer"].map(LABEL_TO_ID).to_numpy()
    folds = folds_frame["source_fold"].to_numpy()
    predictions = np.full(len(frame), -1, dtype=int)

    for fold in sorted(np.unique(folds)):
        train_indices = np.flatnonzero(folds != fold)
        test_indices = np.flatnonzero(folds == fold)
        encoder = SentenceTransformer(MODEL, device="cuda")
        encoder.max_seq_length = 2048
        model = Classifier(encoder).cuda()
        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": args.encoder_lr},
                {"params": model.head.parameters(), "lr": args.head_lr},
            ],
            weight_decay=0.01,
        )
        weights = None
        if args.balanced:
            counts = np.bincount(labels[train_indices], minlength=len(LABELS))
            weights = torch.tensor(
                len(train_indices) / (len(LABELS) * np.maximum(counts, 1)),
                dtype=torch.float32,
                device="cuda",
            )
        criterion = nn.CrossEntropyLoss(weight=weights)
        generator = torch.Generator().manual_seed(2026 + int(fold))
        loader = DataLoader(
            TextDataset([texts[i] for i in train_indices], labels[train_indices]),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=lambda batch: ([item[0] for item in batch], torch.tensor([item[1] for item in batch])),
        )
        model.train()
        for epoch in range(args.epochs):
            losses = []
            for batch_texts, batch_labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_texts), batch_labels.cuda(non_blocking=True))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            print(f"fold={fold} epoch={epoch + 1} loss={np.mean(losses):.5f}", flush=True)
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(test_indices), args.batch_size):
                selected = test_indices[start:start + args.batch_size]
                logits = model([texts[i] for i in selected])
                predictions[selected] = logits.argmax(dim=1).cpu().numpy()
        del model, encoder, optimizer
        torch.cuda.empty_cache()

    predicted_labels = np.asarray(LABELS)[predictions]
    gold_labels = np.asarray(LABELS)[labels]
    tag = "balanced" if args.balanced else "plain"
    scope = f"core{args.max_id + 1}" if args.max_id is not None else "all"
    output_dir = ROOT / "artifacts/icr_ruri_finetune"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({
        "id": frame["id"], "gold_label": gold_labels,
        "prediction": predicted_labels, "source_fold": folds,
    })
    output.to_csv(output_dir / f"oof_{scope}_{tag}.csv", index=False)
    print("accuracy", accuracy_score(gold_labels, predicted_labels))
    print("macro_f1", f1_score(gold_labels, predicted_labels, labels=LABELS, average="macro", zero_division=0))
    print("labels", LABELS)
    print(confusion_matrix(gold_labels, predicted_labels, labels=LABELS))


if __name__ == "__main__":
    main()
