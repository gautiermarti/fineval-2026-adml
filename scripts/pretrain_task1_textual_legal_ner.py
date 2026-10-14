#!/usr/bin/env python3
"""Auxiliary Greek legal NER transfer into the three GRFinNER entity types."""

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_task1_textual import LABELS, LABEL_TO_ID  # noqa: E402


TYPE_MAP = {
    "ORG": "ΟΡΓΑΝΙΣΜΟΣ",
    "PERSON": "ΠΡΟΣΩΠΟ",
    "GPE": "ΤΟΠΟΘΕΣΙΑ",
    "LOCATION-UNK": "ΤΟΠΟΘΕΣΙΑ",
    "LOCATION-NAT": "ΤΟΠΟΘΕΣΙΑ",
}


def mapped_tag(tag: str) -> str:
    if tag == "O":
        return "O"
    prefix, kind = tag.split("-", 1)
    return f"{prefix}-{TYPE_MAP[kind]}" if kind in TYPE_MAP else "O"


def encode_row(row: dict, tokenizer, max_length: int):
    encoded = tokenizer(
        row["words"], is_split_into_words=True, truncation=True,
        max_length=max_length,
    )
    labels = []
    previous_word = None
    for word_index in encoded.word_ids():
        if word_index is None:
            labels.append(-100)
            continue
        tag = mapped_tag(row["ner"][word_index])
        if word_index == previous_word and tag.startswith("B-"):
            tag = "I-" + tag[2:]
        labels.append(LABEL_TO_ID[tag])
        previous_word = word_index
    return dict(encoded) | {"labels": labels}


class LegalDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.features = [encode_row(row, tokenizer, max_length) for row in rows]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def collate(rows, tokenizer):
    labels = [row["labels"] for row in rows]
    features = [{key: value for key, value in row.items() if key != "labels"} for row in rows]
    batch = tokenizer.pad(features, return_tensors="pt")
    length = batch["input_ids"].shape[1]
    batch["labels"] = torch.tensor([
        values + [-100] * (length - len(values)) for values in labels
    ])
    return batch


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="spyrosbriakos/greek_legal_bert_v2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--negative-rows", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_textual_aux/legal_ner_mapped",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = ROOT / "data/external/greek_legal_ner/train.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    positive = [row for row in rows if any(mapped_tag(tag) != "O" for tag in row["ner"])]
    negative = [row for row in rows if all(mapped_tag(tag) == "O" for tag in row["ner"])]
    random.shuffle(negative)
    selected = positive + negative[:args.negative_rows]
    random.shuffle(selected)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dataset = LegalDataset(selected, tokenizer, args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID, ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    print(
        f"positive_rows={len(positive)} negative_rows={min(len(negative), args.negative_rows)} "
        f"training_rows={len(dataset)} device={device}", flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach())
        print(f"epoch={epoch} mean_train_loss={total_loss / len(loader):.12f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "source": "joelniklaus/greek_legal_ner",
        "source_license": "cc-by-nc-sa-4.0",
        "source_split": "train",
        "source_test_used": False,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "positive_rows": len(positive),
        "negative_rows": min(len(negative), args.negative_rows),
        "mapped_types": TYPE_MAP,
        "plutus_labels_used": False,
    }, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
