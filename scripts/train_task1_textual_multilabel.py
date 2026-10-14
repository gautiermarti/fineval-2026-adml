#!/usr/bin/env python3
"""Train overlapping BIO heads for person, organization, and location entities."""

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import re
import sys

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import ENTITY_TYPES, repair_embedded_label  # noqa: E402


def align_overlapping(text: str, entities: list[tuple[str, str]]):
    """Align repeated mentions once per type while permitting cross-type overlap."""
    output = []
    used = {kind: [] for kind in ENTITY_TYPES}
    for index, (entity, kind) in sorted(
        enumerate(entities), key=lambda item: (-len(item[1][0]), item[0]),
    ):
        del index
        forms = tuple(dict.fromkeys((entity, repair_embedded_label(entity))))
        candidates = sorted({
            match.span()
            for form in forms
            for match in re.finditer(re.escape(form), text)
        })
        chosen = next((span for span in candidates if not any(
            span[0] < right and span[1] > left for left, right in used[kind]
        )), None)
        if chosen is None:
            continue
        output.append((*chosen, kind))
        used[kind].append(chosen)
    return sorted(output)


def make_examples(frame: pd.DataFrame, tokenizer, drop_fully_unaligned=False):
    examples, dropped = [], []
    for index, row in frame.iterrows():
        text = str(row.text)
        spans = parse_answer(row.answer)
        annotations = align_overlapping(text, spans)
        if drop_fully_unaligned and spans and not annotations:
            dropped.append(int(index))
            continue
        encoded = tokenizer(
            text, return_offsets_mapping=True, truncation=True, max_length=512,
        )
        labels = []
        for left, right in encoded.offset_mapping:
            token_labels = [-100] * len(ENTITY_TYPES) if right <= left else [0] * len(ENTITY_TYPES)
            if right > left:
                for start, end, kind in annotations:
                    if left >= start and right <= end:
                        token_labels[ENTITY_TYPES.index(kind)] = 1 if left == start else 2
            labels.append(token_labels)
        examples.append((encoded, labels, text, spans))
    return examples, dropped


class MultiHeadNER(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(ENTITY_TYPES) * 3)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        logits = self.classifier(self.dropout(hidden)).view(
            hidden.shape[0], hidden.shape[1], len(ENTITY_TYPES), 3,
        )
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.flatten(0, 2), labels.flatten())
        return logits, loss


class NERDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        encoded, labels, _, _ = self.examples[index]
        return {
            key: torch.tensor(value)
            for key, value in encoded.items() if key != "offset_mapping"
        } | {"labels": torch.tensor(labels)}


def collate(rows):
    length = max(row["input_ids"].numel() for row in rows)
    output = {}
    for key in rows[0]:
        if key == "labels":
            output[key] = torch.stack([
                nn.functional.pad(row[key], (0, 0, 0, length - row[key].shape[0]), value=-100)
                for row in rows
            ])
        else:
            output[key] = torch.stack([
                nn.functional.pad(row[key], (0, length - row[key].numel()), value=0)
                for row in rows
            ])
    return output


def decode(logits, examples):
    predictions = []
    for row_logits, (encoded, _, text, _) in zip(logits, examples):
        labels = row_logits.argmax(-1).tolist()
        entities = []
        for type_index, kind in enumerate(ENTITY_TYPES):
            current = None
            for (left, right), token_labels in zip(encoded.offset_mapping, labels):
                label = token_labels[type_index]
                if right <= left or label == 0:
                    if current:
                        entities.append((text[current[0]:current[1]], kind, current[0]))
                        current = None
                    continue
                if label == 1 or current is None:
                    if current:
                        entities.append((text[current[0]:current[1]], kind, current[0]))
                    current = [left, right]
                else:
                    current[1] = right
            if current:
                entities.append((text[current[0]:current[1]], kind, current[0]))
        predictions.append([(entity, kind) for entity, kind, _ in sorted(entities, key=lambda x: x[2])])
    return predictions


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="spyrosbriakos/greek_legal_bert_v2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_textual/multilabel_legal_seed7",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = pd.read_parquet(ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet")
    validation = pd.read_parquet(ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_examples, dropped = make_examples(train, tokenizer, drop_fully_unaligned=True)
    valid_examples, _ = make_examples(validation, tokenizer)
    print(f"train_rows={len(train_examples)} dropped={dropped} validation_rows={len(valid_examples)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadNER(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader = DataLoader(
        NERDataset(train_examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            _, loss = model(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        logits = []
        with torch.no_grad():
            for start in range(0, len(valid_examples), args.batch_size):
                rows = [NERDataset(valid_examples)[i] for i in range(
                    start, min(start + args.batch_size, len(valid_examples))
                )]
                batch = collate(rows)
                values, _ = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                )
                logits.extend(values.cpu())
        predictions = decode(logits, valid_examples)
        value = sum(
            score(parse_answer(row.answer), prediction)[2]
            for row, prediction in zip(validation.itertuples(), predictions)
        ) / len(validation)
        print(f"epoch={epoch} validation_macro_entity_f1={value:.12f}", flush=True)
        if value > best:
            best = value
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_dir / "model.pt")
            tokenizer.save_pretrained(args.output_dir)
            (args.output_dir / "metrics.json").write_text(json.dumps({
                "validation_macro_entity_f1": best,
                "model": args.model,
                "best_epoch": epoch,
                "seed": args.seed,
                "dropped_fully_unaligned_train_rows": dropped,
                "test_split_used_for_selection": False,
            }, indent=2) + "\n")


if __name__ == "__main__":
    main()
