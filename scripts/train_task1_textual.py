#!/usr/bin/env python3
"""Train a Greek financial textual NER model on GRFinNER train only."""

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import random
import re
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task1_numeric_baseline import parse_answer, score  # noqa: E402


ENTITY_TYPES = ("ΠΡΟΣΩΠΟ", "ΟΡΓΑΝΙΣΜΟΣ", "ΤΟΠΟΘΕΣΙΑ")
LABELS = ["O"] + [f"{prefix}-{kind}" for kind in ENTITY_TYPES for prefix in ("B", "I")]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def entity_type_f1(
    gold: list[list[tuple[str, str]]],
    predictions: list[list[tuple[str, str]]],
) -> dict[str, float]:
    """Micro entity F1 within each textual entity type."""
    result = {}
    for entity_type in ENTITY_TYPES:
        hits = guesses = targets = 0
        for target, prediction in zip(gold, predictions, strict=True):
            target_set = {item for item in target if item[1] == entity_type}
            prediction_set = {item for item in prediction if item[1] == entity_type}
            hits += len(target_set & prediction_set)
            guesses += len(prediction_set)
            targets += len(target_set)
        precision = hits / guesses if guesses else 0.0
        recall = hits / targets if targets else 0.0
        result[entity_type] = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return result


def load_release_split(split: str) -> tuple[pd.DataFrame, Path, str]:
    path = ROOT / f"data/external/plutus_finner_text/{split}-00000-of-00001.parquet"
    payload_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_parquet(path).copy()
    frame["source_split"] = split
    frame["source_row_index"] = range(len(frame))
    return frame, path, payload_hash


def repair_embedded_label(entity: str) -> str:
    """Undo the released dataset's accidental replacement of Latin PER."""
    return entity.replace("ΠΡΟΣΩΠΟ", "PER") if "ΠΡΟΣΩΠΟ" in entity else entity


def encode_release_entity(entity: str, entity_type: str) -> str:
    """Reproduce the release's systematic uppercase ``PER`` answer corruption.

    Mixed/lowercase ``per`` remains intact in published labels, so this applies
    only to uppercase ``PER`` inside predicted organization names.
    """
    entity = re.sub(r"\s+", " ", entity).strip()
    if entity_type == "ΟΡΓΑΝΙΣΜΟΣ" and "PER" in entity:
        return entity.replace("PER", "ΠΡΟΣΩΠΟ")
    return entity


def align_annotations(text: str, spans: list[tuple[str, str]]) -> list[tuple[int, int, str]]:
    """Align answer entities, retaining the first/outer entity for nested spans."""
    annotations: list[tuple[int, int, str]] = []
    used: list[tuple[int, int]] = []
    # Dataset answers normally put the containing organization before a nested
    # location. Longest-first makes that convention explicit and deterministic.
    indexed = list(enumerate(spans))
    indexed.sort(key=lambda item: (-len(item[1][0]), item[0]))
    for _, (entity, entity_type) in indexed:
        forms = (entity, repair_embedded_label(entity))
        candidates = [
            match.span()
            for form in dict.fromkeys(forms)
            for match in re.finditer(re.escape(form), text)
        ]
        candidates.sort()
        chosen = next((span for span in candidates if not any(
            span[0] < right and span[1] > left for left, right in used
        )), None)
        if chosen is None:
            continue
        annotations.append((*chosen, entity_type))
        used.append(chosen)
    return sorted(annotations)


def make_examples(frame: pd.DataFrame, tokenizer, drop_fully_unaligned: bool = False):
    examples = []
    dropped = []
    for index, row in frame.iterrows():
        text = str(row.text)
        spans = parse_answer(row.answer)
        annotations = align_annotations(text, spans)
        if drop_fully_unaligned and spans and not annotations:
            dropped.append(int(index))
            continue
        encoded = tokenizer(
            text, return_offsets_mapping=True, truncation=True, max_length=512,
        )
        labels = []
        for left, right in encoded.offset_mapping:
            label = "O"
            if right > left:
                for start, end, entity_type in annotations:
                    if left >= start and right <= end:
                        label = ("B-" if left == start else "I-") + entity_type
                        break
            labels.append(LABEL_TO_ID[label])
        examples.append((encoded, labels, text, spans))
    return examples, dropped


class TextualDataset(Dataset):
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


def collate(batch):
    length = max(row["input_ids"].numel() for row in batch)
    output = {}
    for key in batch[0]:
        padding = -100 if key == "labels" else 0
        output[key] = torch.stack([
            torch.nn.functional.pad(row[key], (0, length - row[key].numel()), value=padding)
            for row in batch
        ])
    return output


def decode(logits, examples):
    results = []
    for logit, (encoded, _, text, _) in zip(logits, examples):
        predicted = logit.argmax(-1).tolist()
        entities = []
        current = None
        for (left, right), label_id in zip(encoded.offset_mapping, predicted):
            if right <= left or label_id == 0:
                continue
            prefix, entity_type = LABELS[label_id].split("-", 1)
            if prefix == "B" or current is None or current[2] != entity_type:
                if current:
                    entities.append(current)
                current = [left, right, entity_type]
            else:
                current[1] = right
        if current:
            entities.append(current)
        results.append([(text[left:right], kind) for left, right, kind in entities])
    return results


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="spyrosbriakos/greek_legal_bert_v2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--train-splits", nargs="+", choices=("train", "validation", "test"),
        default=["train"],
        help="Released Plutus splits used for training (default: train).",
    )
    parser.add_argument(
        "--evaluation-split", choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_textual/legal_base_seed7",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(set(args.train_splits)) != len(args.train_splits):
        raise ValueError("--train-splits must not contain duplicates")
    if args.evaluation_split in args.train_splits:
        raise ValueError("evaluation split must be disjoint from training splits")
    loaded_train = [load_release_split(split) for split in args.train_splits]
    train = pd.concat([item[0] for item in loaded_train], ignore_index=True)
    evaluation, evaluation_path, evaluation_hash = load_release_split(
        args.evaluation_split
    )
    train_provenance = [
        {"split": split, "path": str(path.relative_to(ROOT)), "sha256": digest,
         "rows": len(frame)}
        for split, (frame, path, digest) in zip(args.train_splits, loaded_train, strict=True)
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_examples, dropped = make_examples(train, tokenizer, drop_fully_unaligned=True)
    evaluation_examples, _ = make_examples(evaluation, tokenizer)
    print(
        f"train_rows={len(train_examples)} dropped_fully_unaligned={dropped} "
        f"evaluation_split={args.evaluation_split} "
        f"evaluation_rows={len(evaluation_examples)}",
        flush=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID, ignore_mismatched_sizes=True,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device_name = (
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01,
    )
    loader = DataLoader(
        TextualDataset(train_examples), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate,
    )
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(evaluation_examples), args.batch_size):
                examples = evaluation_examples[start:start + args.batch_size]
                rows = [TextualDataset(evaluation_examples)[index] for index in range(
                    start, min(start + args.batch_size, len(evaluation_examples))
                )]
                batch = collate(rows)
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits.cpu()
                predictions.extend(decode(logits, examples))
        macro_f1 = sum(
            score(parse_answer(row.answer), prediction)[2]
            for row, prediction in zip(evaluation.itertuples(), predictions, strict=True)
        ) / len(evaluation)
        print(
            f"epoch={epoch} {args.evaluation_split}_macro_entity_f1={macro_f1:.12f}",
            flush=True,
        )
        if macro_f1 > best:
            best = macro_f1
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            gold = [parse_answer(row.answer) for row in evaluation.itertuples()]
            row_scores = [
                score(target, prediction)[2]
                for target, prediction in zip(gold, predictions, strict=True)
            ]
            metrics = {
                "evaluation_macro_entity_f1": best,
                f"{args.evaluation_split}_macro_entity_f1": best,
                "evaluation_type_f1": entity_type_f1(gold, predictions),
                "model": args.model,
                "best_epoch": epoch,
                "seed": args.seed,
                "dropped_fully_unaligned_train_rows": dropped,
                "train_splits": args.train_splits,
                "evaluation_split": args.evaluation_split,
                "train_provenance": train_provenance,
                "evaluation_provenance": {
                    "split": args.evaluation_split,
                    "path": str(evaluation_path.relative_to(ROOT)),
                    "sha256": evaluation_hash,
                    "rows": len(evaluation),
                },
                "test_split_used_for_training": "test" in args.train_splits,
                "test_split_used_for_selection": args.evaluation_split == "test",
                "validation_split_used_for_training": "validation" in args.train_splits,
                "device": device_name,
            }
            (args.output_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
            )
            with (args.output_dir / "evaluation_predictions.jsonl").open("w") as handle:
                for row, prediction, row_f1 in zip(
                    evaluation.itertuples(), predictions, row_scores, strict=True
                ):
                    handle.write(json.dumps({
                        "evaluation_split": args.evaluation_split,
                        "source_row_index": int(row.source_row_index),
                        "text_sha256": hashlib.sha256(str(row.text).encode()).hexdigest(),
                        "entities": prediction,
                        "row_macro_entity_f1": row_f1,
                    }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
