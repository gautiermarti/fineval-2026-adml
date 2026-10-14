#!/usr/bin/env python3
"""Generate entity-disjoint OOF predictions from a nested Greek BERT NER model.

The fixed-epoch fold models provide an honest incumbent error distribution for
learning GLiNER add/replace/retain-nested operations.  Neither the official
validation split nor the public test is read.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.task1_folds import (  # noqa: E402
    assign_entity_disjoint_folds,
    audit_entity_disjoint_folds,
)
from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import ENTITY_TYPES, encode_release_entity  # noqa: E402
from train_task1_textual_multilabel import (  # noqa: E402
    MultiHeadNER,
    NERDataset,
    collate,
    make_examples,
)


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def dump_jsonl(path: Path, rows) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))


def decode_entities(logits, examples):
    """Decode nested heads with char offsets and mean token confidence."""
    predictions = []
    for row_logits, (encoded, _, text, _) in zip(logits, examples):
        probabilities = row_logits.softmax(-1)
        labels = probabilities.argmax(-1).tolist()
        entities = []
        for type_index, kind in enumerate(ENTITY_TYPES):
            current = None
            confidence = []
            for token_index, ((left, right), token_labels) in enumerate(
                zip(encoded.offset_mapping, labels)
            ):
                label = token_labels[type_index]
                if right <= left or label == 0:
                    if current:
                        raw_surface = text[current[0]:current[1]]
                        entities.append({
                            "start": current[0], "end": current[1], "type": kind,
                            "surface_text": raw_surface,
                            "entity": encode_release_entity(raw_surface, kind),
                            "confidence": sum(confidence) / len(confidence),
                        })
                        current, confidence = None, []
                    continue
                token_confidence = float(probabilities[token_index, type_index, label])
                if label == 1 or current is None:
                    if current:
                        raw_surface = text[current[0]:current[1]]
                        entities.append({
                            "start": current[0], "end": current[1], "type": kind,
                            "surface_text": raw_surface,
                            "entity": encode_release_entity(raw_surface, kind),
                            "confidence": sum(confidence) / len(confidence),
                        })
                    current, confidence = [left, right], [token_confidence]
                else:
                    current[1] = right
                    confidence.append(token_confidence)
            if current:
                raw_surface = text[current[0]:current[1]]
                entities.append({
                    "start": current[0], "end": current[1], "type": kind,
                    "surface_text": raw_surface,
                    "entity": encode_release_entity(raw_surface, kind),
                    "confidence": sum(confidence) / len(confidence),
                })
        entities.sort(key=lambda item: (item["start"], item["end"], item["type"]))
        predictions.append(entities)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spyrosbriakos/greek_legal_bert_v2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", dest="folds", type=int, action="append")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_multilabel_oof_legal_seed7",
    )
    args = parser.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu for an intentional CPU run")
    requested_folds = list(range(args.n_splits)) if args.folds is None else args.folds
    if any(fold < 0 or fold >= args.n_splits for fold in requested_folds):
        raise ValueError("requested fold is outside [0, n_splits)")

    frame = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
    )
    entity_rows = [parse_answer(answer) for answer in frame["answer"]]
    assignments = assign_entity_disjoint_folds(
        entity_rows, n_splits=args.n_splits, seed=args.seed,
    )
    split_audit = audit_entity_disjoint_folds(entity_rows, assignments)
    if split_audit["leaking_entity_keys"]:
        raise AssertionError("entity-disjoint split leaked a linked entity")

    config = {
        "base_model": args.model,
        "architecture": "independent overlapping BIO heads",
        "seed": args.seed,
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "split": "normalized exact organization/person connected components",
        "official_validation_or_public_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(
            f"{args.output_dir} contains a different configuration; use a new path"
        )
    dump_json(config_path, config)
    pd.DataFrame({
        "row_index": range(len(frame)), "fold": assignments,
    }).to_csv(args.output_dir / "folds.csv", index=False)
    split_audit["assignment_sha256"] = hashlib.sha256(
        ",".join(map(str, assignments)).encode()
    ).hexdigest()
    dump_json(args.output_dir / "split_audit.json", split_audit)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    device = torch.device("cpu" if args.cpu else "cuda")
    for fold in requested_folds:
        fold_dir = args.output_dir / f"fold_{fold}"
        prediction_path = fold_dir / "predictions.jsonl"
        if prediction_path.exists():
            print(f"fold={fold} already complete; skipping", flush=True)
            continue
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_seed = args.seed + fold
        random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        train_indices = [index for index, value in enumerate(assignments) if value != fold]
        held_indices = [index for index, value in enumerate(assignments) if value == fold]
        train_frame = frame.iloc[train_indices]
        held_frame = frame.iloc[held_indices]
        train_examples, dropped = make_examples(
            train_frame, tokenizer, drop_fully_unaligned=True,
        )
        held_examples, _ = make_examples(held_frame, tokenizer)
        print(
            f"fold={fold} seed={fold_seed} train_rows={len(train_examples)} "
            f"held_rows={len(held_examples)} dropped={dropped}", flush=True,
        )

        model = MultiHeadNER(args.model).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.01,
        )
        generator = torch.Generator().manual_seed(fold_seed)
        loader = DataLoader(
            NERDataset(train_examples), batch_size=args.batch_size, shuffle=True,
            collate_fn=collate, generator=generator,
        )
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext()
                )
                with context:
                    _, loss = model(**batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.detach().cpu())
            print(
                f"fold={fold} epoch={epoch} mean_loss={total_loss / len(loader):.8f}",
                flush=True,
            )

        model.eval()
        logits = []
        with torch.no_grad():
            for start in range(0, len(held_examples), args.batch_size):
                stop = min(start + args.batch_size, len(held_examples))
                rows = [NERDataset(held_examples)[index] for index in range(start, stop)]
                batch = collate(rows)
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext()
                )
                with context:
                    values, _ = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                    )
                logits.extend(values.float().cpu())
        predictions = decode_entities(logits, held_examples)
        records = []
        fold_scores = []
        for row_index, entities in zip(held_indices, predictions):
            predicted_set = {(item["entity"], item["type"]) for item in entities}
            fold_scores.append(score(entity_rows[row_index], list(predicted_set))[2])
            records.append({
                "row_index": int(row_index), "fold": fold, "entities": entities,
                "answer": "\n".join(
                    f"{entity}, {kind}" for entity, kind in sorted(predicted_set)
                ),
            })
        dump_jsonl(prediction_path, records)
        dump_json(fold_dir / "metrics.json", {
            "fold": fold,
            "seed": fold_seed,
            "train_rows": len(train_examples),
            "held_rows": len(held_examples),
            "dropped_fully_unaligned_train_rows": dropped,
            "held_macro_entity_f1": sum(fold_scores) / len(fold_scores),
        })
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    completed, completed_folds = [], []
    for fold in range(args.n_splits):
        path = args.output_dir / f"fold_{fold}" / "predictions.jsonl"
        if path.exists():
            completed_folds.append(fold)
            completed.extend(
                json.loads(line) for line in path.read_text().splitlines() if line.strip()
            )
    indices = [int(record["row_index"]) for record in completed]
    if len(indices) != len(set(indices)):
        raise AssertionError("duplicate OOF predictions")
    oof_score = None
    if len(completed_folds) == args.n_splits:
        if sorted(indices) != list(range(len(frame))):
            raise AssertionError("fold predictions do not cover all rows exactly once")
        completed.sort(key=lambda record: int(record["row_index"]))
        dump_jsonl(args.output_dir / "oof_predictions.jsonl", completed)
        row_scores = []
        for record in completed:
            predicted = {
                (item["entity"], item["type"]) for item in record["entities"]
            }
            row_scores.append(
                score(entity_rows[int(record["row_index"])], list(predicted))[2]
            )
        oof_score = sum(row_scores) / len(row_scores)
    manifest = config | {
        "completed_folds": completed_folds,
        "complete": len(completed_folds) == args.n_splits,
        "rows_predicted": len(completed),
        "oof_macro_entity_f1": oof_score,
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
