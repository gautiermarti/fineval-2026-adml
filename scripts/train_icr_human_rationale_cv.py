#!/usr/bin/env python3
"""Source-disjoint JF-ICR training with exact human-rationale supervision."""

from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.icr_rationale_multitask import rationale_token_labels  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402
from train_icr_japanese_cv import (  # noqa: E402
    LABELS, LABEL_TO_ID, build_oof_output, encode_pair,
)


DEFAULT_MODEL = "cl-tohoku/bert-base-japanese-v3"
DEFAULT_REVISION = "65243d6e5629b969c77309f217bd7b1a79d43c7e"
RATIONALE_PATH = ROOT / "data/external/ebisu_human_rationales/rationales.jsonl"
RATIONALE_SUMMARY = ROOT / "data/external/ebisu_human_rationales/summary.json"
EXPECTED_RATIONALE_SHA256 = (
    "b0023a2e3fa7b6ec505ca19667ec831105a138107fa1b1a8948a2f153caac452"
)


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_rationale_map() -> tuple[dict[int, list[dict]], dict]:
    if file_digest(RATIONALE_PATH) != EXPECTED_RATIONALE_SHA256:
        raise ValueError("curated rationale JSONL differs from preregistered input")
    summary = json.loads(RATIONALE_SUMMARY.read_text())
    expected = {
        "raw_tasks": 95,
        "matched_unique_benchmark_rows": 94,
        "rows_with_exact_gold_aligned_rationale": 75,
        "exact_rationale_spans": 200,
        "source_folds_covered": [0, 1, 3, 4],
        "current_or_hidden_test_used": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"rationale audit changed: {key}={summary.get(key)!r}, expected {value!r}"
            )
    rows = [
        json.loads(line) for line in RATIONALE_PATH.read_text().splitlines()
        if line.strip()
    ]
    if len({int(row["benchmark_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate benchmark IDs in rationale corpus")
    return {
        int(row["benchmark_id"]): list(row["rationale_spans"])
        for row in rows if row["rationale_spans"]
    }, summary


class RationaleDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        max_length: int,
        rationale_map: dict[int, list[dict]] | None = None,
    ):
        rationale_map = rationale_map or {}
        self.features = []
        self.audit = {
            "rows": len(frame),
            "rationale_rows_available": 0,
            "rationale_rows_supervised": 0,
            "rationale_spans_attempted": 0,
            "rationale_spans_recovered": 0,
            "positive_rationale_tokens": 0,
            "negative_response_tokens": 0,
        }
        if tokenizer.sep_token_id is None:
            raise ValueError("tokenizer has no separator token")
        for row in frame.itertuples(index=False):
            parsed = parse_icr_query(row.query)
            encoded = encode_pair(
                tokenizer, parsed.question, parsed.response, max_length,
            )
            spans = rationale_map.get(int(row.id), [])
            span_ids = [
                tokenizer(span["text"], add_special_tokens=False)["input_ids"]
                for span in spans
            ]
            token_labels, recovered = rationale_token_labels(
                encoded["input_ids"], span_ids,
                sep_token_id=tokenizer.sep_token_id,
                unk_token_id=tokenizer.unk_token_id,
            )
            if spans:
                self.audit["rationale_rows_available"] += 1
                self.audit["rationale_spans_attempted"] += len(spans)
            if recovered:
                self.audit["rationale_rows_supervised"] += 1
                self.audit["rationale_spans_recovered"] += recovered
                self.audit["positive_rationale_tokens"] += token_labels.count(1)
                self.audit["negative_response_tokens"] += token_labels.count(0)
            self.features.append(encoded | {
                "labels": LABEL_TO_ID[str(row.answer)],
                "rationale_labels": token_labels,
            })

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index]


def collate(batch: list[dict], tokenizer) -> dict[str, torch.Tensor]:
    labels = torch.tensor([row["labels"] for row in batch], dtype=torch.long)
    rationale = [row["rationale_labels"] for row in batch]
    features = [
        {
            key: value for key, value in row.items()
            if key not in {"labels", "rationale_labels"}
        }
        for row in batch
    ]
    encoded = tokenizer.pad(features, return_tensors="pt")
    token_labels = torch.full(
        encoded["input_ids"].shape, -100, dtype=torch.long,
    )
    for index, values in enumerate(rationale):
        token_labels[index, :len(values)] = torch.tensor(values, dtype=torch.long)
    encoded["labels"] = labels
    encoded["rationale_labels"] = token_labels
    return encoded


class RationaleClassifier(nn.Module):
    def __init__(self, model: str, revision: str | None):
        super().__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            model,
            revision=revision,
            num_labels=len(LABELS),
            ignore_mismatched_sizes=True,
        )
        hidden = int(self.classifier.config.hidden_size)
        probability = float(getattr(
            self.classifier.config, "hidden_dropout_prob", 0.1,
        ))
        self.rationale_dropout = nn.Dropout(probability)
        self.rationale_head = nn.Linear(hidden, 2)

    def forward(self, inputs: dict[str, torch.Tensor], rationale: bool):
        outputs = self.classifier(
            **inputs, output_hidden_states=rationale, return_dict=True,
        )
        token_logits = None
        if rationale:
            token_logits = self.rationale_head(
                self.rationale_dropout(outputs.hidden_states[-1])
            )
        return outputs.logits, token_logits


def set_fold_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rationale-weight", type=float, default=0.10)
    parser.add_argument("--rationale-positive-weight", type=float, default=3.0)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-train-steps", type=int,
        help="Per-fold smoke cap only; omit for performance experiments.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task2_human_rationale_cv/seed7",
    )
    args = parser.parse_args()
    if args.rationale_weight < 0 or args.rationale_positive_weight <= 0:
        raise ValueError("rationale loss weights must be non-negative/positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.bf16 and device.type != "cuda":
        raise ValueError("--bf16 requires CUDA")

    rationale_map, rationale_summary = load_rationale_map()
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    source = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    frame["fold"] = source.loc[frame.id].source_fold.to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    available_folds = sorted(int(value) for value in frame.fold.unique())
    fold_values = available_folds if args.folds is None else args.folds
    if len(set(fold_values)) != len(fold_values) or not set(fold_values) <= set(available_folds):
        raise ValueError("--folds contains duplicates or unavailable source folds")

    all_logits = np.full((len(frame), len(LABELS)), np.nan, dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_audits = []
    for fold in fold_values:
        fold_seed = args.seed + 1000 * int(fold)
        set_fold_seed(fold_seed)
        train_frame = frame[frame.fold != fold]
        valid_frame = frame[frame.fold == fold]
        train_rationale_map = {
            int(row_id): rationale_map[int(row_id)]
            for row_id in train_frame.id if int(row_id) in rationale_map
        }
        train = RationaleDataset(
            train_frame, tokenizer, args.max_length, train_rationale_map,
        )
        # Empty by construction: held-out annotations cannot enter evaluation
        # preprocessing or any training loss.
        valid = RationaleDataset(valid_frame, tokenizer, args.max_length, {})
        if valid.audit["rationale_rows_available"] != 0:
            raise AssertionError("validation rationale leakage")
        generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(
            train, batch_size=args.batch_size, shuffle=True, generator=generator,
            collate_fn=lambda rows: collate(rows, tokenizer),
        )
        model = RationaleClassifier(args.model, args.revision).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.01,
        )
        steps = 0
        component_loss_audit = None
        for _ in range(args.epochs):
            model.train()
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch.pop("labels")
                rationale_labels = batch.pop("rationale_labels")
                has_rationale = bool(
                    args.rationale_weight > 0 and (rationale_labels != -100).any()
                )
                optimizer.zero_grad()
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    logits, token_logits = model(batch, has_rationale)
                    classification_loss = F.cross_entropy(logits, labels)
                    if has_rationale:
                        weights = logits.new_tensor([
                            1.0, args.rationale_positive_weight,
                        ])
                        rationale_loss = F.cross_entropy(
                            token_logits.transpose(1, 2), rationale_labels,
                            weight=weights, ignore_index=-100,
                        )
                    else:
                        rationale_loss = logits.sum() * 0.0
                    loss = classification_loss + args.rationale_weight * rationale_loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite training loss")
                loss.backward()
                optimizer.step()
                if component_loss_audit is None and has_rationale:
                    component_loss_audit = {
                        "classification": float(classification_loss.detach().cpu()),
                        "rationale": float(rationale_loss.detach().cpu()),
                        "weighted_total": float(loss.detach().cpu()),
                    }
                steps += 1
                if args.max_train_steps is not None and steps >= args.max_train_steps:
                    break
            if args.max_train_steps is not None and steps >= args.max_train_steps:
                break

        model.eval()
        fold_logits = []
        loader = DataLoader(
            valid, batch_size=args.batch_size,
            collate_fn=lambda rows: collate(rows, tokenizer),
        )
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels")
                batch.pop("rationale_labels")
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    logits, _ = model(batch, False)
                fold_logits.append(logits.float().cpu().numpy())
        fold_logits = np.concatenate(fold_logits)
        all_logits[valid_frame.index] = fold_logits
        gold = np.asarray([LABEL_TO_ID[str(value)] for value in valid_frame.answer])
        raw_accuracy = float(np.mean(fold_logits.argmax(axis=1) == gold))
        audit = {
            "fold": int(fold),
            "fold_seed": fold_seed,
            "train_rows": len(train_frame),
            "validation_rows": len(valid_frame),
            "steps": steps,
            "raw_argmax_accuracy": raw_accuracy,
            "train_rationale_audit": train.audit,
            "validation_rationale_audit": valid.audit,
            "component_loss_audit": component_loss_audit,
        }
        fold_audits.append(audit)
        print(json.dumps(audit, ensure_ascii=False), flush=True)
        fold_dir = args.output_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.classifier.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)
        torch.save(model.rationale_head.state_dict(), fold_dir / "rationale_head.pt")
        build_oof_output(frame, all_logits).to_csv(
            args.output_dir / "oof_predictions.partial.csv", index=False,
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = build_oof_output(frame, all_logits)
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    evaluated = ~np.isnan(all_logits).any(axis=1)
    all_gold = np.asarray([LABEL_TO_ID[str(value)] for value in frame.answer])
    raw_accuracy = float(np.mean(
        all_logits[evaluated].argmax(axis=1) == all_gold[evaluated]
    ))
    metrics = {
        "experiment": "human_rationale_token_multitask",
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "rationale_weight": args.rationale_weight,
        "rationale_positive_weight": args.rationale_positive_weight,
        "bf16": args.bf16,
        "device": str(device),
        "folds_completed": fold_values,
        "rows_evaluated": int(evaluated.sum()),
        "raw_argmax_accuracy": raw_accuracy,
        "max_train_steps": args.max_train_steps,
        "fold_audits": fold_audits,
        "rationale_summary": rationale_summary,
        "input_sha256": {
            "jf_icr_public": file_digest(
                ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
            ),
            "source_folds": file_digest(
                ROOT / "data/derived/icr_source_folds.csv"
            ),
            "rationales": file_digest(RATIONALE_PATH),
            "rationale_summary": file_digest(RATIONALE_SUMMARY),
        },
        "leakage_guards": {
            "held_out_rationales_loaded_into_validation_dataset": False,
            "held_out_labels_used_for_training": False,
            "current_or_hidden_test_used": False,
        },
        "submission_policy_status": rationale_summary["competition_use_status"],
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
