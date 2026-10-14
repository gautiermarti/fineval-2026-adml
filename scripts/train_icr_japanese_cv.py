#!/usr/bin/env python3
"""Source-disjoint CV for a Japanese JF-ICR sequence classifier."""

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

from fineval.parsing import parse_icr_query  # noqa: E402


DEFAULT_MODEL = "cl-tohoku/bert-base-japanese-v3"
LABELS = ["-2", "-1", "0", "+1", "+2"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def build_oof_output(frame: pd.DataFrame, all_logits: np.ndarray) -> pd.DataFrame:
    """Build an aligned OOF table from every completed fold."""
    evaluated = ~np.isnan(all_logits).any(axis=1)
    predictions = all_logits[evaluated].argmax(axis=1)
    return pd.DataFrame({
        "id": frame.loc[evaluated, "id"],
        "gold": frame.loc[evaluated, "answer"],
        "prediction": [LABELS[index] for index in predictions],
        "fold": frame.loc[evaluated, "fold"],
        **{
            f"logit_{label}": all_logits[evaluated, index]
            for index, label in enumerate(LABELS)
        },
    })


def encode_pair(
    tokenizer, question: str, response: str, max_length: int,
    truncation_profile: str = "longest_first",
    prepend_cls: bool = False,
) -> dict:
    """Encode an ICR pair, optionally retaining both ends of a long response."""
    # rinna/japanese-roberta-base was pretrained with a literal [CLS] prefix,
    # but its T5 tokenizer does not insert that token via add_special_tokens.
    if prepend_cls:
        question = tokenizer.cls_token + question
    if truncation_profile == "longest_first":
        return tokenizer(question, response, truncation=True, max_length=max_length)
    if truncation_profile != "response_head_tail":
        raise ValueError(f"unknown truncation profile: {truncation_profile}")
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=True)
    if budget <= 1:
        raise ValueError("max_length is too small for a sequence pair")
    if len(question_ids) > budget - 1:
        left = (budget - 1 + 1) // 2
        question_ids = question_ids[:left] + question_ids[-(budget - 1 - left):]
    response_budget = budget - len(question_ids)
    if len(response_ids) > response_budget:
        head = (response_budget + 1) // 2
        tail = response_budget - head
        response_ids = response_ids[:head] + (response_ids[-tail:] if tail else [])
    return tokenizer.prepare_for_model(
        question_ids, response_ids, add_special_tokens=True,
        truncation=False, return_attention_mask=True,
    )


class ICRDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, tokenizer, max_length: int,
        truncation_profile: str = "longest_first",
        prepend_cls: bool = False,
    ):
        self.labels = [LABEL_TO_ID[str(value)] for value in frame.answer]
        self.features = []
        for query in frame["query"]:
            parsed = parse_icr_query(query)
            self.features.append(encode_pair(
                tokenizer, parsed.question, parsed.response, max_length,
                truncation_profile, prepend_cls,
            ))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index] | {"labels": self.labels[index]}


def collate(
    batch: list[dict], tokenizer, *, zero_based_position_ids: bool = False,
) -> dict[str, torch.Tensor]:
    labels = torch.tensor([item["labels"] for item in batch])
    features = [
        {key: value for key, value in item.items() if key != "labels"}
        for item in batch
    ]
    encoded = tokenizer.pad(features, return_tensors="pt")
    if zero_based_position_ids:
        # Rinna's model card requires positions to start at zero. Transformers'
        # RoBERTa default starts after padding_idx, which is incompatible here.
        encoded["position_ids"] = torch.arange(
            encoded["input_ids"].shape[1], dtype=torch.long,
        ).unsqueeze(0).expand_as(encoded["input_ids"])
    encoded["labels"] = labels
    return encoded


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        help="Optional immutable Hugging Face model revision/commit.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--truncation-profile",
        choices=("longest_first", "response_head_tail"),
        default="longest_first",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--bf16", action="store_true",
        help="Use CUDA bfloat16 autocast while retaining FP32 optimizer state.",
    )
    parser.add_argument(
        "--max-folds", type=int,
        help="Run only the first N source folds for a predeclared futility pilot.",
    )
    parser.add_argument(
        "--rinna-roberta-compat", action="store_true",
        help="Use the tokenizer/position convention required by rinna Japanese RoBERTa.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/task2_japanese_cv/seed7",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16 and device.type != "cuda":
        raise ValueError("--bf16 requires a CUDA device")
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    folds = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    frame["fold"] = folds.loc[frame.id].source_fold.to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=False if args.rinna_roberta_compat else None,
    )
    if args.rinna_roberta_compat:
        tokenizer.do_lower_case = True
    all_logits = np.full((len(frame), len(LABELS)), np.nan, dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_values = sorted(frame["fold"].unique())
    if args.max_folds is not None:
        if not 1 <= args.max_folds <= len(fold_values):
            raise ValueError("--max-folds must be between 1 and the number of folds")
        fold_values = fold_values[:args.max_folds]
    for fold in fold_values:
        train_frame = frame[frame.fold != fold]
        valid_frame = frame[frame.fold == fold]
        train = ICRDataset(
            train_frame, tokenizer, args.max_length, args.truncation_profile,
            args.rinna_roberta_compat,
        )
        valid = ICRDataset(
            valid_frame, tokenizer, args.max_length, args.truncation_profile,
            args.rinna_roberta_compat,
        )
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
            collate_fn=lambda items: collate(
                items, tokenizer,
                zero_based_position_ids=args.rinna_roberta_compat,
            ),
        )
        for _ in range(args.epochs):
            model.train()
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        fold_logits = []
        with torch.no_grad():
            valid_loader = DataLoader(
                valid,
                batch_size=args.batch_size,
                collate_fn=lambda items: collate(
                    items, tokenizer,
                    zero_based_position_ids=args.rinna_roberta_compat,
                ),
            )
            for batch in valid_loader:
                batch.pop("labels")
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    logits = model(**batch).logits
                fold_logits.append(logits.float().cpu().numpy())
        fold_logits = np.concatenate(fold_logits)
        all_logits[valid_frame.index] = fold_logits
        predictions = fold_logits.argmax(axis=1)
        accuracy = np.mean(predictions == np.asarray(valid.labels))
        print(f"fold={fold} n={len(valid)} accuracy={accuracy:.6f}", flush=True)
        # Preserve expensive completed folds if a later fold is interrupted.
        # The evaluator still consumes only the final oof_predictions.csv.
        build_oof_output(frame, all_logits).to_csv(
            args.output_dir / "oof_predictions.partial.csv", index=False,
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    evaluated = ~np.isnan(all_logits).any(axis=1)
    predictions = all_logits[evaluated].argmax(axis=1)
    gold = np.asarray([LABEL_TO_ID[str(value)] for value in frame.answer])
    accuracy = float(np.mean(predictions == gold[evaluated]))
    metric_name = "source_cv_accuracy" if evaluated.all() else "partial_source_cv_accuracy"
    print(f"{metric_name}={accuracy:.6f}")
    output = build_oof_output(frame, all_logits)
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    (args.output_dir / "metrics.txt").write_text(
        f"model={args.model}\nrevision={args.revision or 'default'}\n"
        f"seed={args.seed}\nepochs={args.epochs}\n"
        f"bf16={args.bf16}\n"
        f"truncation_profile={args.truncation_profile}\n"
        f"rinna_roberta_compat={args.rinna_roberta_compat}\n"
        f"folds_completed={len(fold_values)}\n"
        f"rows_evaluated={int(evaluated.sum())}\n"
        f"{metric_name}={accuracy:.12f}\n"
    )


if __name__ == "__main__":
    main()
