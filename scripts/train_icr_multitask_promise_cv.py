#!/usr/bin/env python3
"""Source-disjoint JF-ICR training with ordinal and external promise objectives."""

from __future__ import annotations

from argparse import ArgumentParser
from itertools import cycle
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.icr_multitask import (  # noqa: E402
    ICRPromiseMultiTaskModel,
    normalized_span_bounds,
    ordinal_pairwise_loss,
    window_span_tokens,
)
from fineval.parsing import parse_icr_query  # noqa: E402
from train_icr_japanese_cv import (  # noqa: E402
    LABELS, LABEL_TO_ID, build_oof_output, encode_pair,
)


DEFAULT_MODEL = "cl-tohoku/bert-base-japanese-v3"
DEFAULT_REVISION = "65243d6e5629b969c77309f217bd7b1a79d43c7e"
SAFE_AUXILIARY = ROOT / "data/external/ml_promise_japanese_safe/auxiliary_train.jsonl"
SAFE_SUMMARY = ROOT / "data/external/ml_promise_japanese_safe/summary.json"


class ICRDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.features = []
        for row in frame.itertuples(index=False):
            parsed = parse_icr_query(row.query)
            encoded = encode_pair(
                tokenizer, parsed.question, parsed.response, max_length,
            )
            self.features.append(encoded | {
                "labels": LABEL_TO_ID[str(row.answer)],
            })

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index]


def encode_promise_example(tokenizer, row, max_length: int) -> dict:
    positive = str(row.promise_status) == "Yes"
    text = str(row.text)
    span = str(row.promise_span) if positive else ""
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if positive:
        bounds = normalized_span_bounds(text, span)
        if bounds is None:
            raise ValueError(f"ungrounded promise span in {row.auxiliary_id}")
        start, end = bounds
        prefix_ids = tokenizer(
            text[:start], add_special_tokens=False,
        )["input_ids"]
        span_ids = tokenizer(text[start:end], add_special_tokens=False)["input_ids"]
        suffix_ids = tokenizer(
            text[end:], add_special_tokens=False,
        )["input_ids"]
        if len(span_ids) > budget:
            # BERT cannot represent this complete annotation. Retain its valid
            # paragraph label, but do not turn a truncated span into a false
            # "exact" token target.
            input_ids = span_ids[:budget]
            raw_token_labels = [-100] * len(input_ids)
            span_supervised = False
        else:
            input_ids, raw_token_labels = window_span_tokens(
                prefix_ids, span_ids, suffix_ids, budget,
            )
            span_supervised = True
    else:
        input_ids = tokenizer(
            text, add_special_tokens=False,
        )["input_ids"][:budget]
        raw_token_labels = [0] * len(input_ids)
        span_supervised = True
    encoded = tokenizer.prepare_for_model(
        input_ids,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )
    raw_iterator = iter(raw_token_labels)
    encoded["token_labels"] = [
        -100 if special else next(raw_iterator)
        for special in encoded.pop("special_tokens_mask")
    ]
    encoded["promise_labels"] = int(positive)
    encoded["span_supervised"] = span_supervised
    return encoded


class PromiseDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.features = [
            encode_promise_example(tokenizer, row, max_length)
            for row in frame.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index]


def collate_icr(batch: list[dict], tokenizer) -> dict[str, torch.Tensor]:
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    features = [
        {key: value for key, value in item.items() if key != "labels"}
        for item in batch
    ]
    encoded = tokenizer.pad(features, return_tensors="pt")
    encoded["labels"] = labels
    return encoded


def collate_promise(batch: list[dict], tokenizer) -> dict[str, torch.Tensor]:
    promise_labels = torch.tensor([
        item["promise_labels"] for item in batch
    ], dtype=torch.long)
    lengths = [len(item["input_ids"]) for item in batch]
    features = [
        {
            key: value for key, value in item.items()
            if key not in {"promise_labels", "token_labels", "span_supervised"}
        }
        for item in batch
    ]
    encoded = tokenizer.pad(features, return_tensors="pt")
    width = encoded["input_ids"].shape[1]
    token_labels = torch.full(
        (len(batch), width), -100, dtype=torch.long,
    )
    for index, (item, length) in enumerate(zip(batch, lengths, strict=True)):
        token_labels[index, :length] = torch.tensor(item["token_labels"])
    encoded["promise_labels"] = promise_labels
    encoded["token_labels"] = token_labels
    return encoded


def model_inputs(batch: dict[str, torch.Tensor], *label_keys: str) -> dict:
    excluded = set(label_keys)
    return {key: value for key, value in batch.items() if key not in excluded}


def validate_auxiliary_release() -> dict:
    summary = json.loads(SAFE_SUMMARY.read_text())
    required = {
        "accepted_rows": 357,
        "positive_spans_grounded": True,
        "jf_icr_labels_used": False,
        "ml_promise_test_used": False,
        "current_or_hidden_test_used": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"unsafe auxiliary summary: {key}={summary.get(key)!r}, "
                f"expected {expected!r}"
            )
    return summary


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def gradient_norm(module: torch.nn.Module) -> float:
    return float(sum(
        parameter.grad.detach().float().norm().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    ))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aux-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rank-weight", type=float, default=0.20)
    parser.add_argument("--promise-weight", type=float, default=0.25)
    parser.add_argument("--span-weight", type=float, default=0.10)
    parser.add_argument("--margin-per-level", type=float, default=0.25)
    parser.add_argument("--span-positive-weight", type=float, default=3.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-folds", type=int)
    parser.add_argument(
        "--max-train-steps", type=int,
        help="Per-fold smoke-test cap; omit for a real experiment.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task2_multitask_promise_cv/seed7",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if args.bf16 and device.type != "cuda":
        raise ValueError("--bf16 requires CUDA")

    auxiliary_summary = validate_auxiliary_release()
    auxiliary_frame = pd.read_json(SAFE_AUXILIARY, lines=True)
    if len(auxiliary_frame) != auxiliary_summary["accepted_rows"]:
        raise ValueError("auxiliary row count differs from its frozen audit")
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    folds = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    frame["fold"] = folds.loc[frame.id].source_fold.to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision,
    )
    auxiliary = PromiseDataset(
        auxiliary_frame, tokenizer, args.max_length,
    )
    positive_span_tokens = sum(
        sum(value == 1 for value in item["token_labels"])
        for item in auxiliary.features
    )
    if positive_span_tokens <= 0:
        raise ValueError("no positive auxiliary span tokens were encoded")
    supervised_positive_spans = sum(
        item["promise_labels"] == 1 and item["span_supervised"]
        for item in auxiliary.features
    )

    fold_values = sorted(frame.fold.unique())
    if args.max_folds is not None:
        if not 1 <= args.max_folds <= len(fold_values):
            raise ValueError("--max-folds is outside the available fold range")
        fold_values = fold_values[:args.max_folds]
    all_logits = np.full((len(frame), len(LABELS)), np.nan, dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_step_audit = None

    for fold in fold_values:
        train_frame = frame[frame.fold != fold]
        valid_frame = frame[frame.fold == fold]
        train = ICRDataset(train_frame, tokenizer, args.max_length)
        valid = ICRDataset(valid_frame, tokenizer, args.max_length)
        generator = torch.Generator().manual_seed(args.seed + int(fold))
        train_loader = DataLoader(
            train, batch_size=args.batch_size, shuffle=True, generator=generator,
            collate_fn=lambda rows: collate_icr(rows, tokenizer),
        )
        auxiliary_loader = DataLoader(
            auxiliary, batch_size=args.aux_batch_size, shuffle=True,
            generator=generator,
            collate_fn=lambda rows: collate_promise(rows, tokenizer),
        )
        model = ICRPromiseMultiTaskModel(
            args.model, revision=args.revision,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.01,
        )
        completed_steps = 0
        for _ in range(args.epochs):
            model.train()
            for icr_batch, promise_batch in zip(
                train_loader, cycle(auxiliary_loader), strict=False,
            ):
                icr_batch = {key: value.to(device) for key, value in icr_batch.items()}
                promise_batch = {
                    key: value.to(device) for key, value in promise_batch.items()
                }
                optimizer.zero_grad()
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    icr_logits = model.icr_logits(**model_inputs(icr_batch, "labels"))
                    icr_loss = F.cross_entropy(icr_logits, icr_batch["labels"])
                    rank_loss = ordinal_pairwise_loss(
                        icr_logits, icr_batch["labels"],
                        margin_per_level=args.margin_per_level,
                    )
                    promise_logits, span_logits = model.promise_logits(
                        **model_inputs(
                            promise_batch, "promise_labels", "token_labels",
                        )
                    )
                    promise_loss = F.cross_entropy(
                        promise_logits, promise_batch["promise_labels"],
                    )
                    span_weights = span_logits.new_tensor([
                        1.0, args.span_positive_weight,
                    ])
                    span_loss = F.cross_entropy(
                        span_logits.transpose(1, 2), promise_batch["token_labels"],
                        weight=span_weights, ignore_index=-100,
                    )
                    loss = (
                        icr_loss
                        + args.rank_weight * rank_loss
                        + args.promise_weight * promise_loss
                        + args.span_weight * span_loss
                    )
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite multi-task loss")
                loss.backward()
                if first_step_audit is None:
                    first_step_audit = {
                        "component_losses": {
                            "icr": float(icr_loss.detach().cpu()),
                            "ordinal_rank": float(rank_loss.detach().cpu()),
                            "promise": float(promise_loss.detach().cpu()),
                            "span": float(span_loss.detach().cpu()),
                            "weighted_total": float(loss.detach().cpu()),
                        },
                        "gradient_norms": {
                            "encoder": gradient_norm(model.encoder),
                            "icr_head": gradient_norm(model.icr_head),
                            "promise_head": gradient_norm(model.promise_head),
                            "span_head": gradient_norm(model.span_head),
                        },
                    }
                    if any(
                        value <= 0.0 or not np.isfinite(value)
                        for value in first_step_audit["gradient_norms"].values()
                    ):
                        raise ValueError(
                            f"missing/non-finite first-step gradient: {first_step_audit}"
                        )
                optimizer.step()
                completed_steps += 1
                if (
                    args.max_train_steps is not None
                    and completed_steps >= args.max_train_steps
                ):
                    break
            if (
                args.max_train_steps is not None
                and completed_steps >= args.max_train_steps
            ):
                break

        model.eval()
        fold_logits = []
        with torch.no_grad():
            loader = DataLoader(
                valid, batch_size=args.batch_size,
                collate_fn=lambda rows: collate_icr(rows, tokenizer),
            )
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    logits = model.icr_logits(**model_inputs(batch, "labels"))
                fold_logits.append(logits.float().cpu().numpy())
        fold_logits = np.concatenate(fold_logits)
        all_logits[valid_frame.index] = fold_logits
        raw_accuracy = float(np.mean(
            fold_logits.argmax(axis=1)
            == np.asarray([LABEL_TO_ID[str(value)] for value in valid_frame.answer])
        ))
        print(
            f"fold={fold} rows={len(valid_frame)} steps={completed_steps} "
            f"raw_accuracy={raw_accuracy:.6f}",
            flush=True,
        )
        fold_dir = args.output_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)
        torch.save({
            "icr_head": model.icr_head.state_dict(),
            "promise_head": model.promise_head.state_dict(),
            "span_head": model.span_head.state_dict(),
        }, fold_dir / "multitask_heads.pt")
        build_oof_output(frame, all_logits).to_csv(
            args.output_dir / "oof_predictions.partial.csv", index=False,
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = build_oof_output(frame, all_logits)
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    evaluated = ~np.isnan(all_logits).any(axis=1)
    gold = np.asarray([LABEL_TO_ID[str(value)] for value in frame.answer])
    raw_accuracy = float(np.mean(
        all_logits[evaluated].argmax(axis=1) == gold[evaluated]
    ))
    metrics = {
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "folds_completed": len(fold_values),
        "rows_evaluated": int(evaluated.sum()),
        "raw_accuracy": raw_accuracy,
        "auxiliary_rows": len(auxiliary),
        "auxiliary_hosts": int(auxiliary_frame.source_host.nunique()),
        "positive_span_tokens": positive_span_tokens,
        "supervised_positive_spans": supervised_positive_spans,
        "positive_spans_exceeding_model_window": int(
            (auxiliary_frame.promise_status == "Yes").sum()
            - supervised_positive_spans
        ),
        "loss_weights": {
            "icr": 1.0,
            "ordinal_rank": args.rank_weight,
            "promise": args.promise_weight,
            "span": args.span_weight,
            "span_positive": args.span_positive_weight,
            "margin_per_level": args.margin_per_level,
        },
        "max_train_steps": args.max_train_steps,
        "first_step_audit": first_step_audit,
        "input_sha256": {
            "jf_icr_public": file_digest(
                ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
            ),
            "source_folds": file_digest(
                ROOT / "data/derived/icr_source_folds.csv"
            ),
            "auxiliary_train": file_digest(SAFE_AUXILIARY),
            "auxiliary_summary": file_digest(SAFE_SUMMARY),
        },
        "leakage_guards": {
            "benchmark_labels_used_for_auxiliary_curation": False,
            "ml_promise_test_used": False,
            "current_or_hidden_test_used": False,
            "overlapping_issuers_excluded": True,
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
