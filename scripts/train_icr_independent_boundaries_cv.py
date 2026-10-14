#!/usr/bin/env python3
"""Source-disjoint JF-ICR training with four independent boundary heads.

This is the neural successor to the CPU boundary diagnostic.  It deliberately
retains the complete question/response and does not concatenate retrieved
anchors, which were harmful in the frozen CPU ablation.  Unlike the rejected
monotonic cumulative pilot, the four decisions do not share a scalar score or
learned cutpoints; monotonicity is imposed only during decoding.
"""

from __future__ import annotations

import argparse
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
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.icr_anchors import (  # noqa: E402
    LABELS, balanced_boundary_pos_weights, cumulative_targets,
    decode_cumulative_probabilities,
)
from fineval.parsing import parse_icr_query  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from train_icr_japanese_cv import DEFAULT_MODEL, encode_pair  # noqa: E402


def source_equal_weights(groups: pd.Series) -> np.ndarray:
    counts = groups.value_counts()
    result = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    return result / result.mean()


class BoundaryDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, tokenizer, max_length: int,
        truncation_profile: str,
    ):
        indices = np.asarray([LABELS.index(str(label)) for label in frame.answer])
        self.targets = cumulative_targets(indices).astype(np.float32)
        self.weights = frame.source_weight.to_numpy(dtype=np.float32)
        self.features = []
        for query in frame["query"]:
            item = parse_icr_query(query)
            self.features.append(encode_pair(
                tokenizer, item.question, item.response, max_length,
                truncation_profile=truncation_profile,
            ))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index] | {
            "targets": self.targets[index],
            "sample_weight": self.weights[index],
        }


class BoundaryCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, records: list[dict]) -> dict[str, torch.Tensor]:
        features = [{
            key: value for key, value in record.items()
            if key not in {"targets", "sample_weight"}
        } for record in records]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["targets"] = torch.tensor(np.stack([
            record["targets"] for record in records
        ]), dtype=torch.float32)
        batch["sample_weight"] = torch.tensor([
            record["sample_weight"] for record in records
        ], dtype=torch.float32)
        return batch


class IndependentBoundaryClassifier(nn.Module):
    def __init__(self, model_name: str, revision: str | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, revision=revision)
        hidden = self.encoder.config.hidden_size
        dropout = getattr(self.encoder.config, "classifier_dropout", None)
        if dropout is None:
            dropout = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(float(dropout))
        self.boundaries = nn.Linear(hidden, 4)

    def forward(self, **features) -> torch.Tensor:
        encoded = self.encoder(**features)
        pooled = getattr(encoded, "pooler_output", None)
        if pooled is None:
            pooled = encoded.last_hidden_state[:, 0]
        return self.boundaries(self.dropout(pooled))


def evaluate_predictions(
    frame: pd.DataFrame, logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    raw = decode_cumulative_probabilities(probabilities)
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])
    ruled = np.where(rules != "+1", rules, raw)
    gold = frame.answer.astype(str).to_numpy()
    return raw, ruled, float(np.mean(raw == gold)), float(np.mean(ruled == gold))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--truncation-profile",
        choices=("longest_first", "response_head_tail"), default="longest_first",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-folds", type=int,
        help="Run the first N folds for the predeclared futility sequence.",
    )
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_independent_boundaries_cv/seed7",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.bf16 and device.type != "cuda":
        raise ValueError("--bf16 requires CUDA")

    frame = pd.read_parquet(
        ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    ).sort_values("id").reset_index(drop=True)
    metadata = (
        pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")
        .set_index("id").loc[frame.id].reset_index(drop=True)
    )
    frame["fold"] = metadata.source_fold.astype(int)
    frame["source_group"] = metadata.source_group.astype(str)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    all_logits = np.full((len(frame), 4), np.nan, dtype=np.float32)
    folds = sorted(frame.fold.unique())
    if args.max_folds is not None:
        if not 1 <= args.max_folds <= len(folds):
            raise ValueError("--max-folds must be in [1, number of folds]")
        folds = folds[:args.max_folds]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold in folds:
        fold_seed = args.seed * 100 + int(fold)
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        train_frame = frame[frame.fold != fold].copy()
        valid_frame = frame[frame.fold == fold].copy()
        train_frame["source_weight"] = source_equal_weights(train_frame.source_group)
        valid_frame["source_weight"] = 1.0
        train_targets = cumulative_targets(np.asarray([
            LABELS.index(str(label)) for label in train_frame.answer
        ]))
        positive_weights = balanced_boundary_pos_weights(
            train_targets, train_frame.source_weight.to_numpy(),
        )
        positive_weights_tensor = torch.tensor(
            positive_weights, dtype=torch.float32, device=device,
        )
        train_data = BoundaryDataset(
            train_frame, tokenizer, args.max_length, args.truncation_profile,
        )
        valid_data = BoundaryDataset(
            valid_frame, tokenizer, args.max_length, args.truncation_profile,
        )
        model = IndependentBoundaryClassifier(args.model, args.revision).to(device)
        head_ids = {id(parameter) for parameter in model.boundaries.parameters()}
        optimizer = torch.optim.AdamW([
            {
                "params": [parameter for parameter in model.parameters()
                           if id(parameter) not in head_ids],
                "lr": args.learning_rate,
            },
            {"params": model.boundaries.parameters(), "lr": args.head_learning_rate},
        ], weight_decay=0.01)
        generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            generator=generator, collate_fn=BoundaryCollator(tokenizer),
        )
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in train_loader:
                targets = batch.pop("targets").to(device)
                sample_weight = batch.pop("sample_weight").to(device)
                features = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    logits = model(**features)
                    element_loss = F.binary_cross_entropy_with_logits(
                        logits, targets, pos_weight=positive_weights_tensor,
                        reduction="none",
                    )
                    loss = (element_loss.mean(dim=1) * sample_weight).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            print(
                f"fold={fold} epoch={epoch + 1} loss={np.mean(losses):.6f}",
                flush=True,
            )

        model.eval()
        fold_logits = []
        with torch.no_grad():
            loader = DataLoader(
                valid_data, batch_size=args.batch_size,
                collate_fn=BoundaryCollator(tokenizer),
            )
            for batch in loader:
                batch.pop("targets")
                batch.pop("sample_weight")
                features = {key: value.to(device) for key, value in batch.items()}
                fold_logits.append(model(**features).float().cpu())
        fold_logits_array = torch.cat(fold_logits).numpy()
        all_logits[valid_frame.index] = fold_logits_array
        raw, ruled, raw_accuracy, ruled_accuracy = evaluate_predictions(
            valid_frame, fold_logits_array,
        )
        fold_result = {
            "fold": int(fold), "rows": len(valid_frame),
            "raw_accuracy": raw_accuracy, "ruled_accuracy": ruled_accuracy,
            "incumbent_ruled_accuracy": float(np.mean(
                pd.read_csv(
                    ROOT / "artifacts/icr_nested_lexical_blend/predictions.csv"
                ).set_index("id").loc[valid_frame.id, "incumbent"].astype(int).to_numpy()
                == valid_frame.answer.astype(int).to_numpy()
            )),
            "positive_weights": positive_weights.tolist(),
        }
        fold_results.append(fold_result)
        print(json.dumps(fold_result, ensure_ascii=False), flush=True)
        if args.save_fold_models:
            fold_dir = args.output_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), fold_dir / "model_state.pt")
            tokenizer.save_pretrained(fold_dir / "tokenizer")
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

        evaluated = ~np.isnan(all_logits).any(axis=1)
        partial = frame.loc[evaluated, ["id", "answer", "fold"]].copy()
        partial_raw, partial_ruled, _, _ = evaluate_predictions(
            frame.loc[evaluated], all_logits[evaluated],
        )
        partial["raw_prediction"] = partial_raw
        partial["ruled_prediction"] = partial_ruled
        for index in range(4):
            partial[f"boundary_logit_{index}"] = all_logits[evaluated, index]
        partial.to_csv(args.output_dir / "oof_predictions_partial.csv", index=False)

    evaluated = ~np.isnan(all_logits).any(axis=1)
    evaluated_frame = frame.loc[evaluated]
    raw, ruled, raw_accuracy, ruled_accuracy = evaluate_predictions(
        evaluated_frame, all_logits[evaluated],
    )
    summary = {
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "epochs": args.epochs,
        "completed_folds": [int(fold) for fold in folds],
        "evaluated_rows": int(evaluated.sum()),
        "raw_accuracy": raw_accuracy,
        "ruled_accuracy": ruled_accuracy,
        "folds": fold_results,
        "objective": "four independent source-balanced cumulative BCE heads",
        "raw_anchors_concatenated": False,
        "monotonicity": "decode-time least-squares projection only",
        "promotion_gate": (
            "fold 0 improves incumbent; fold 1 non-negative; final >=0.72, "
            "three folds improve/tie, at most one regression, source-bootstrap P>=0.90"
        ),
        "promotion_claim": False,
        "current_or_hidden_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
