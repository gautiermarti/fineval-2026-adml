#!/usr/bin/env python3
"""Entity-fold-disjoint neural ranker for bounded Task 1 annotation edits."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_task1_oof_annotation_program import (  # noqa: E402
    gliner_candidates, load_oof_incumbent, make_operations,
)
from fineval.task1_annotation_ops import apply_ranked_operations  # noqa: E402
from task1_numeric_baseline import parse_answer, score  # noqa: E402


DEFAULT_MODEL = "spyrosbriakos/greek_legal_bert_v2"
DEFAULT_REVISION = "e06fba0eb7a107f401055bb174f9e1ba1a29ba17"
SURGERY_PATH = ROOT / "artifacts/task1_span_surgery_v1/span_choice_pairs.jsonl"
TRAIN_PATH = ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
FOLDS_PATH = ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/folds.csv"
GLINER_PATH = ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/oof_raw_predictions.jsonl"
INCUMBENT_PATH = ROOT / "artifacts/task1_multilabel_oof_legal_seed7/oof_predictions.jsonl"
LINEAR_FOLD0_DELTA = 0.07788318452380953


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def marked_context(
    context: str, start: int, end: int, *, context_start: int = 0,
) -> str:
    relative_start, relative_end = start - context_start, end - context_start
    if not (0 <= relative_start < relative_end <= len(context)):
        raise ValueError("candidate span is outside its context")
    return (
        context[:relative_start] + " [MASK] "
        + context[relative_start:relative_end] + " [MASK] "
        + context[relative_end:]
    )


def encode_candidate(
    tokenizer, context: str, candidate: dict, descriptor: str,
    max_length: int, *, context_start: int = 0,
) -> dict:
    marked = marked_context(
        context, int(candidate["start"]), int(candidate["end"]),
        context_start=context_start,
    )
    second = (
        f"{candidate.get('entity_type', candidate.get('type', ''))} "
        f"{candidate.get('surface', candidate.get('entity', ''))} {descriptor}"
    )
    return tokenizer(
        marked, second, truncation="longest_first", max_length=max_length,
    )


class SurgeryDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = []
        for row in rows:
            common = {
                "context": row["context"],
                "context_start": int(row["context_start"]),
            }
            preferred = encode_candidate(
                tokenizer, common["context"], row["preferred"], "candidate",
                max_length, context_start=common["context_start"],
            )
            rejected = encode_candidate(
                tokenizer, common["context"], row["rejected"],
                "candidate",
                max_length, context_start=common["context_start"],
            )
            self.rows.append({
                "preferred": preferred,
                "rejected": rejected,
                "weight": float(row["pair_weight"]),
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class OperationDataset(Dataset):
    def __init__(
        self, operations: list[dict], frame: pd.DataFrame, tokenizer,
        max_length: int,
    ):
        self.rows = []
        for index, operation in enumerate(operations):
            row_index = int(operation["row_index"])
            candidate = {
                "start": int(operation["candidate_start"]),
                "end": int(operation["candidate_end"]),
                "type": operation["candidate"][1],
                "entity": operation["candidate"][0],
            }
            removed = "; ".join(
                f"{entity} {kind}" for entity, kind in operation["remove"]
            )
            descriptor = f"{operation['operation']} remove {removed}"
            encoded = encode_candidate(
                tokenizer, str(frame.iloc[row_index].text), candidate,
                descriptor, max_length,
            )
            self.rows.append({
                "operation_index": index,
                "encoded": encoded,
                "label": int(operation["beneficial"]),
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def collate_surgery(batch: list[dict], tokenizer) -> dict:
    return {
        "preferred": tokenizer.pad(
            [row["preferred"] for row in batch], return_tensors="pt",
        ),
        "rejected": tokenizer.pad(
            [row["rejected"] for row in batch], return_tensors="pt",
        ),
        "weights": torch.tensor([row["weight"] for row in batch]),
    }


def collate_operations(batch: list[dict], tokenizer) -> dict:
    encoded = tokenizer.pad(
        [row["encoded"] for row in batch], return_tensors="pt",
    )
    encoded["labels"] = torch.tensor([
        row["label"] for row in batch
    ], dtype=torch.float)
    encoded["operation_indices"] = torch.tensor([
        row["operation_index"] for row in batch
    ], dtype=torch.long)
    return encoded


def module_gradient_norm(model) -> float:
    return float(sum(
        parameter.grad.detach().float().norm().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    ))


def row_policy_metrics(
    gold: list[set], baseline: list[set], operations: list[dict],
    probabilities: np.ndarray, fold_rows: np.ndarray,
) -> tuple[dict, list[dict]]:
    prediction, accepted = apply_ranked_operations(
        baseline, operations, probabilities, 0.5,
    )
    base_scores = np.asarray([
        score(list(target), list(guess))[2]
        for target, guess in zip(gold, baseline)
    ])
    candidate_scores = np.asarray([
        score(list(target), list(guess))[2]
        for target, guess in zip(gold, prediction)
    ])
    delta = candidate_scores[fold_rows] - base_scores[fold_rows]
    return {
        "rows": int(fold_rows.sum()),
        "baseline_macro_entity_f1": float(base_scores[fold_rows].mean()),
        "policy_macro_entity_f1": float(candidate_scores[fold_rows].mean()),
        "delta": float(delta.mean()),
        "rows_improved": int((delta > 1e-12).sum()),
        "rows_harmed": int((delta < -1e-12).sum()),
        "accepted_operations": sum(
            int(fold_rows[int(operation["row_index"])]) for operation in accepted
        ),
    }, accepted


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--pretrain-learning-rate", type=float, default=2e-5)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-folds", type=int, default=1)
    parser.add_argument("--max-pretrain-steps", type=int)
    parser.add_argument("--max-finetune-steps", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_neural_operation_ranker_seed7_v1",
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
        raise ValueError("CUDA requested but unavailable")
    if args.bf16 and device.type != "cuda":
        raise ValueError("--bf16 requires CUDA")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    frame = pd.read_parquet(TRAIN_PATH)
    folds_frame = pd.read_csv(FOLDS_PATH).sort_values("row_index")
    if folds_frame.row_index.astype(int).tolist() != list(range(len(frame))):
        raise ValueError("entity-linked folds do not align to training rows")
    folds = folds_frame.fold.astype(int).to_numpy()
    surgery_rows = load_jsonl(SURGERY_PATH)
    if len(surgery_rows) != 7945:
        raise ValueError("span-surgery corpus differs from its frozen audit")
    candidates = gliner_candidates(frame, GLINER_PATH)
    baseline, occurrences = load_oof_incumbent(INCUMBENT_PATH, len(frame))
    gold = [set(parse_answer(answer)) for answer in frame.answer]
    operations = make_operations(
        frame, gold, candidates, baseline, occurrences, folds.tolist(),
    )
    if len(operations) != 418 or sum(row["beneficial"] for row in operations) != 87:
        raise ValueError("genuine operation corpus differs from its frozen audit")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision,
    )
    fold_values = sorted(set(folds))[:args.max_folds]
    operation_probabilities = np.full(len(operations), np.nan, dtype=float)
    fold_summaries = []
    first_step_audits = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold in fold_values:
        surgery_train = [row for row in surgery_rows if int(row["fold"]) != fold]
        operation_train_indices = [
            index for index, row in enumerate(operations) if int(row["fold"]) != fold
        ]
        operation_valid_indices = [
            index for index, row in enumerate(operations) if int(row["fold"]) == fold
        ]
        surgery = SurgeryDataset(surgery_train, tokenizer, args.max_length)
        operation_train = OperationDataset(
            [operations[index] for index in operation_train_indices],
            frame, tokenizer, args.max_length,
        )
        operation_valid = OperationDataset(
            [operations[index] for index in operation_valid_indices],
            frame, tokenizer, args.max_length,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, revision=args.revision, num_labels=1,
            problem_type="regression", ignore_mismatched_sizes=True,
        ).to(device)

        generator = torch.Generator().manual_seed(args.seed + int(fold))
        pretrain_loader = DataLoader(
            surgery, batch_size=args.batch_size, shuffle=True, generator=generator,
            collate_fn=lambda rows: collate_surgery(rows, tokenizer),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.pretrain_learning_rate, weight_decay=0.01,
        )
        pretrain_steps = 0
        stage_audit = {}
        for _ in range(args.pretrain_epochs):
            model.train()
            for batch in pretrain_loader:
                preferred = {key: value.to(device) for key, value in batch["preferred"].items()}
                rejected = {key: value.to(device) for key, value in batch["rejected"].items()}
                weights = batch["weights"].to(device)
                optimizer.zero_grad()
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    preferred_score = model(**preferred).logits[:, 0]
                    rejected_score = model(**rejected).logits[:, 0]
                    pair_loss = F.softplus(-(preferred_score - rejected_score))
                    loss = (pair_loss * weights).sum() / weights.sum().clamp_min(1e-12)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite surgery loss")
                loss.backward()
                if "pretrain" not in stage_audit:
                    stage_audit["pretrain"] = {
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm": module_gradient_norm(model),
                    }
                optimizer.step()
                pretrain_steps += 1
                if args.max_pretrain_steps and pretrain_steps >= args.max_pretrain_steps:
                    break
            if args.max_pretrain_steps and pretrain_steps >= args.max_pretrain_steps:
                break

        finetune_loader = DataLoader(
            operation_train, batch_size=args.batch_size, shuffle=True,
            generator=generator,
            collate_fn=lambda rows: collate_operations(rows, tokenizer),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.finetune_learning_rate, weight_decay=0.01,
        )
        finetune_steps = 0
        for _ in range(args.finetune_epochs):
            model.train()
            for batch in finetune_loader:
                labels = batch.pop("labels").to(device)
                batch.pop("operation_indices")
                encoded = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad()
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    logits = model(**encoded).logits[:, 0]
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite operation loss")
                loss.backward()
                if "finetune" not in stage_audit:
                    stage_audit["finetune"] = {
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm": module_gradient_norm(model),
                    }
                optimizer.step()
                finetune_steps += 1
                if args.max_finetune_steps and finetune_steps >= args.max_finetune_steps:
                    break
            if args.max_finetune_steps and finetune_steps >= args.max_finetune_steps:
                break

        model.eval()
        held_scores = []
        held_local_indices = []
        validation_loader = DataLoader(
            operation_valid, batch_size=args.batch_size,
            collate_fn=lambda rows: collate_operations(rows, tokenizer),
        )
        with torch.no_grad():
            for batch in validation_loader:
                batch.pop("labels")
                held_local_indices.extend(batch.pop("operation_indices").tolist())
                encoded = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    held_scores.extend(model(**encoded).logits[:, 0].float().cpu().tolist())
        probabilities = torch.sigmoid(torch.tensor(held_scores)).numpy()
        # OperationDataset indices are local; restore the original operation order.
        restored = np.empty(len(operation_valid_indices), dtype=float)
        restored[np.asarray(held_local_indices)] = probabilities
        operation_probabilities[operation_valid_indices] = restored
        fold_rows = folds == fold
        metrics, accepted = row_policy_metrics(
            gold, baseline,
            [operations[index] for index in operation_valid_indices],
            restored, fold_rows,
        )
        held_labels = np.asarray([
            operations[index]["beneficial"] for index in operation_valid_indices
        ])
        metrics.update({
            "fold": int(fold),
            "operation_rows": len(operation_valid_indices),
            "operation_average_precision": float(
                average_precision_score(held_labels, restored)
            ),
            "operation_roc_auc": float(roc_auc_score(held_labels, restored)),
            "pretrain_steps": pretrain_steps,
            "finetune_steps": finetune_steps,
            "passes_fold0_futility_gate": bool(
                fold != 0 or (
                    metrics["delta"] >= LINEAR_FOLD0_DELTA - 1e-12
                    and metrics["rows_harmed"] <= 2
                )
            ),
        })
        fold_summaries.append(metrics)
        first_step_audits.append({"fold": int(fold), **stage_audit})
        fold_dir = args.output_dir / f"fold{fold}"
        model.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)
        (fold_dir / "accepted_operations.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in accepted
        ))
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    smoke = args.max_pretrain_steps is not None or args.max_finetune_steps is not None
    if smoke:
        for row in fold_summaries:
            row["passes_fold0_futility_gate"] = None
    operation_output = []
    for index, probability in enumerate(operation_probabilities):
        if np.isnan(probability):
            continue
        operation_output.append({
            "operation_index": index,
            "row_index": int(operations[index]["row_index"]),
            "fold": int(operations[index]["fold"]),
            "beneficial": int(operations[index]["beneficial"]),
            "probability": float(probability),
        })
    (args.output_dir / "oof_operation_predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in operation_output
    ))
    summary = {
        "evaluation_kind": "pipeline_smoke_only" if smoke else "entity_fold_disjoint",
        "gate_applied": not smoke,
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "folds_completed": len(fold_values),
        "surgery_pairs": len(surgery_rows),
        "genuine_operations": len(operations),
        "beneficial_operations": sum(row["beneficial"] for row in operations),
        "fixed_threshold": 0.5,
        "linear_fold0_delta_to_match": LINEAR_FOLD0_DELTA,
        "folds": fold_summaries,
        "first_step_audits": first_step_audits,
        "input_sha256": {
            "training": digest(TRAIN_PATH),
            "folds": digest(FOLDS_PATH),
            "surgery": digest(SURGERY_PATH),
            "gliner_oof": digest(GLINER_PATH),
            "incumbent_oof": digest(INCUMBENT_PATH),
        },
        "validation_used": False,
        "public_test_used": False,
        "hidden_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
