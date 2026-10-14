#!/usr/bin/env python3
"""Generate entity-disjoint out-of-fold GLiNER proposals for GRFinNER.

Only the official 320-row training labels are read.  Rows sharing a normalized
organization or person are kept in one fold, preventing exact issuer/person
memorization from inflating the proposal-policy training data.  Fold models are
ephemeral; low-threshold held-out proposals and optional input-only validation
proposals are retained.  The validation answer column is never loaded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.task1_folds import (  # noqa: E402
    assign_entity_disjoint_folds,
    audit_entity_disjoint_folds,
)
from task1_numeric_baseline import parse_answer  # noqa: E402
from train_task1_gliner import (  # noqa: E402
    LABEL_TO_GREEK,
    configure_max_width,
    make_gliner_dataset,
    train_model_compat,
)


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def dump_jsonl(path: Path, rows) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="urchade/gliner_multi-v2.1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--fold", dest="folds", type=int, action="append",
        help="Run only this fold (repeatable); defaults to all missing folds.",
    )
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-width", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--save-fold-models", action="store_true",
        help="Persist each trained fold model for later hidden-test inference.",
    )
    parser.add_argument(
        "--validation-input", type=Path,
        default=ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet",
        help="Optional text-only split predicted by every fold model.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed7",
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
        "seed": args.seed,
        "n_splits": args.n_splits,
        "max_steps_per_fold": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "max_width": args.max_width,
        "proposal_threshold": args.threshold,
        "save_fold_models": bool(args.save_fold_models),
        "split": "normalized exact organization/person connected components",
        "validation_text_predicted": str(args.validation_input),
        "validation_answer_column_loaded": False,
        "public_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(
            f"{args.output_dir} already contains a different configuration; "
            "choose a new output directory"
        )
    dump_json(config_path, config)
    folds_frame = pd.DataFrame({
        "row_index": range(len(frame)), "fold": assignments,
    })
    folds_frame.to_csv(args.output_dir / "folds.csv", index=False)
    split_audit["assignment_sha256"] = hashlib.sha256(
        ",".join(map(str, assignments)).encode()
    ).hexdigest()
    dump_json(args.output_dir / "split_audit.json", split_audit)
    validation_text = pd.read_parquet(args.validation_input, columns=["text"])

    from gliner import GLiNER

    fold_audits = {}
    for fold in requested_folds:
        fold_dir = args.output_dir / f"fold_{fold}"
        prediction_path = fold_dir / "raw_predictions.jsonl"
        validation_prediction_path = fold_dir / "validation_raw_predictions.jsonl"
        saved_model = fold_dir / "final"
        model_saved = (saved_model / "gliner_config.json").exists()
        if (
            prediction_path.exists() and validation_prediction_path.exists()
            and (not args.save_fold_models or model_saved)
        ):
            print(f"fold={fold} already complete; skipping", flush=True)
            continue
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_seed = args.seed + fold
        random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        train_indices = [
            index for index, value in enumerate(assignments) if value != fold
        ]
        held_indices = [
            index for index, value in enumerate(assignments) if value == fold
        ]
        train_frame = frame.iloc[train_indices]
        held_frame = frame.iloc[held_indices]
        print(
            f"fold={fold} seed={fold_seed} train_rows={len(train_frame)} "
            f"held_rows={len(held_frame)}",
            flush=True,
        )

        model = GLiNER.from_pretrained(args.model, map_location="cpu")
        configure_max_width(model, args.max_width)
        train_dataset, alignment_audit = make_gliner_dataset(
            train_frame, model.data_processor.words_splitter, args.max_width,
        )
        training_args = model.create_training_args(
            output_dir=fold_dir / "ephemeral_checkpoints",
            learning_rate=args.encoder_learning_rate,
            others_lr=args.head_learning_rate,
            weight_decay=0.01,
            others_weight_decay=0.01,
            warmup_ratio=0.10,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            logging_steps=max(1, args.max_steps // 8),
            save_strategy="no",
            use_cpu=args.cpu,
            bf16=not args.cpu,
            dataloader_num_workers=1,
            report_to="none",
            eval_strategy="no",
            remove_unused_columns=False,
            seed=fold_seed,
            data_seed=fold_seed,
        )
        trainer = train_model_compat(model, train_dataset, training_args)
        if args.save_fold_models:
            model.save_pretrained(saved_model)
        raw = model.batch_predict_entities(
            held_frame["text"].astype(str).tolist(), list(LABEL_TO_GREEK),
            flat_ner=False, multi_label=False, threshold=args.threshold,
            batch_size=args.batch_size,
        )
        validation_raw = model.batch_predict_entities(
            validation_text["text"].astype(str).tolist(), list(LABEL_TO_GREEK),
            flat_ner=False, multi_label=False, threshold=args.threshold,
            batch_size=args.batch_size,
        )
        records = [
            {"row_index": int(row_index), "fold": fold, "entities": entities}
            for row_index, entities in zip(held_indices, raw)
        ]
        dump_jsonl(prediction_path, records)
        dump_jsonl(validation_prediction_path, [
            {"row_index": row_index, "fold": fold, "entities": entities}
            for row_index, entities in enumerate(validation_raw)
        ])
        fold_audits[str(fold)] = {
            "seed": fold_seed,
            "train_rows": len(train_indices),
            "held_rows": len(held_indices),
            "held_proposals": sum(len(row) for row in raw),
            "validation_rows_predicted_input_only": len(validation_raw),
            "validation_proposals": sum(len(row) for row in validation_raw),
            "alignment": alignment_audit,
        }
        dump_json(fold_dir / "audit.json", fold_audits[str(fold)])
        del trainer, model
        gc.collect()
        if not args.cpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

    completed_records = []
    completed_folds = []
    for fold in range(args.n_splits):
        path = args.output_dir / f"fold_{fold}" / "raw_predictions.jsonl"
        if not path.exists():
            continue
        completed_folds.append(fold)
        completed_records.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    row_indices = [int(record["row_index"]) for record in completed_records]
    if len(row_indices) != len(set(row_indices)):
        raise AssertionError("duplicate OOF row predictions")
    if len(completed_folds) == args.n_splits:
        if sorted(row_indices) != list(range(len(frame))):
            raise AssertionError("complete folds do not cover every training row exactly once")
        completed_records.sort(key=lambda record: int(record["row_index"]))
        dump_jsonl(args.output_dir / "oof_raw_predictions.jsonl", completed_records)

        # Exact-span fold ensemble.  Missing proposals contribute zero to the
        # all-fold mean, making agreement an explicit confidence signal.
        validation_by_fold = []
        for fold in range(args.n_splits):
            path = args.output_dir / f"fold_{fold}" / "validation_raw_predictions.jsonl"
            rows = [
                json.loads(line) for line in path.read_text().splitlines() if line.strip()
            ]
            if len(rows) != len(validation_text):
                raise AssertionError("fold validation prediction count mismatch")
            validation_by_fold.append(rows)
        mean_records, vote_records = [], []
        for row_index in range(len(validation_text)):
            candidates = {}
            for fold in range(args.n_splits):
                for entity in validation_by_fold[fold][row_index]["entities"]:
                    key = (
                        int(entity["start"]), int(entity["end"]),
                        str(entity["label"]), str(entity["text"]),
                    )
                    candidates.setdefault(key, {})[fold] = float(entity["score"])
            mean_entities, vote_entities = [], []
            for (start, end, label, entity_text), scores_by_fold in candidates.items():
                fold_scores = [scores_by_fold.get(fold, 0.0) for fold in range(args.n_splits)]
                present_scores = list(scores_by_fold.values())
                enriched = {
                    "start": start, "end": end, "label": label, "text": entity_text,
                    "score": sum(fold_scores) / args.n_splits,
                    "mean_score_all_folds": sum(fold_scores) / args.n_splits,
                    "mean_score_present": sum(present_scores) / len(present_scores),
                    "vote_count": len(present_scores),
                    "fold_scores": fold_scores,
                }
                mean_entities.append(enriched)
                if len(present_scores) >= (args.n_splits // 2 + 1):
                    vote_entities.append(enriched | {"score": enriched["mean_score_present"]})
            mean_entities.sort(key=lambda row: (row["start"], row["end"], row["label"]))
            vote_entities.sort(key=lambda row: (row["start"], row["end"], row["label"]))
            mean_records.append({"row_index": row_index, "entities": mean_entities})
            vote_records.append({"row_index": row_index, "entities": vote_entities})
        dump_jsonl(args.output_dir / "validation_ensemble_mean_raw.jsonl", mean_records)
        dump_jsonl(args.output_dir / "validation_ensemble_vote3_raw.jsonl", vote_records)
    manifest = config | {
        "completed_folds": completed_folds,
        "complete": len(completed_folds) == args.n_splits,
        "rows_predicted": len(completed_records),
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
