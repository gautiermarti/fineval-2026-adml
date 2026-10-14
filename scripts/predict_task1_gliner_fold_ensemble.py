#!/usr/bin/env python3
"""Infer fold-mean and vote views from persisted Task 1 GLiNER models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))


def ensemble(validation_by_fold: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    n_splits = len(validation_by_fold)
    rows = len(validation_by_fold[0])
    mean_records, vote_records = [], []
    for row_index in range(rows):
        candidates = {}
        for fold in range(n_splits):
            for entity in validation_by_fold[fold][row_index]["entities"]:
                key = (
                    int(entity["start"]), int(entity["end"]),
                    str(entity["label"]), str(entity["text"]),
                )
                candidates.setdefault(key, {})[fold] = float(entity["score"])
        mean_entities, vote_entities = [], []
        for (start, end, label, text), scores_by_fold in candidates.items():
            fold_scores = [scores_by_fold.get(fold, 0.0) for fold in range(n_splits)]
            present = list(scores_by_fold.values())
            enriched = {
                "start": start, "end": end, "label": label, "text": text,
                "score": sum(fold_scores) / n_splits,
                "mean_score_all_folds": sum(fold_scores) / n_splits,
                "mean_score_present": sum(present) / len(present),
                "vote_count": len(present), "fold_scores": fold_scores,
            }
            mean_entities.append(enriched)
            if len(present) >= (n_splits // 2 + 1):
                vote_entities.append(enriched | {"score": enriched["mean_score_present"]})
        mean_entities.sort(key=lambda row: (row["start"], row["end"], row["label"]))
        vote_entities.sort(key=lambda row: (row["start"], row["end"], row["label"]))
        mean_records.append({"row_index": row_index, "entities": mean_entities})
        vote_records.append({"row_index": row_index, "entities": vote_entities})
    return mean_records, vote_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu for intentional CPU inference")
    from gliner import GLiNER

    frame = pd.read_parquet(args.input, columns=["text"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_by_fold = []
    for fold in range(args.n_splits):
        model_dir = args.models_dir / f"fold_{fold}" / "final"
        if not (model_dir / "gliner_config.json").exists():
            raise FileNotFoundError(f"missing persisted fold model: {model_dir}")
        model = GLiNER.from_pretrained(model_dir, map_location="cpu")
        if not args.cpu:
            model = model.to("cuda")
        raw = model.batch_predict_entities(
            frame.text.astype(str).tolist(), ["person", "organization", "location"],
            flat_ner=False, multi_label=False, threshold=args.threshold,
            batch_size=args.batch_size,
        )
        records = [
            {"row_index": index, "fold": fold, "entities": entities}
            for index, entities in enumerate(raw)
        ]
        dump_jsonl(args.output_dir / f"fold_{fold}_raw.jsonl", records)
        validation_by_fold.append(records)
        del model
        gc.collect()
        if not args.cpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean, vote = ensemble(validation_by_fold)
    dump_jsonl(args.output_dir / "ensemble_mean_raw.jsonl", mean)
    dump_jsonl(args.output_dir / "ensemble_vote3_raw.jsonl", vote)
    manifest = {
        "rows": len(frame), "folds": args.n_splits,
        "threshold": args.threshold, "device": "cpu" if args.cpu else "cuda",
        "answer_column_accessed": False,
        "mean_entities": sum(len(row["entities"]) for row in mean),
        "vote_entities": sum(len(row["entities"]) for row in vote),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
