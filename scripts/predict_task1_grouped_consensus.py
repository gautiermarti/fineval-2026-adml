#!/usr/bin/env python3
"""Deploy the validated two-seed Task 1 textual correction cascade.

All learned components are refit only on official training OOF disagreements.
The prediction frame is loaded text-only and is never required to contain an
``answer`` column.  Running this script on the released validation text is a
byte-level reproducibility check for the eventual hidden-test path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_task1_oof_annotation_program import (  # noqa: E402
    fit_components,
    gliner_candidates,
    load_oof_incumbent,
    load_validation_incumbent,
    make_operations,
    transform,
)
from evaluate_task1_annotation_policy import LEGAL_RE  # noqa: E402
from evaluate_task1_gliner_boundary_repair import (  # noqa: E402
    locate_one, starts_with_unannotated_article,
)
from fineval.task1_annotation_ops import apply_ranked_operations  # noqa: E402
from fineval.task1_operation_consensus import (  # noqa: E402
    apply_consensus_operations,
    grouped_consensus_operations,
)
from fineval.task1_policy_composition import compose_policy_deltas  # noqa: E402
from task1_numeric_baseline import parse_answer  # noqa: E402
from train_task1_textual import repair_embedded_label  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_sets(paths: list[Path], rows: int) -> list[set[tuple[str, str]]]:
    output = [set() for _ in range(rows)]
    for path in paths:
        records = load_jsonl(path)
        if len(records) != rows:
            raise ValueError(f"{path}: expected {rows} rows, found {len(records)}")
        for fallback, record in enumerate(records):
            output[int(record.get("row_index", fallback))].update(
                parse_answer(record.get("answer", ""))
            )
    return output


def dump_predictions(path: Path, rows: list[set[tuple[str, str]]]) -> None:
    path.write_text("".join(
        json.dumps({
            "row_index": index,
            "answer": "\n".join(
                f"{entity}, {kind}" for entity, kind in sorted(items)
            ),
        }, ensure_ascii=False) + "\n"
        for index, items in enumerate(rows)
    ))


def fit_operation_model(
    raw_path: Path, incumbent_path: Path, folds_path: Path, max_features: int,
):
    train = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
    )
    gold = [set(parse_answer(answer)) for answer in train.answer]
    candidates = gliner_candidates(train, raw_path)
    incumbent, occurrences = load_oof_incumbent(incumbent_path, len(train))
    folds_frame = pd.read_csv(folds_path).sort_values("row_index")
    if folds_frame.row_index.astype(int).tolist() != list(range(len(train))):
        raise ValueError(f"{folds_path}: incomplete training fold map")
    folds = folds_frame.fold.astype(int).tolist()
    operations = make_operations(
        train, gold, candidates, incumbent, occurrences, folds,
    )
    indices = np.arange(len(operations))
    return fit_components(operations, indices, max_features)


def accepted_operations(
    frame: pd.DataFrame, raw_path: Path, baseline: list[set[tuple[str, str]]],
    components, threshold: float,
) -> list[dict]:
    candidates = gliner_candidates(frame, raw_path)
    _, occurrences = load_validation_incumbent_from_sets(frame, baseline)
    operations = make_operations(
        frame, None, candidates, baseline, occurrences,
    )
    if not operations:
        return []
    indices = np.arange(len(operations))
    probabilities = components[-1].predict_proba(
        transform(operations, indices, components)
    )[:, 1]
    _, accepted = apply_ranked_operations(
        baseline, operations, probabilities, threshold,
    )
    return accepted


def load_validation_incumbent_from_sets(frame, rows):
    """Create the occurrence structure expected by the operation generator."""
    occurrences = [[] for _ in range(len(frame))]
    for row_index, items in enumerate(rows):
        text = str(frame.iloc[row_index].text)
        for entity, kind in items:
            forms = tuple(dict.fromkeys((entity, repair_embedded_label(entity))))
            seen_spans = set()
            for form in forms:
                for match in re.finditer(re.escape(form), text):
                    if match.span() in seen_spans:
                        continue
                    seen_spans.add(match.span())
                    occurrences[row_index].append({
                        "entity": entity, "type": kind, "surface_text": form,
                        "start": match.start(), "end": match.end(), "confidence": 1.0,
                    })
    return rows, occurrences


def structural_policy(
    frame: pd.DataFrame, baseline: list[set[tuple[str, str]]],
    raw_path: Path, threshold: float,
) -> tuple[list[set[tuple[str, str]]], list[dict]]:
    candidates = gliner_candidates(frame, raw_path)
    predictions = [set(row) for row in baseline]
    repairs = []
    flattened = sorted(
        (candidate for row in candidates for candidate in row),
        key=lambda row: row["score"], reverse=True,
    )
    for candidate in flattened:
        if candidate["type"] != "ΟΡΓΑΝΙΣΜΟΣ":
            continue
        if candidate["score"] < threshold or not LEGAL_RE.search(candidate["entity"]):
            continue
        if starts_with_unannotated_article(candidate["entity"]):
            continue
        row_index = int(candidate["row_index"])
        proposed = candidate["entity"], candidate["type"]
        if proposed in predictions[row_index]:
            continue
        text = str(frame.iloc[row_index].text)
        spans = []
        for entity, kind in predictions[row_index]:
            located = locate_one(text, entity)
            if located is not None:
                spans.append((*located, entity, kind))
        overlapping = [
            row for row in spans
            if row[0] < candidate["end"] and row[1] > candidate["start"]
        ]
        same_type = [row for row in overlapping if row[3] == candidate["type"]]
        if not same_type or not all(
            candidate["start"] <= left and right <= candidate["end"]
            for left, right, _, _ in same_type
        ):
            continue
        removed = {(entity, kind) for _, _, entity, kind in same_type}
        removed.update({
            (entity, kind) for left, right, entity, kind in overlapping
            if candidate["start"] <= left and right <= candidate["end"]
            and len(entity.strip()) <= 2
        })
        predictions[row_index].difference_update(removed)
        predictions[row_index].add(proposed)
        repairs.append({
            "row_index": row_index, "added": list(proposed),
            "removed": [list(item) for item in sorted(removed)],
            "gliner_score": float(candidate["score"]),
        })
    return predictions, repairs


def union_policy(
    frame: pd.DataFrame, baseline: list[set[tuple[str, str]]],
    raw_path: Path, threshold: float,
) -> tuple[list[set[tuple[str, str]]], list[dict]]:
    candidates = gliner_candidates(frame, raw_path)
    predictions = [set(row) for row in baseline]
    best = {}
    for row in candidates:
        for candidate in row:
            key = candidate["row_index"], candidate["entity"], candidate["type"]
            if key not in best or candidate["score"] > best[key]["score"]:
                best[key] = candidate
    additions = []
    for candidate in sorted(best.values(), key=lambda row: row["score"], reverse=True):
        if candidate["score"] < threshold:
            continue
        row_index = int(candidate["row_index"])
        item = candidate["entity"], candidate["type"]
        if item in predictions[row_index]:
            continue
        predictions[row_index].add(item)
        additions.append({
            "row_index": row_index, "entity": item[0], "type": item[1],
            "score": float(candidate["score"]),
        })
    return predictions, additions


def gate(path: Path) -> float:
    summary = json.loads(path.read_text())
    return float(summary["chosen_oof_gate"]["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, nargs="+", required=True)
    parser.add_argument("--seed7-full-raw", type=Path, required=True)
    parser.add_argument("--seed7-mean-raw", type=Path, required=True)
    parser.add_argument("--seed7-vote-raw", type=Path, required=True)
    parser.add_argument("--seed19-mean-raw", type=Path)
    parser.add_argument("--seed19-vote-raw", type=Path)
    parser.add_argument(
        "--seed7-train-raw", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/oof_raw_predictions.jsonl",
    )
    parser.add_argument(
        "--seed19-train-raw", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed19_width16_ensemble/oof_raw_predictions.jsonl",
    )
    parser.add_argument(
        "--train-incumbent", type=Path,
        default=ROOT / "artifacts/task1_multilabel_oof_legal_seed7/oof_predictions.jsonl",
    )
    parser.add_argument(
        "--seed7-folds", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/folds.csv",
    )
    parser.add_argument(
        "--seed19-folds", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed19_width16_ensemble/folds.csv",
    )
    parser.add_argument(
        "--seed7-summary", type=Path,
        default=ROOT / "artifacts/task1_oof_annotation_program_seed7/summary.json",
    )
    parser.add_argument(
        "--seed19-summary", type=Path,
        default=ROOT / "artifacts/task1_oof_annotation_program_seed19_foldmean/summary.json",
    )
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument(
        "--consensus-mode", choices=("two_seed", "single_seed"),
        default="two_seed",
    )
    parser.add_argument("--minimum-support", type=int)
    parser.add_argument("--minimum-groups", type=int)
    parser.add_argument("--structural-threshold", type=float, default=0.935)
    parser.add_argument("--union-threshold", type=float, default=0.94)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_grouped_consensus_predictions",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input, columns=["text"])
    baseline = load_sets(args.incumbent, len(frame))
    seed7_model = fit_operation_model(
        args.seed7_train_raw, args.train_incumbent, args.seed7_folds,
        args.max_features,
    )
    view_specs = [
        ("seed7", args.seed7_full_raw, seed7_model, gate(args.seed7_summary)),
        ("seed7", args.seed7_mean_raw, seed7_model, gate(args.seed7_summary)),
        ("seed7", args.seed7_vote_raw, seed7_model, gate(args.seed7_summary)),
    ]
    if args.consensus_mode == "two_seed":
        if args.seed19_mean_raw is None or args.seed19_vote_raw is None:
            parser.error("two_seed mode requires both seed-19 raw views")
        seed19_model = fit_operation_model(
            args.seed19_train_raw, args.train_incumbent, args.seed19_folds,
            args.max_features,
        )
        view_specs.extend([
            ("seed19", args.seed19_mean_raw, seed19_model, gate(args.seed19_summary)),
            ("seed19", args.seed19_vote_raw, seed19_model, gate(args.seed19_summary)),
        ])
    minimum_support = args.minimum_support
    if minimum_support is None:
        minimum_support = 3 if args.consensus_mode == "two_seed" else 2
    minimum_groups = args.minimum_groups
    if minimum_groups is None:
        minimum_groups = 2 if args.consensus_mode == "two_seed" else 1
    views = []
    for group, raw, components, threshold in view_specs:
        views.append(accepted_operations(frame, raw, baseline, components, threshold))
    consensus = grouped_consensus_operations(
        views, [item[0] for item in view_specs],
        minimum_support, minimum_groups,
    )
    operation = apply_consensus_operations(baseline, consensus)
    structural, repairs = structural_policy(
        frame, baseline, args.seed7_mean_raw, args.structural_threshold,
    )
    union, additions = union_policy(
        frame, baseline, args.seed7_mean_raw, args.union_threshold,
    )
    composed = [
        compose_policy_deltas(
            baseline[index], [operation[index], structural[index], union[index]],
        )
        for index in range(len(frame))
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, predictions in (
        ("operation_predictions.jsonl", operation),
        ("structural_predictions.jsonl", structural),
        ("union_predictions.jsonl", union),
        ("predictions.jsonl", composed),
    ):
        dump_predictions(args.output_dir / name, predictions)
    (args.output_dir / "selected_operations.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in consensus
    ))
    (args.output_dir / "repairs.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in repairs
    ))
    (args.output_dir / "additions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in additions
    ))
    manifest = {
        "rows": len(frame), "prediction_answer_column_accessed": False,
        "consensus_mode": args.consensus_mode,
        "seed7_operation_gate": gate(args.seed7_summary),
        "seed19_operation_gate": (
            gate(args.seed19_summary) if args.consensus_mode == "two_seed" else None
        ),
        "operation_views": [str(item[1]) for item in view_specs],
        "view_groups": [item[0] for item in view_specs],
        "minimum_support": minimum_support,
        "minimum_groups": minimum_groups,
        "selected_operations": len(consensus),
        "structural_threshold": args.structural_threshold,
        "structural_repairs": len(repairs),
        "union_threshold": args.union_threshold,
        "union_additions": len(additions),
        "current_or_hidden_labels_used": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
