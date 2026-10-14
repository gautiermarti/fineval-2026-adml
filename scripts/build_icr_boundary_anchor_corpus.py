#!/usr/bin/env python3
"""Build source-disjoint adjacent-boundary comparison data for JF-ICR.

Unlike a conventional cumulative head, each binary decision is conditioned on
two concrete training anchors immediately below and above that ordinal
boundary.  Anchor retrieval is fitted inside each outer source fold and uses no
target label.  Gold labels are used only to train/evaluate the four boundary
decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_events import event_candidates  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402


LABELS = ("-2", "-1", "0", "+1", "+2")
BOUNDARIES = (
    ("-2|-1", "more committed than explicit rejection or impossibility"),
    ("-1|0", "more committed than refusal, withholding, or non-disclosure"),
    ("0|+1", "expresses a forward action or intention rather than neutral explanation"),
    ("+1|+2", "is definite or action-backed rather than a qualified intention"),
)
PROTOTYPES = {
    (0, "below"): "明確に実施しない。不可能であり、再検討の余地はない。",
    (0, "above"): "現時点では予定がないが、状況次第で再検討する余地がある。",
    (1, "below"): "回答を控え、具体的な予定を示さない。",
    (1, "above"): "状況を説明するが、将来の行動は約束しない。",
    (2, "below"): "中立的な説明にとどまり、将来の行動を示さない。",
    (2, "above"): "条件付きながら、将来実施する意向を示す。",
    (3, "below"): "実施を検討または予定しているが、確定していない。",
    (3, "above"): "時期または具体的行動を伴って実施を明確に確約する。",
}


def input_view(question: str, response: str) -> str:
    return question + " [Q-A] " + response


def compact_record(frame: pd.DataFrame, parsed: list, position: int, similarity: float) -> dict:
    item = parsed[position]
    events = event_candidates(item.question, item.response)[:3]
    return {
        "anchor_kind": "retrieved_training_example",
        "id": int(frame.iloc[position].id),
        "similarity": float(similarity),
        "question": item.question,
        "response": item.response,
        "top_events": [event.text for event in events],
    }


def prototype_record(boundary_index: int, side: str) -> dict:
    text = PROTOTYPES[(boundary_index, side)]
    return {
        "anchor_kind": "rubric_prototype",
        "id": None,
        "similarity": None,
        "question": "境界基準例",
        "response": text,
        "top_events": [text],
    }


def nearest_anchor(
    target_position: int,
    candidates: np.ndarray,
    similarities: np.ndarray,
    source_groups: np.ndarray,
) -> tuple[int | None, bool]:
    candidates = candidates[candidates != target_position]
    if not len(candidates):
        return None, False
    different_source = candidates[
        source_groups[candidates] != source_groups[target_position]
    ]
    used_different_source = bool(len(different_source))
    pool = different_source if used_different_source else candidates
    return int(pool[np.argmax(similarities[pool])]), used_different_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet",
    )
    parser.add_argument(
        "--folds", type=Path,
        default=ROOT / "data/derived/icr_source_folds.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_boundary_anchors_v2",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).sort_values("id").reset_index(drop=True)
    metadata = (
        pd.read_csv(args.folds).set_index("id").loc[frame.id].reset_index(drop=True)
    )
    parsed = [parse_icr_query(query) for query in frame["query"]]
    texts = [input_view(item.question, item.response) for item in parsed]
    label_indices = np.asarray([LABELS.index(str(label)) for label in frame.answer])
    source_groups = metadata.source_group.astype(str).to_numpy()
    source_folds = metadata.source_fold.astype(int).to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_folds = {}
    global_counts = Counter()
    anchor_source_counts = defaultdict(set)
    for fold in sorted(set(source_folds)):
        train_positions = np.flatnonzero(source_folds != fold)
        evaluation_positions = np.flatnonzero(source_folds == fold)
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=2,
            max_features=60_000, sublinear_tf=True,
        )
        train_matrix = vectorizer.fit_transform([texts[index] for index in train_positions])
        all_matrix = vectorizer.transform(texts)
        similarities = (all_matrix @ train_matrix.T).toarray()
        # Expand train-column similarities back to global row positions.
        global_similarities = np.full((len(frame), len(frame)), -np.inf)
        global_similarities[:, train_positions] = similarities

        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_records = {"train": [], "evaluation": []}
        split_positions = {
            "train": train_positions,
            "evaluation": evaluation_positions,
        }
        different_source_count = 0
        retrieved_anchor_count = 0
        prototype_anchor_count = 0
        for split_name, positions in split_positions.items():
            for position in positions:
                item = parsed[position]
                target_events = event_candidates(item.question, item.response)[:3]
                boundaries = []
                for boundary_index, (boundary_name, meaning) in enumerate(BOUNDARIES):
                    below = train_positions[label_indices[train_positions] <= boundary_index]
                    above = train_positions[label_indices[train_positions] > boundary_index]
                    below_position, below_different = nearest_anchor(
                        position, below, global_similarities[position], source_groups,
                    )
                    above_position, above_different = nearest_anchor(
                        position, above, global_similarities[position], source_groups,
                    )
                    for anchor_position, is_different in (
                        (below_position, below_different),
                        (above_position, above_different),
                    ):
                        if anchor_position is None:
                            prototype_anchor_count += 1
                        else:
                            retrieved_anchor_count += 1
                            different_source_count += int(is_different)
                            anchor_source_counts[boundary_name].add(
                                source_groups[anchor_position]
                            )
                    below_record = (
                        compact_record(
                            frame, parsed, below_position,
                            global_similarities[position, below_position],
                        )
                        if below_position is not None
                        else prototype_record(boundary_index, "below")
                    )
                    above_record = (
                        compact_record(
                            frame, parsed, above_position,
                            global_similarities[position, above_position],
                        )
                        if above_position is not None
                        else prototype_record(boundary_index, "above")
                    )
                    boundaries.append({
                        "boundary_index": boundary_index,
                        "boundary": boundary_name,
                        "meaning": meaning,
                        "gold_above_boundary": bool(label_indices[position] > boundary_index),
                        "below_anchor": below_record,
                        "above_anchor": above_record,
                    })
                    global_counts[
                        f"{boundary_name}:{int(label_indices[position] > boundary_index)}"
                    ] += 1
                split_records[split_name].append({
                    "id": int(frame.iloc[position].id),
                    "source_fold": int(fold),
                    "source_group": source_groups[position],
                    "gold_label": str(frame.iloc[position].answer),
                    "question": item.question,
                    "response": item.response,
                    "top_event_candidates": [event.text for event in target_events],
                    "boundaries": boundaries,
                })
        for split_name, records in split_records.items():
            (fold_dir / f"{split_name}.jsonl").write_text("".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in records
            ))
        summary_folds[str(fold)] = {
            "training_rows": len(train_positions),
            "evaluation_rows": len(evaluation_positions),
            "boundary_records_written": sum(
                len(record["boundaries"])
                for records in split_records.values() for record in records
            ),
            "rubric_prototype_anchors": prototype_anchor_count,
            "anchors_from_different_source_group_rate": (
                different_source_count / retrieved_anchor_count
            ),
        }

    summary = {
        "official_rows": len(frame),
        "outer_evaluation": "five reconstructed-source-disjoint folds",
        "boundaries": [name for name, _ in BOUNDARIES],
        "anchor_retrieval": "outer-train-only character TF-IDF; target labels unused",
        "anchor_records_contain_gold_labels": False,
        "target_view": "full response plus three input-only event candidates",
        "folds": summary_folds,
        "boundary_class_counts_across_all_fold_copies": dict(global_counts),
        "anchor_source_groups_by_boundary": {
            boundary: len(groups) for boundary, groups in anchor_source_counts.items()
        },
        "old_cumulative_head_difference": (
            "independent anchor-conditioned boundary comparisons rather than one "
            "pooled scalar with four shared cutpoints"
        ),
        "current_or_hidden_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
