#!/usr/bin/env python3
"""Source-disjoint CPU diagnostic for input-only decisive-event views."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_events import event_candidates  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402


LABELS = ("-2", "-1", "0", "+1", "+2")


def view_text(question: str, response: str, view: str) -> str:
    candidates = event_candidates(question, response)
    if view == "full_response":
        body = response
    elif view == "top1_event":
        body = candidates[0].text
    elif view == "top3_events":
        chosen = sorted(candidates[:3], key=lambda row: (row.start, row.end))
        body = " ".join(row.text for row in chosen)
    else:
        raise ValueError(f"unknown view {view}")
    return question + " [Q-A] " + body


def source_equal_weights(groups: pd.Series) -> np.ndarray:
    counts = groups.value_counts()
    weights = np.asarray([1 / counts[value] for value in groups], dtype=float)
    return weights / weights.mean()


def accuracy_summary(gold: list[str], prediction: list[str], folds: list[int]) -> dict:
    per_fold = {}
    for fold in sorted(set(folds)):
        indices = [index for index, value in enumerate(folds) if value == fold]
        per_fold[str(fold)] = {
            "rows": len(indices),
            "accuracy": sum(gold[index] == prediction[index] for index in indices) / len(indices),
        }
    matrix = confusion_matrix(gold, prediction, labels=list(LABELS))
    recalls = {
        label: (
            float(matrix[index, index] / matrix[index].sum())
            if matrix[index].sum() else 0.0
        )
        for index, label in enumerate(LABELS)
    }
    return {
        "rows": len(gold),
        "accuracy": sum(left == right for left, right in zip(gold, prediction)) / len(gold),
        "folds": per_fold,
        "class_recall": recalls,
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": matrix.tolist(),
        "predicted_label_counts": dict(Counter(prediction)),
    }


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
        default=ROOT / "artifacts/icr_event_views_v1",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).sort_values("id").reset_index(drop=True)
    metadata = (
        pd.read_csv(args.folds).set_index("id").loc[frame.id].reset_index(drop=True)
    )
    parsed = [parse_icr_query(query) for query in frame["query"]]
    views = {
        view: [view_text(row.question, row.response, view) for row in parsed]
        for view in ("full_response", "top1_event", "top3_events")
    }
    summaries = {}
    output_rows = []
    for view, texts in views.items():
        predictions = np.empty(len(frame), dtype=object)
        for fold in sorted(metadata.source_fold.unique()):
            train_mask = metadata.source_fold.to_numpy() != fold
            valid_mask = ~train_mask
            vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(2, 5), min_df=2,
                max_features=60_000, sublinear_tf=True,
            )
            train_matrix = vectorizer.fit_transform(
                [text for text, keep in zip(texts, train_mask) if keep]
            )
            valid_matrix = vectorizer.transform(
                [text for text, keep in zip(texts, valid_mask) if keep]
            )
            model = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=3000,
                solver="lbfgs", random_state=7,
            )
            model.fit(
                train_matrix, frame.loc[train_mask, "answer"],
                sample_weight=source_equal_weights(
                    metadata.loc[train_mask, "source_group"]
                ),
            )
            predictions[valid_mask] = model.predict(valid_matrix)
        gold = frame.answer.astype(str).tolist()
        predicted = predictions.tolist()
        summaries[view] = accuracy_summary(
            gold, predicted, metadata.source_fold.astype(int).tolist(),
        )
        output_rows.extend(
            {
                "id": int(identifier), "view": view, "gold": target,
                "prediction": guess, "source_fold": int(fold),
            }
            for identifier, target, guess, fold in zip(
                frame.id, gold, predicted, metadata.source_fold,
            )
        )
    summary = {
        "evaluation": "five reconstructed-source-disjoint folds",
        "event_ranking": "input only; no label or held-fold statistic",
        "model": "source-equal-weighted character TF-IDF logistic diagnostic",
        "views": summaries,
        "incumbent_neural_accuracy": 0.691699604743083,
        "end_to_end_promotion_claim": False,
        "current_or_hidden_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
