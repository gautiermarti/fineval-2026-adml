#!/usr/bin/env python3
"""Leakage-aware diagnostic baselines for JF-ICR public data."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


LABELS = ["-2", "-1", "0", "+1", "+2"]


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    parsed = frame["query"].map(parse_icr_query)
    # Preserve the question because commitment is relative to what was asked.
    text = [f"質問: {x.question}\n回答: {x.response}" for x in parsed]
    labels = frame["answer"].to_numpy()

    features = FeatureUnion(
        [
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=80_000)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=40_000)),
        ]
    )
    model = Pipeline(
        [
            ("features", features),
            ("classifier", LogisticRegression(C=2.0, max_iter=2_000)),
        ]
    )
    # Two folds are required because the rarest class has only two examples.
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=2026)
    text = np.asarray(text)
    fold_scores = []
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    prediction_counts: dict[str, int] = {}
    for train, test in cv.split(text, labels):
        fitted = clone(model).fit(text[train], labels[train])
        predictions = fitted.predict(text[test])
        fold_scores.append(accuracy_score(labels[test], predictions))
        matrix += confusion_matrix(labels[test], predictions, labels=LABELS)
        for label, count in zip(*np.unique(predictions, return_counts=True), strict=True):
            prediction_counts[label] = prediction_counts.get(label, 0) + int(count)
    print(f"repeated_stratified_accuracy_mean={np.mean(fold_scores):.6f}")
    print(f"repeated_stratified_accuracy_std={np.std(fold_scores):.6f}")
    print("labels", LABELS)
    print("confusion_matrix")
    print(matrix)
    print("prediction_counts", prediction_counts)
    print("WARNING: this is a diagnostic random split, not the system promotion score.")

    fold_path = ROOT / "data/derived/icr_semantic_folds.csv"
    if fold_path.exists():
        fold_frame = pd.read_csv(fold_path).set_index("id").loc[frame["id"]]
        semantic_scores = []
        semantic_matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
        for fold in sorted(fold_frame["fold"].unique()):
            test = np.flatnonzero(fold_frame["fold"].to_numpy() == fold)
            train = np.flatnonzero(fold_frame["fold"].to_numpy() != fold)
            predictions = clone(model).fit(text[train], labels[train]).predict(text[test])
            semantic_scores.append(accuracy_score(labels[test], predictions))
            semantic_matrix += confusion_matrix(labels[test], predictions, labels=LABELS)
        print(f"semantic_group_accuracy_mean={np.mean(semantic_scores):.6f}")
        print(f"semantic_group_accuracy_weighted={np.trace(semantic_matrix) / semantic_matrix.sum():.6f}")
        print("semantic_group_confusion_matrix")
        print(semantic_matrix)

    source_path = ROOT / "data/derived/icr_source_folds.csv"
    if source_path.exists():
        source_frame = pd.read_csv(source_path, dtype={"gold_label": str}).set_index("id").loc[frame["id"]]
        source_scores = []
        source_matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
        source_folds = source_frame["source_fold"].to_numpy()
        for fold in sorted(np.unique(source_folds)):
            test = np.flatnonzero(source_folds == fold)
            train = np.flatnonzero(source_folds != fold)
            predictions = clone(model).fit(text[train], labels[train]).predict(text[test])
            source_scores.append(accuracy_score(labels[test], predictions))
            source_matrix += confusion_matrix(labels[test], predictions, labels=LABELS)
        print(f"source_group_accuracy_mean={np.mean(source_scores):.6f}")
        print(f"source_group_accuracy_weighted={np.trace(source_matrix) / source_matrix.sum():.6f}")
        print("source_group_confusion_matrix")
        print(source_matrix)


if __name__ == "__main__":
    main()
