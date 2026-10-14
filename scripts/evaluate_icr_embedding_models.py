#!/usr/bin/env python3
"""Evaluate lightweight embedding classifiers on fixed group-held-out folds."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["-2", "-1", "0", "+1", "+2"]
ORDINAL_VALUES = {label: index for index, label in enumerate(LABELS)}


def optimal_thresholds(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit ordered cut points that maximize training accuracy by dynamic programming."""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    targets = np.array([ORDINAL_VALUES[label] for label in labels[order]])
    n = len(scores)
    classes = len(LABELS)
    prefix = np.zeros((classes, n + 1), dtype=int)
    for cls in range(classes):
        prefix[cls, 1:] = np.cumsum(targets == cls)
    dp = np.full((classes + 1, n + 1), -10**9, dtype=int)
    previous = np.full((classes + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0
    for used in range(1, classes + 1):
        cls = used - 1
        for end in range(n + 1):
            for start in range(end + 1):
                value = dp[used - 1, start] + prefix[cls, end] - prefix[cls, start]
                if value > dp[used, end]:
                    dp[used, end] = value
                    previous[used, end] = start
    cuts = []
    end = n
    for used in range(classes, 1, -1):
        start = previous[used, end]
        cuts.append(start)
        end = start
    cuts.reverse()
    thresholds = []
    for cut in cuts:
        if cut == 0:
            thresholds.append(sorted_scores[0] - 1e-6)
        elif cut == n:
            thresholds.append(sorted_scores[-1] + 1e-6)
        else:
            thresholds.append((sorted_scores[cut - 1] + sorted_scores[cut]) / 2)
    return np.asarray(thresholds)


def evaluate_ordinal(
    embeddings: np.ndarray, labels: np.ndarray, folds: np.ndarray, *, weighted: bool
) -> np.ndarray:
    predictions = np.empty(len(labels), dtype="<U2")
    numeric = np.array([ORDINAL_VALUES[label] for label in labels], dtype=float)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        model = Ridge(alpha=10.0)
        sample_weight = None
        if weighted:
            counts = pd.Series(labels[train]).value_counts()
            sample_weight = np.array([len(counts) / (len(counts) * counts[label]) for label in labels[train]])
        model.fit(embeddings[train], numeric[train], sample_weight=sample_weight)
        train_scores = model.predict(embeddings[train])
        thresholds = optimal_thresholds(train_scores, labels[train])
        indices = np.digitize(model.predict(embeddings[test]), thresholds)
        predictions[test] = np.asarray(LABELS)[indices]
    return predictions


def models() -> dict:
    return {
        "logistic": LogisticRegression(C=1.0, max_iter=3_000),
        "logistic_balanced": LogisticRegression(C=0.3, max_iter=3_000, class_weight="balanced"),
        "ridge": RidgeClassifier(alpha=10.0),
        "ridge_balanced": RidgeClassifier(alpha=10.0, class_weight="balanced"),
        # Embeddings are L2-normalized, so Euclidean ranking is equivalent to
        # cosine ranking for nearest-neighbor geometry.
        "nearest_centroid": NearestCentroid(metric="euclidean"),
        "knn_7": KNeighborsClassifier(n_neighbors=7, metric="cosine", weights="distance"),
        "rbf_svc": SVC(C=2.0, gamma="scale"),
        "rbf_svc_balanced": SVC(C=1.0, gamma="scale", class_weight="balanced"),
        "lda_shrinkage": make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--embeddings", type=Path,
        default=ROOT / "data/derived/icr_multilingual_embeddings.npy",
    )
    parser.add_argument(
        "--assignments", type=Path,
        default=ROOT / "data/derived/icr_semantic_folds.csv",
    )
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--max-id", type=int)
    parser.add_argument("--tag", default="default")
    args = parser.parse_args()
    embeddings = np.load(args.embeddings)
    assignments = pd.read_csv(args.assignments, dtype={"gold_label": str})
    if "gold_label" not in assignments:
        assignments["gold_label"] = assignments["answer"].astype(str)
    if args.max_id is not None:
        keep = assignments["id"] <= args.max_id
        embeddings = embeddings[keep.to_numpy()]
        assignments = assignments.loc[keep].reset_index(drop=True)
    labels = assignments["gold_label"].astype(str).to_numpy(dtype=str)
    folds = assignments[args.fold_column].to_numpy()
    rows = []
    matrices = {}
    prediction_table = pd.DataFrame({"id": assignments["id"], "gold_label": labels, "fold": folds})
    for name, estimator in models().items():
        predictions = np.empty(len(labels), dtype="<U2")
        for fold in sorted(np.unique(folds)):
            train = folds != fold
            test = folds == fold
            predictions[test] = np.asarray(
                clone(estimator).fit(embeddings[train], labels[train]).predict(embeddings[test]),
                dtype=str,
            )
        matrix = confusion_matrix(labels, predictions, labels=LABELS)
        matrices[name] = matrix
        prediction_table[name] = predictions
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(labels, predictions),
                "macro_f1": f1_score(labels, predictions, labels=LABELS, average="macro", zero_division=0),
                **{
                    f"recall_{label}": matrix[i, i] / matrix[i].sum() if matrix[i].sum() else 0
                    for i, label in enumerate(LABELS)
                },
            }
        )
    for name, weighted in [("ordinal_ridge", False), ("ordinal_ridge_balanced", True)]:
        predictions = evaluate_ordinal(embeddings, labels, folds, weighted=weighted)
        matrix = confusion_matrix(labels, predictions, labels=LABELS)
        matrices[name] = matrix
        prediction_table[name] = predictions
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(labels, predictions),
                "macro_f1": f1_score(labels, predictions, labels=LABELS, average="macro", zero_division=0),
                **{
                    f"recall_{label}": matrix[i, i] / matrix[i].sum() if matrix[i].sum() else 0
                    for i, label in enumerate(LABELS)
                },
            }
        )
    results = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False)
    output_dir = ROOT / "artifacts/icr_embedding_models" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "summary.csv", index=False)
    prediction_table.to_csv(output_dir / "oof_predictions.csv", index=False)
    with (output_dir / "confusion_matrices.txt").open("w") as handle:
        for name, matrix in matrices.items():
            handle.write(f"{name}\nlabels={LABELS}\n{matrix}\n\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
