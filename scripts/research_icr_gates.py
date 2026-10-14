#!/usr/bin/env python3
"""Research precision-first minority-label gates for the JF-ICR majority system."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402

TARGETS = ["0", "+2", "-1", "-2"]


def text_views(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    parsed = frame["query"].map(parse_icr_query)
    question = np.asarray([item.question for item in parsed])
    response = np.asarray([item.response for item in parsed])
    combined = np.asarray([
        f"質問: {item.question}\n回答: {item.response}" for item in parsed
    ])
    # The end of an IR answer frequently contains the decisive modality.
    response_tail = np.asarray([text[-500:] for text in response])
    return {
        "combined": combined,
        "response": response,
        "response_tail": response_tail,
        "question": question,
    }


def estimator(c: float) -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(2, 5), min_df=2,
                max_features=100_000, sublinear_tf=True,
            )),
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=50_000, sublinear_tf=True,
            )),
        ])),
        ("classifier", LogisticRegression(
            C=c, max_iter=3_000, class_weight="balanced", random_state=2026,
        )),
    ])


def cross_fitted_probabilities(
    texts: np.ndarray, labels: np.ndarray, folds: np.ndarray, target: str, c: float,
) -> np.ndarray:
    probabilities = np.zeros(len(labels), dtype=float)
    binary = labels == target
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        if len(np.unique(binary[train])) < 2:
            probabilities[test] = 0.0
            continue
        fitted = estimator(c).fit(texts[train], binary[train])
        positive_column = int(np.flatnonzero(fitted.classes_)[0])
        probabilities[test] = fitted.predict_proba(texts[test])[:, positive_column]
    return probabilities


def frontier(
    labels: np.ndarray, probabilities: np.ndarray, target: str, split: str,
    scheme: str, view: str, c: float,
) -> list[dict]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        selected = probabilities >= threshold
        predictions = np.full(len(labels), "+1", dtype="<U2")
        predictions[selected] = target
        true_positive = int(np.sum(selected & (labels == target)))
        rows.append({
            "scheme": scheme, "split": split, "view": view, "C": c,
            "target": target, "threshold": threshold,
            "accuracy": accuracy_score(labels, predictions),
            "selected": int(selected.sum()),
            "correct_overrides": true_positive,
            "override_precision": true_positive / selected.sum() if selected.any() else 0.0,
            "net_gain": int(np.sum(predictions == labels) - np.sum(labels == "+1")),
        })
    return rows


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    labels = frame["answer"].astype(str).to_numpy()
    views = text_views(frame)
    source = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    semantic = pd.read_csv(ROOT / "data/derived/icr_semantic_folds.csv").set_index("id")
    fold_schemes = {
        "source": source.loc[frame["id"], "source_fold"].to_numpy(),
        "semantic": semantic.loc[frame["id"], "fold"].to_numpy(),
    }
    probability_table = pd.DataFrame({"id": frame["id"], "gold": labels})
    results = []
    for scheme, folds in fold_schemes.items():
        for view in ("combined", "response", "response_tail", "question"):
            for c in (0.1, 0.3, 1.0, 3.0, 10.0):
                for target in TARGETS:
                    probabilities = cross_fitted_probabilities(
                        views[view], labels, folds, target, c,
                    )
                    column = f"{scheme}__{view}__c{c:g}__{target}"
                    probability_table[column] = probabilities
                    for split, mask in (
                        ("all", np.ones(len(frame), dtype=bool)),
                        ("core", frame["id"].to_numpy() <= 93),
                        ("expanded", frame["id"].to_numpy() > 93),
                    ):
                        results.extend(frontier(
                            labels[mask], probabilities[mask], target, split,
                            scheme, view, c,
                        ))
                    print(scheme, view, c, target, flush=True)
    output_dir = ROOT / "artifacts/icr_gates"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(results)
    scores.to_csv(output_dir / "binary_gate_frontiers.csv", index=False)
    probability_table.to_csv(output_dir / "oof_probabilities.csv", index=False)

    viable = scores[
        (scores["selected"] >= 2)
        & (scores["override_precision"] >= 0.60)
        & (scores["net_gain"] > 0)
    ].sort_values(
        ["scheme", "split", "net_gain", "override_precision"],
        ascending=[True, True, False, False],
    )
    viable.groupby(["scheme", "split"], group_keys=False).head(12).to_csv(
        output_dir / "best_binary_gates.csv", index=False,
    )
    print("\nBEST GATES")
    print(viable.groupby(["scheme", "split"], group_keys=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
