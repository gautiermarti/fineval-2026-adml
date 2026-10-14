#!/usr/bin/env python3
"""Train issuer-held-out ML-Promise detectors and featurize JF-ICR inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_events import event_candidates  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def candidate_is_promise(candidate: str, promise: str) -> bool:
    left, right = normalize(candidate), normalize(promise)
    return bool(left and right and (left in right or right in left))


def row_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    weights = np.asarray([1.0 / counts[group] for group in groups])
    return weights / weights.mean()


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2,
        max_features=80_000, sublinear_tf=True,
    )


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=3000,
        solver="liblinear", random_state=20260805,
    )


def auxiliary_candidates(rows: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    texts, labels, owners, hosts = [], [], [], []
    for owner, row in enumerate(rows):
        candidates = event_candidates("", row["text"])
        if not candidates:
            candidates = [type("Candidate", (), {"text": row["text"]})()]
        for candidate in candidates:
            texts.append(candidate.text)
            labels.append(int(candidate_is_promise(candidate.text, row["promise_span"])))
            owners.append(owner)
            hosts.append(row["source_host"])
    return texts, np.asarray(labels), np.asarray(owners), np.asarray(hosts)


def train_row_model(
    texts: list[str], labels: np.ndarray, hosts: np.ndarray,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    model = make_classifier()
    model.fit(matrix, labels, sample_weight=row_weights(hosts))
    return vectorizer, model


def train_span_model(
    texts: list[str], labels: np.ndarray, owners: np.ndarray, hosts: np.ndarray,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    owner_counts = Counter(owners.tolist())
    source_counts = Counter(hosts.tolist())
    weights = np.asarray([
        1.0 / owner_counts[int(owner)] / source_counts[str(host)]
        for owner, host in zip(owners, hosts, strict=True)
    ])
    weights /= weights.mean()
    model = make_classifier()
    model.fit(matrix, labels, sample_weight=weights)
    return vectorizer, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auxiliary", type=Path,
        default=ROOT / "data/external/ml_promise_japanese_safe/auxiliary_train.jsonl",
    )
    parser.add_argument(
        "--jf-icr", type=Path,
        default=ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/ml_promise_japanese_transfer_features_v1",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.auxiliary)
    texts = [row["text"] for row in rows]
    labels = np.asarray([row["promise_status"] == "Yes" for row in rows], dtype=int)
    hosts = np.asarray([row["source_host"] for row in rows])
    candidate_texts, candidate_labels, candidate_owners, candidate_hosts = auxiliary_candidates(rows)
    positive_rows_without_candidate = [
        index for index, label in enumerate(labels)
        if label and not np.any(candidate_labels[candidate_owners == index])
    ]
    if positive_rows_without_candidate:
        raise ValueError(
            f"positive spans not represented by event candidates: {positive_rows_without_candidate}"
        )

    row_oof = np.full(len(rows), np.nan)
    span_scores = np.full(len(candidate_texts), np.nan)
    fold_rows = []
    for host in sorted(set(hosts)):
        row_train, row_valid = hosts != host, hosts == host
        row_vectorizer, row_model = train_row_model(
            [text for text, keep in zip(texts, row_train, strict=True) if keep],
            labels[row_train], hosts[row_train],
        )
        row_oof[row_valid] = row_model.predict_proba(
            row_vectorizer.transform([
                text for text, keep in zip(texts, row_valid, strict=True) if keep
            ])
        )[:, 1]

        candidate_train = candidate_hosts != host
        candidate_valid = ~candidate_train
        span_vectorizer, span_model = train_span_model(
            [text for text, keep in zip(candidate_texts, candidate_train, strict=True) if keep],
            candidate_labels[candidate_train],
            candidate_owners[candidate_train], candidate_hosts[candidate_train],
        )
        span_scores[candidate_valid] = span_model.predict_proba(
            span_vectorizer.transform([
                text for text, keep in zip(candidate_texts, candidate_valid, strict=True) if keep
            ])
        )[:, 1]
        fold_rows.append({
            "held_out_host": host,
            "rows": int(row_valid.sum()),
            "promise_rate": float(labels[row_valid].mean()),
            "row_balanced_accuracy_at_0.5": float(balanced_accuracy_score(
                labels[row_valid], row_oof[row_valid] >= 0.5,
            )),
            "row_macro_f1_at_0.5": float(f1_score(
                labels[row_valid], row_oof[row_valid] >= 0.5, average="macro",
            )),
        })
    if np.isnan(row_oof).any() or np.isnan(span_scores).any():
        raise AssertionError("incomplete auxiliary OOF predictions")

    top1_hits, reciprocal_ranks = [], []
    top_span_scores = np.zeros(len(rows))
    for owner, label in enumerate(labels):
        positions = np.flatnonzero(candidate_owners == owner)
        ordered = positions[np.argsort(-span_scores[positions], kind="stable")]
        top_span_scores[owner] = span_scores[ordered[0]]
        if label:
            relevance = candidate_labels[ordered]
            top1_hits.append(bool(relevance[0]))
            reciprocal_ranks.append(1.0 / (int(np.flatnonzero(relevance)[0]) + 1))

    row_vectorizer, row_model = train_row_model(texts, labels, hosts)
    span_vectorizer, span_model = train_span_model(
        candidate_texts, candidate_labels, candidate_owners, candidate_hosts,
    )

    # Do not load the JF-ICR answer column. These transferred features are
    # fixed before any source-disjoint target evaluation.
    jf_frame = pd.read_parquet(args.jf_icr, columns=["id", "query"])
    feature_rows = []
    for row in jf_frame.itertuples(index=False):
        parsed = parse_icr_query(row.query)
        candidates = event_candidates(parsed.question, parsed.response)
        if not candidates:
            raise ValueError(f"no event candidates for JF-ICR id {row.id}")
        scores = span_model.predict_proba(
            span_vectorizer.transform([candidate.text for candidate in candidates])
        )[:, 1]
        overlap = np.asarray([candidate.question_overlap for candidate in candidates])
        joint = scores * (0.5 + overlap)
        promise_order = np.argsort(-scores, kind="stable")
        joint_order = np.argsort(-joint, kind="stable")
        top_three = np.sort(scores)[-3:]
        feature_rows.append({
            "id": int(row.id),
            "external_row_promise_probability": float(row_model.predict_proba(
                row_vectorizer.transform([parsed.response])
            )[0, 1]),
            "external_max_event_promise_probability": float(scores.max()),
            "external_mean_top3_event_probability": float(top_three.mean()),
            "external_tail_event_probability": float(scores[-1]),
            "external_high_probability_event_count": int(np.sum(scores >= 0.5)),
            "external_top_event_relative_position": float(
                candidates[int(promise_order[0])].start / max(len(parsed.response), 1)
            ),
            "external_top_promise_event": candidates[int(promise_order[0])].text,
            "external_top_joint_event": candidates[int(joint_order[0])].text,
            "external_top_joint_score": float(joint[int(joint_order[0])]),
            "event_candidate_count": len(candidates),
        })

    summary = {
        "auxiliary_rows": len(rows),
        "auxiliary_hosts": len(set(hosts)),
        "auxiliary_promise_rate": float(labels.mean()),
        "auxiliary_event_candidates": len(candidate_texts),
        "auxiliary_positive_event_candidates": int(candidate_labels.sum()),
        "evaluation": "leave-one-issuer-out on ML-Promise training annotations",
        "row_roc_auc": float(roc_auc_score(labels, row_oof)),
        "row_balanced_accuracy_at_0.5": float(balanced_accuracy_score(labels, row_oof >= 0.5)),
        "row_macro_f1_at_0.5": float(f1_score(labels, row_oof >= 0.5, average="macro")),
        "positive_row_top1_span_recall": float(np.mean(top1_hits)),
        "positive_row_span_mrr": float(np.mean(reciprocal_ranks)),
        "jf_icr_rows_featurized": len(feature_rows),
        "jf_icr_columns_loaded": ["id", "query"],
        "jf_icr_labels_used": False,
        "ml_promise_test_used": False,
        "current_or_hidden_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "original_index": [row["original_index"] for row in rows],
        "source_host": hosts,
        "gold_promise": labels,
        "oof_row_probability": row_oof,
        "oof_top_span_probability": top_span_scores,
    }).to_csv(args.output_dir / "auxiliary_oof_rows.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "auxiliary_folds.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(args.output_dir / "jf_icr_features.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
