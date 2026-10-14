#!/usr/bin/env python3
"""Nested leave-company-out risk model for sparse PolyFiQA edits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.metrics import POLYFIQA_TOKEN_RE  # noqa: E402
from fineval.polyfiqa_family_gate import exact_cluster_bootstrap  # noqa: E402
from fineval.polyfiqa_edits import split_answer  # noqa: E402


COMPANIES = ("HON", "JNJ", "MSFT", "UVV")
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02)
NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,.]*(?:%|[BMKbm])?")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def company(record: dict) -> str:
    value = str(record["internal_id"]).split("_", 1)[0]
    if value not in COMPANIES:
        raise ValueError(f"unexpected company: {value}")
    return value


def tokens(text: str) -> list[str]:
    return POLYFIQA_TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def coverage(answer: str, source: str) -> float:
    answer_tokens = Counter(tokens(answer))
    source_tokens = Counter(tokens(source))
    total = sum(answer_tokens.values())
    return sum((answer_tokens & source_tokens).values()) / total if total else 0.0


def feature_record(row: dict, operation: str, candidate: str, source: str) -> tuple[str, dict]:
    baseline = row["candidates"]["baseline"]
    baseline_core, baseline_evidence = split_answer(baseline)
    candidate_core, candidate_evidence = split_answer(candidate)
    base_tokens, candidate_tokens = tokens(baseline), tokens(candidate)
    common = sum((Counter(base_tokens) & Counter(candidate_tokens)).values())
    union = sum((Counter(base_tokens) | Counter(candidate_tokens)).values())
    operation_family, _, operation_variant = operation.partition(":")
    numeric = {
        f"tier={row['tier']}": 1,
        f"family={row['family']}": 1,
        f"operation={operation_family}": 1,
        f"variant={operation_variant or 'none'}": 1,
        f"operation_variant={operation}": 1,
        "baseline_tokens": len(base_tokens) / 100.0,
        "candidate_tokens": len(candidate_tokens) / 100.0,
        "token_delta": (len(candidate_tokens) - len(base_tokens)) / 100.0,
        "token_jaccard": common / union if union else 1.0,
        "baseline_numbers": len(NUMBER_RE.findall(baseline)) / 10.0,
        "candidate_numbers": len(NUMBER_RE.findall(candidate)) / 10.0,
        "number_delta": (
            len(NUMBER_RE.findall(candidate)) - len(NUMBER_RE.findall(baseline))
        ) / 10.0,
        "baseline_core_tokens": len(tokens(baseline_core)) / 100.0,
        "candidate_core_tokens": len(tokens(candidate_core)) / 100.0,
        "baseline_evidence_tokens": len(tokens(baseline_evidence)) / 100.0,
        "candidate_evidence_tokens": len(tokens(candidate_evidence)) / 100.0,
        "baseline_none_evidence": int("Evidence: None" in baseline_evidence),
        "candidate_none_evidence": int("Evidence: None" in candidate_evidence),
        "baseline_source_coverage": coverage(baseline, source),
        "candidate_source_coverage": coverage(candidate, source),
    }
    text = (
        f"{row['tier']} {row['family']} [OP] {operation} "
        f"[QUESTION] {row['question']} [BASE] {baseline} [CANDIDATE] {candidate}"
    )
    return text, numeric


def flatten(records: list[dict], sources: dict[int, str]) -> list[dict]:
    examples = []
    for row in records:
        baseline_score = float(row["scores"]["baseline"])
        for operation, candidate in row["candidates"].items():
            if operation == "baseline":
                continue
            text, numeric = feature_record(
                row, operation, candidate, sources[int(row["row_index"])],
            )
            examples.append({
                "row_index": int(row["row_index"]), "company": company(row),
                "operation": operation, "answer": candidate,
                "text": text, "numeric": numeric,
                "delta": float(row["scores"][operation]) - baseline_score,
            })
    return examples


def fit_predict(train: list[dict], test: list[dict], alpha: float) -> np.ndarray:
    text_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        max_features=50_000, sublinear_tf=True,
    )
    numeric_vectorizer = DictVectorizer()
    train_text = text_vectorizer.fit_transform([row["text"] for row in train])
    test_text = text_vectorizer.transform([row["text"] for row in test])
    train_numeric = numeric_vectorizer.fit_transform([row["numeric"] for row in train])
    test_numeric = numeric_vectorizer.transform([row["numeric"] for row in test])
    train_matrix = hstack([train_text, train_numeric], format="csr")
    test_matrix = hstack([test_text, test_numeric], format="csr")
    row_counts = Counter((row["company"], row["row_index"]) for row in train)
    # Each historical row and each company carry equal total training mass.
    unique_rows_by_company = Counter()
    for key in row_counts:
        unique_rows_by_company[key[0]] += 1
    sample_weight = np.asarray([
        1.0 / (
            unique_rows_by_company[row["company"]]
            * row_counts[(row["company"], row["row_index"])]
        )
        for row in train
    ])
    model = Ridge(alpha=alpha, solver="lsqr").fit(
        train_matrix, np.asarray([row["delta"] for row in train]),
        sample_weight=sample_weight,
    )
    return model.predict(test_matrix)


def select_rows(
    row_records: list[dict], examples: list[dict], predictions: np.ndarray,
    threshold: float,
) -> list[dict]:
    by_row = defaultdict(list)
    for example, predicted_gain in zip(examples, predictions, strict=True):
        by_row[example["row_index"]].append((float(predicted_gain), example))
    selected = []
    for row in row_records:
        row_index = int(row["row_index"])
        choices = [item for item in by_row[row_index] if item[0] >= threshold]
        if choices:
            predicted_gain, example = max(
                choices, key=lambda item: (item[0], -len(item[1]["answer"])),
            )
            operation = example["operation"]
        else:
            predicted_gain, operation = 0.0, "baseline"
        baseline_score = float(row["scores"]["baseline"])
        selected_score = float(row["scores"][operation])
        selected.append({
            "row_index": row_index, "task_id": row["task_id"],
            "held_company": company(row), "tier": row["tier"],
            "family": row["family"], "selected": operation,
            "predicted_gain": predicted_gain,
            "baseline_score": baseline_score, "selected_score": selected_score,
            "delta": selected_score - baseline_score,
            "answer": row["candidates"][operation],
        })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lattice", type=Path,
        default=ROOT / "artifacts/polyfiqa_sparse_edit_oracle_v1/lattice.jsonl",
    )
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/finmmeval_task2/task2_test_questions.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_sparse_edit_risk_model_v1",
    )
    args = parser.parse_args()

    records = load_jsonl(args.lattice)
    input_rows = load_jsonl(args.input)
    sources = {
        index: f"{row['question']}\n{row['query']}" for index, row in enumerate(input_rows)
    }
    for row in records:
        row["question"] = input_rows[int(row["row_index"])]["question"]
    examples = flatten(records, sources)
    all_selected = []
    fold_rows = []
    for held_company in COMPANIES:
        outer_train_records = [row for row in records if company(row) != held_company]
        outer_test_records = [row for row in records if company(row) == held_company]
        candidate_scores = []
        for alpha in ALPHAS:
            inner_predictions = []
            for inner_company in sorted(set(COMPANIES) - {held_company}):
                inner_train = [
                    row for row in examples
                    if row["company"] not in {held_company, inner_company}
                ]
                inner_test = [
                    row for row in examples if row["company"] == inner_company
                ]
                prediction = fit_predict(inner_train, inner_test, alpha)
                inner_predictions.extend(zip(inner_test, prediction, strict=True))
            inner_examples = [item[0] for item in inner_predictions]
            predicted = np.asarray([item[1] for item in inner_predictions])
            for threshold in THRESHOLDS:
                selected = select_rows(
                    outer_train_records, inner_examples, predicted, threshold,
                )
                candidate_scores.append({
                    "alpha": alpha, "threshold": threshold,
                    "score": sum(row["selected_score"] for row in selected) / len(selected),
                    "delta": sum(row["delta"] for row in selected) / len(selected),
                    "changed": sum(row["selected"] != "baseline" for row in selected),
                })
        best_score = max(row["score"] for row in candidate_scores)
        tied = [row for row in candidate_scores if row["score"] == best_score]
        chosen = max(tied, key=lambda row: (row["threshold"], row["alpha"]))
        outer_train = [row for row in examples if row["company"] != held_company]
        outer_test = [row for row in examples if row["company"] == held_company]
        predictions = fit_predict(outer_train, outer_test, chosen["alpha"])
        selected = select_rows(
            outer_test_records, outer_test, predictions, chosen["threshold"],
        )
        all_selected.extend(selected)
        fold_rows.append({
            "held_company": held_company,
            "selected_alpha": chosen["alpha"],
            "selected_threshold": chosen["threshold"],
            "inner_score": chosen["score"],
            "inner_delta": chosen["delta"],
            "inner_changed": chosen["changed"],
            "rows": len(selected),
            "changed": sum(row["selected"] != "baseline" for row in selected),
            "baseline": sum(row["baseline_score"] for row in selected) / len(selected),
            "selected": sum(row["selected_score"] for row in selected) / len(selected),
            "delta": sum(row["delta"] for row in selected) / len(selected),
        })
    all_selected.sort(key=lambda row: row["row_index"])
    bootstrap = exact_cluster_bootstrap(
        all_selected, cluster_key="held_company", delta_key="delta",
    )
    baseline = sum(row["baseline_score"] for row in all_selected) / len(all_selected)
    selected_score = sum(row["selected_score"] for row in all_selected) / len(all_selected)
    summary = {
        "evaluation": "nested leave-company-out sparse-edit ridge risk model",
        "rows": len(all_selected), "baseline_rouge": baseline,
        "selected_rouge": selected_score, "delta": selected_score - baseline,
        "changed_rows": sum(row["selected"] != "baseline" for row in all_selected),
        "improved_rows": sum(row["delta"] > 0 for row in all_selected),
        "harmed_rows": sum(row["delta"] < 0 for row in all_selected),
        "folds": fold_rows, "company_cluster_bootstrap": bootstrap,
        "nested_hyperparameter_selection": True,
        "current_hidden_answers_used": False,
    }
    summary["promotion_eligible"] = bool(
        summary["delta"] > 0
        and sum(row["delta"] >= 0 for row in fold_rows) >= 3
        and sum(row["delta"] < -0.002 for row in fold_rows) <= 1
        and bootstrap["p_gain"] >= 0.90
        and bootstrap["p05"] >= 0.0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "folds.csv", index=False)
    pd.DataFrame(all_selected).drop(columns=["answer"]).to_csv(
        args.output_dir / "audit.csv", index=False,
    )
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({"task_id": row["task_id"], "answer": row["answer"]}, ensure_ascii=False) + "\n"
        for row in all_selected
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
