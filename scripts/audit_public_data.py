#!/usr/bin/env python3
"""Profile the immutable July 28 public datasets."""

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.metrics import accuracy, rouge1_f1  # noqa: E402


def audit_subtask2() -> None:
    path = ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    frame = pd.read_parquet(path)
    counts = frame["answer"].value_counts()
    majority = counts.idxmax()
    predictions = [majority] * len(frame)
    print("SUBTASK 2")
    print(f"rows={len(frame)} columns={list(frame.columns)}")
    print(f"labels={counts.sort_index().to_dict()}")
    print(f"majority_label={majority} majority_accuracy={accuracy(frame['answer'].tolist(), predictions):.6f}")
    print(f"duplicate_queries={frame.duplicated('query').sum()} unique_ids={frame['id'].nunique()}")

    pattern = re.compile(
        r"Financial Question:\s*(.*?)\s*Company Response:\s*(.*?)\s*Directly output",
        re.DOTALL,
    )
    parsed = frame["query"].str.extract(pattern)
    print(f"parsed_question_response={parsed.notna().all(axis=1).sum()}/{len(frame)}")
    print()


def audit_subtask3() -> None:
    path = ROOT / "data/raw/subtask3/public-00000-of-00001.parquet"
    frame = pd.read_parquet(path)
    identity = rouge1_f1(frame["answer"].tolist(), frame["answer"].tolist())
    print("SUBTASK 3")
    print(f"rows={len(frame)} columns={list(frame.columns)}")
    print(f"document_bundles={frame['task_id'].nunique()}")
    print(f"questions_per_bundle={frame.groupby('task_id').size().value_counts().sort_index().to_dict()}")
    print(f"unique_question_templates={frame['question'].nunique()}")
    print(f"easy_format_queries={frame['query'].str.contains('News Evidence:', regex=False).sum()}")
    print(f"expert_format_queries={frame['query'].str.contains('Financial Statements Evidence:', regex=False).sum()}")
    print(f"identity_rouge1_f1={identity:.6f}")
    print(f"exact_duplicate_answers={frame.duplicated('answer').sum()}")


if __name__ == "__main__":
    audit_subtask2()
    audit_subtask3()

