#!/usr/bin/env python3
"""Audit PolyFiQA reference formatting and numerical provenance."""

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from fineval.tables import parse_markdown_tables  # noqa: E402


QUESTION_TYPES = ["revenue", "balance", "cash_flow", "rd_ratio"]


def answer_numbers(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?", text)


def classify_format(answer: str) -> str:
    if "News Evidence:" in answer:
        return "news_evidence"
    if "Evidence from News:" in answer:
        return "evidence_from_news"
    if "Financial Statement Evidence:" in answer:
        return "expert_statement_evidence"
    return "unstructured"


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask3/public-00000-of-00001.parquet")
    frame["question_type"] = frame.groupby("task_id", sort=False).cumcount().map(dict(enumerate(QUESTION_TYPES)))
    records = []
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        tables = parse_markdown_tables(context.financial_statements)
        records.append(
            {
                "task_id": row["task_id"],
                "question_type": row["question_type"],
                "answer_format": classify_format(row["answer"]),
                "answer_chars": len(row["answer"]),
                "answer_numbers": "|".join(answer_numbers(row["answer"])),
                "contains_none": bool(re.search(r"\bNone\b", row["answer"], re.IGNORECASE)),
                "statement_tables": len(tables),
                "statement_rows": sum(len(table.rows) for table in tables),
                "financial_statement_evidence": "Financial Statement Evidence:" in row["answer"],
            }
        )
    audit = pd.DataFrame(records)
    output_dir = ROOT / "artifacts/polyfiqa_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "reference_audit.csv", index=False)
    print("formats")
    print(pd.crosstab(audit["question_type"], audit["answer_format"]).to_string())
    print("none")
    print(audit.groupby("question_type")["contains_none"].agg(["sum", "count"]).to_string())
    print("tables", audit.groupby("task_id")["statement_tables"].first().describe().to_dict())
    print("expert_format_anomalies")
    print(audit[audit["financial_statement_evidence"]][["task_id", "question_type", "answer_format"]].to_string(index=False))


if __name__ == "__main__":
    main()

