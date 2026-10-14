#!/usr/bin/env python3
"""Deterministic revenue-trend and R&D-ratio baselines."""

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.finance import extract_rd_ratio, extract_revenue  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402


COMPANIES = {"MSFT": "Microsoft", "HON": "Honeywell", "JNJ": "Johnson & Johnson", "UVV": "Universal Corporation"}
KNOWN_CORRUPT = {("MSFT_20210427", "rd_ratio")}


def company(task_id: str) -> str:
    return COMPANIES[task_id.split("_")[0]]


def compact_amount(value: float, unit: str) -> str:
    """Use the compact dollar notation prevalent in PolyFiQA references."""
    amount = abs(value)
    if unit == "thousands":
        return f"${amount / 1_000:.1f}M" if amount >= 1_000 else f"${amount / 1_000:.3f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.1f}B"
    return f"${amount:,.0f}M"


def precise_amount(value: float, unit: str) -> str:
    """Preserve statement precision for recent reference-style calculations."""
    amount = abs(value)
    if unit == "thousands":
        return f"${amount / 1_000:.3f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.3f}B"
    return f"${amount:,.0f}M"


def revenue_years(task_id: str, financial_statements: str = "") -> tuple[int, int]:
    company_code, date = task_id.split("_", 1)
    year, month = int(date[:4]), int(date[4:6])
    statement_years = sorted({
        int(value) for value in re.findall(r"\b20(?:1\d|2\d)\b", financial_statements)
    }, reverse=True)
    if year >= 2024 and len(statement_years) >= 2:
        return statement_years[0], statement_years[1]
    if company_code == "MSFT" and month >= 8:
        year += 1
    return year, year - 1


def revenue_answer(task_id: str, result, financial_statements: str = "") -> str:
    current = compact_amount(result.current, result.unit)
    if result.previous is None:
        body = f"{company(task_id)} reported revenue of {current}."
    else:
        previous = compact_amount(result.previous, result.unit)
        direction = "increased" if result.change_percent >= 0 else "decreased"
        change = "increase" if result.change_percent >= 0 else "decrease"
        current_year, previous_year = revenue_years(task_id, financial_statements)
        body = (
            f"{company(task_id)}'s revenue {direction} from {previous} in {previous_year} "
            f"to {current} in {current_year}, a {abs(result.change_percent):.1f}% "
            f"year-over-year {change}."
        )
    return f"Answer: {body}\nNews Evidence: None."


def rd_answer(task_id: str, result) -> str:
    if result is None:
        return "Answer: None.\nNews Evidence: None."
    year = int(task_id.split("_", 1)[1][:4])
    if year >= 2024 and result.periods:
        rd, revenue, ratio = result.periods[0]
        body = (
            f"R&D ratio is {ratio:.1f}% "
            f"({precise_amount(rd, result.unit)} ÷ {precise_amount(revenue, result.unit)})"
        )
        if len(result.periods) >= 3:
            cumulative_rd, cumulative_revenue, cumulative_ratio = result.periods[2]
            body += (
                f" and cumulative R&D ratio is {cumulative_ratio:.1f}% "
                f"({precise_amount(cumulative_rd, result.unit)} ÷ "
                f"{precise_amount(cumulative_revenue, result.unit)})"
            )
        return f"Answer: {body}.\nNews Evidence: None."
    return (
        f"Answer: {company(task_id)}'s R&D ratio was "
        f"{result.ratio_percent / 100:.3f} ({result.ratio_percent:.1f}%).\n"
        "News Evidence: None."
    )


def evaluate(frame: pd.DataFrame, kind: str) -> tuple[float, float, list[dict]]:
    predictions, rows = [], []
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        if kind == "revenue":
            result = extract_revenue(context.financial_statements)
            prediction = revenue_answer(row["task_id"], result, context.financial_statements)
        else:
            result = extract_rd_ratio(context.financial_statements)
            prediction = rd_answer(row["task_id"], result)
        predictions.append(prediction)
        rows.append({"task_id": row["task_id"], "prediction": prediction, "reference": row["answer"]})
    clean = ~frame.apply(lambda row: (row["task_id"], kind) in KNOWN_CORRUPT, axis=1)
    all_score = rouge1_f1(frame["answer"].tolist(), predictions)
    clean_predictions = [prediction for prediction, keep in zip(predictions, clean, strict=True) if keep]
    clean_score = rouge1_f1(frame.loc[clean, "answer"].tolist(), clean_predictions)
    return all_score, clean_score, rows


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask3/public-00000-of-00001.parquet")
    output_dir = ROOT / "artifacts/polyfiqa_revenue_rd"
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = [
        ("revenue", frame[frame["question"].str.startswith("What trends")]),
        ("rd_ratio", frame[frame["question"].str.startswith("What is the company's R&D")]),
    ]
    for kind, subset in configurations:
        all_score, clean_score, rows = evaluate(subset.copy(), kind)
        print(f"{kind}_all_rows_rouge1_f1={all_score:.6f}")
        print(f"{kind}_clean_rows_rouge1_f1={clean_score:.6f}")
        pd.DataFrame(rows).to_csv(output_dir / f"{kind}_predictions.csv", index=False)


if __name__ == "__main__":
    main()
