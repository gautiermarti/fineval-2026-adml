#!/usr/bin/env python3
"""Validate prediction alignment, schema, word budget, and numeric grounding."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.finance import (  # noqa: E402
    extract_balance, extract_cash_flows, extract_rd_ratio, extract_revenue,
)
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from fineval.validation import validate_answer  # noqa: E402
from baseline_polyfiqa_cashflow import money as cash_money  # noqa: E402
from predict_polyfiqa_hybrid import ASIAN_NUMBERS, input_grounded_task_id  # noqa: E402


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame(
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    )


def asian_derived_numbers(row: pd.Series) -> tuple[str, ...]:
    facts = ASIAN_NUMBERS.get(input_grounded_task_id(row))
    if facts is None:
        return ()
    if row.question.startswith("What trends"):
        return (f"{facts['revenue'][2]:.1f}%",)
    if row.question.startswith("How does"):
        return (f"{facts['leverage']:.2f}",)
    if row.question.startswith("What is the company's R&D") and facts["rd"]:
        return (f"{facts['rd'][2]:.2f}%",)
    return ()


def derived_numbers(
    question: str, statements: str, task_id: str = "",
) -> tuple[str, ...]:
    """Return deterministic calculator outputs that need not occur verbatim."""
    values = []
    try:
        if question.startswith("What trends"):
            result = extract_revenue(statements)
            values.append(f"{abs(result.change_percent):.1f}%")
        elif question.startswith("How does"):
            result = extract_balance(statements)
            values.extend([
                str(result.total_liabilities), f"{result.liability_to_equity:.2f}",
            ])
            if result.previous_liability_to_equity is not None:
                values.extend([
                    str(result.previous_total_liabilities),
                    f"{result.previous_liability_to_equity:.2f}",
                ])
        elif question.startswith("Are there"):
            result = extract_cash_flows(statements)
            cash_values = result.operating + result.investing + result.financing
            values.extend(map(str, cash_values))
            # Cash-flow realization renders table values in explicit M/B units.
            # Universal OCR packets sometimes lose their table-level
            # "in thousands" caption, so mirror the predictor's conservative
            # issuer-specific scale rule when declaring derived numbers.
            cash_unit = (
                "thousands"
                if task_id.startswith("UVV") and result.unit == "units"
                else result.unit
            )
            values.extend(cash_money(value, cash_unit) for value in cash_values)
        elif question.startswith("What is the company's R&D"):
            result = extract_rd_ratio(statements)
            if result:
                values.extend([f"{result.ratio_percent:.1f}%", f"{result.ratio_percent / 100:.3f}"])
                values.extend(f"{ratio:.2f}%" for _, _, ratio in result.periods)
    except (ValueError, ZeroDivisionError):
        pass
    return tuple(values)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict", action="store_true",
        help="exit nonzero if any answer fails structure, word, or grounding checks",
    )
    args = parser.parse_args()
    frame = load_input(args.input)
    records = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if len(records) != len(frame):
        raise SystemExit(f"row-count mismatch: {len(records)} predictions for {len(frame)} rows")
    rows = []
    for position, ((_, row), record) in enumerate(zip(frame.iterrows(), records)):
        # Submission-ready files are intentionally minimal and may omit the
        # redundant row index.  If present it must align; task_id always must.
        if (
            ("row_index" in record and record["row_index"] != position)
            or record.get("task_id") != row.task_id
        ):
            raise SystemExit(f"alignment mismatch at row {position}")
        context = parse_polyfiqa_query(row.query)
        packet = "\n".join([
            context.financial_statements, context.english_news, context.chinese_news,
            context.japanese_news, context.spanish_news, context.greek_news,
        ])
        easy = row.question.startswith((
            "What trends", "How does", "Are there", "What is the company's R&D",
        ))
        validation = validate_answer(
            record["answer"], packet, "easy" if easy else "expert",
            allowed_numbers=(
                derived_numbers(
                    row.question, context.financial_statements, str(row.task_id),
                )
                + asian_derived_numbers(row)
            ),
        )
        rows.append({
            "row_index": position, "task_id": row.task_id,
            "words": len(record["answer"].split()),
            "valid_structure": validation.valid_structure,
            "within_word_limit": validation.within_word_limit,
            "unsupported_numbers": " | ".join(validation.unsupported_numbers),
            "valid": validation.valid,
        })
    result = pd.DataFrame(rows)
    output = args.output or args.predictions.with_suffix(".validation.csv")
    result.to_csv(output, index=False)
    print(
        f"rows={len(result)} aligned=true structure={result.valid_structure.sum()} "
        f"word_budget={result.within_word_limit.sum()} numeric={result.unsupported_numbers.eq('').sum()} "
        f"fully_valid={result.valid.sum()} max_words={result.words.max()}"
    )
    if not result.valid.all():
        print(result.loc[~result.valid].head(20).to_string(index=False))
        if args.strict:
            raise SystemExit(
                f"strict PolyFiQA validation failed for {(~result.valid).sum()} rows"
            )


if __name__ == "__main__":
    main()
