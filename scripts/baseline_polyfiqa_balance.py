#!/usr/bin/env python3
"""Deterministic balance-sheet baseline with clean/all-row ROUGE."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.finance import extract_balance  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402


COMPANIES = {"MSFT": "Microsoft", "HON": "Honeywell", "JNJ": "Johnson & Johnson", "UVV": "Universal Corporation"}
KNOWN_CORRUPT = {"MSFT_20210427"}


def realize(task_id: str, result) -> str:
    company = COMPANIES[task_id.split("_")[0]]
    value = result.current_assets
    if result.unit == "thousands":
        amount = f"${value / 1_000_000:.1f}B" if value >= 1_000_000 else f"${value / 1_000:.1f}M"
    elif value >= 1_000:
        amount = f"${value / 1_000:.1f}B"
    else:
        amount = f"${value:,.0f}M"
    return (
        f"Answer: {company}'s total current assets were {amount}. "
        f"The ratio of total liabilities to total equity was {result.liability_to_equity:.2f}.\n"
        "News Evidence: None."
    )


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask3/public-00000-of-00001.parquet")
    frame = frame[frame["question"].str.startswith("How does")].copy()
    predictions = []
    diagnostics = []
    for _, row in frame.iterrows():
        context = parse_polyfiqa_query(row["query"])
        result = extract_balance(context.financial_statements)
        prediction = realize(row["task_id"], result)
        predictions.append(prediction)
        diagnostics.append(
            {
                "task_id": row["task_id"], "current_assets": result.current_assets,
                "total_liabilities": result.total_liabilities, "total_equity": result.total_equity,
                "ratio": result.liability_to_equity, "unit": result.unit,
                "prediction": prediction, "reference": row["answer"],
            }
        )
    clean = ~frame["task_id"].isin(KNOWN_CORRUPT)
    print(f"all_rows_rouge1_f1={rouge1_f1(frame['answer'].tolist(), predictions):.6f}")
    print(f"clean_rows_rouge1_f1={rouge1_f1(frame.loc[clean, 'answer'].tolist(), list(pd.Series(predictions)[clean.to_numpy()])):.6f}")
    output_dir = ROOT / "artifacts/polyfiqa_balance"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
