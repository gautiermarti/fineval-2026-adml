import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_polyfiqa_cashflow import fiscal_years, money
from baseline_polyfiqa_revenue_rd import compact_amount, revenue_years
from predict_polyfiqa import (
    cash_acquisition_phrase,
)
from validate_polyfiqa_predictions import derived_numbers


def test_compact_money_units() -> None:
    assert money(17_504, "millions") == "$17.504 billion"
    assert money(7_047, "millions") == "$7.047 billion"
    assert money(939, "millions") == "$939 million"
    assert money(-116_739, "thousands") == "$116.7 million"


def test_uvv_ocr_cash_scaling_is_declared_as_a_grounded_derived_number() -> None:
    statement = """| Metric | Current | Prior |
|---|---:|---:|
| Net cash flows from operating activities | 47 | (116,739) |
| Net cash used by investing activities | (8,200) | (5,500) |
| Net cash flows from financing activities | 191 | (7,600) |
"""
    allowed = derived_numbers(
        "Are there any significant changes in cash flows?", statement,
        "UVV_20200805",
    )
    assert "$0.047 million" in allowed
    assert "$116.7 million" in allowed


def test_fiscal_year_convention() -> None:
    assert fiscal_years("MSFT_20201027") == (2021, 2020)
    assert fiscal_years("MSFT_20200730") == (2020, 2019)
    assert fiscal_years("HON_20201030") == (2020, 2019)
    assert fiscal_years("JNJ_20201023") is None


def test_material_cash_acquisition_phrase_uses_current_cumulative_column() -> None:
    statement = """| Metric | Q cur | Q old | YTD cur | YTD old |
|---|---:|---:|---:|---:|
| Cash paid for acquisitions, net of cash acquired | (1,575) | (301) | (67,790) | (1,329) |
"""
    result = SimpleNamespace(
        investing=(-10_700.0, -3_264.0, -82_122.0, -13_546.0),
        unit="millions",
    )
    assert cash_acquisition_phrase(statement, result) == (
        "Investing outflows were driven primarily by acquisitions totaling "
        "$67.790 billion."
    )


def test_immaterial_cash_acquisition_abstains() -> None:
    statement = """| Metric | Current | Prior |
|---|---:|---:|
| Cash paid for acquisitions, net of cash acquired | (5) | (2) |
"""
    result = SimpleNamespace(investing=(-371.0, -273.0), unit="millions")
    assert cash_acquisition_phrase(statement, result) == ""


def test_revenue_formatting_convention() -> None:
    assert compact_amount(35_021, "millions") == "$35.0B"
    assert revenue_years("MSFT_20201027") == (2021, 2020)
    assert revenue_years("MSFT_20240730") == (2024, 2023)
    assert revenue_years("JNJ_20201023") == (2020, 2019)
