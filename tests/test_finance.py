from fineval.finance import extract_balance, extract_capex, extract_cash_flows, extract_rd_ratio, extract_revenue, statement_unit


def test_statement_unit_accepts_dollars_qualifier() -> None:
    assert statement_unit("(Dollars in Millions)") == "millions"


def test_extract_balance() -> None:
    text = """| A | B |
|---|---|
| (In millions) | |
| Total current assets | 1,200 |
| Total liabilities | 800 |
| Total shareholders' equity | 400 |
| Total assets | 1,200 |
"""
    result = extract_balance(text)
    assert result.current_assets == 1200
    assert result.liability_to_equity == 2
    assert result.unit == "millions"


def test_extract_balance_retains_previous_period() -> None:
    text = """| A | B | C |
|---|---|---|
| (In millions) | | |
| Total current assets | 1,200 | 1,100 |
| Total liabilities | 800 | 700 |
| Total shareholders' equity | 400 | 500 |
"""
    result = extract_balance(text)
    assert result.previous_current_assets == 1100
    assert result.previous_liability_to_equity == 1.4


def test_extract_revenue_and_rd_ratio() -> None:
    text = """| A | B | C |
|---|---|---|
| (In millions) | | |
| Total revenue | 2,000 | 1,000 |
| Research and development | 200 | 100 |
"""
    revenue = extract_revenue(text)
    ratio = extract_rd_ratio(text)
    assert revenue.change_percent == 100
    assert ratio is not None
    assert ratio.ratio_percent == 10


def test_extract_cash_flows() -> None:
    text = """| A | B | C |
|---|---|---|
| (In millions) | | |
| Net cash provided by operating activities | 100 | 80 |
| Net cash used by investing activities | (50) | (40) |
| Net cash used by financing activities | (25) | (20) |
"""
    result = extract_cash_flows(text)
    assert result.operating == (100, 80)
    assert result.investing == (-50, -40)


def test_extract_cash_flows_derives_missing_financing_subtotal() -> None:
    text = """| Metric | 2020 | 2019 |
|---|---:|---:|
| Net cash flows from operating activities | 100 | 80 |
| Net cash used by investing activities | (50) | (40) |
| Cash flows from financing activities | | |
| Dividends to shareholders | (20) | (18) |
| Proceeds from short-term debt | 5 | 3 |
| Repayment of long-term debt | (4) | (2) |
"""
    result = extract_cash_flows(text)
    assert result.financing == (-19, -17)


def test_derived_financing_subtotal_preserves_blank_period_columns() -> None:
    text = """| Metric | Current | Prior |
|---|---:|---:|
| Net cash flows from operating activities | 100 | 80 |
| Net cash used by investing activities | (50) | (40) |
| Cash flows from financing activities | | |
| Dividends to shareholders | (20) | (18) |
| Proceeds from current-period debt | 5 | — |
| Proceeds from prior-period transaction | — | 30 |
"""
    result = extract_cash_flows(text)
    assert result.financing == (-15, 12)


def test_extract_capex_normalizes_cash_outflows() -> None:
    text = """| Metric | 2020 | 2019 |
|---|---:|---:|
| (In millions) | | |
| Purchase of property, plant and equipment | (906) | (839) |
"""
    result = extract_capex(text)
    assert result.current == 906
    assert result.previous == 839
