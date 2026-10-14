"""Deterministic extraction and calculations over PolyFiQA statements."""

from dataclasses import dataclass
import re

from fineval.tables import MarkdownTable, numeric_cells, parse_markdown_tables


@dataclass(frozen=True)
class BalanceResult:
    current_assets: float
    total_liabilities: float
    total_equity: float
    liability_to_equity: float
    unit: str
    previous_current_assets: float | None = None
    previous_total_liabilities: float | None = None
    previous_total_equity: float | None = None
    previous_liability_to_equity: float | None = None


@dataclass(frozen=True)
class ComparisonResult:
    current: float
    previous: float | None
    change_percent: float | None
    unit: str


@dataclass(frozen=True)
class RDRatioResult:
    research_and_development: float
    revenue: float
    ratio_percent: float
    unit: str
    periods: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class CashFlowResult:
    operating: tuple[float, ...]
    investing: tuple[float, ...]
    financing: tuple[float, ...]
    unit: str


def statement_unit(text: str) -> str:
    match = re.search(r"(?:dollars\s+)?in\s+(millions|thousands)\b", text, re.IGNORECASE)
    return match.group(1).lower() if match else "units"


def _metric_rows(tables: list[MarkdownTable], predicate) -> list[tuple[str, list[float]]]:
    matches = []
    for table in tables:
        for row in table.rows:
            label = row[0].strip()
            if predicate(label.lower()):
                values = [value for _, value, _ in numeric_cells(row)]
                if values:
                    matches.append((label, values))
    return matches


def extract_balance(text: str) -> BalanceResult:
    tables = parse_markdown_tables(text)
    current_rows = _metric_rows(tables, lambda label: label == "total current assets")
    equity_rows = _metric_rows(
        tables,
        lambda label: label in {
            "total stockholders’ equity", "total stockholders' equity",
            "total shareholders’ equity", "total shareholders' equity",
            "total shareowners’ equity", "total shareowners' equity",
        },
    )
    liability_rows = _metric_rows(tables, lambda label: label == "total liabilities")
    asset_rows = _metric_rows(tables, lambda label: label == "total assets")
    if not current_rows or not equity_rows:
        raise ValueError("missing current-assets or equity row")
    current_assets = current_rows[0][1][0]
    total_equity = equity_rows[0][1][0]
    previous_current_assets = current_rows[0][1][1] if len(current_rows[0][1]) > 1 else None
    previous_total_equity = equity_rows[0][1][1] if len(equity_rows[0][1]) > 1 else None
    if liability_rows:
        total_liabilities = liability_rows[0][1][0]
        previous_total_liabilities = (
            liability_rows[0][1][1] if len(liability_rows[0][1]) > 1 else None
        )
    elif asset_rows:
        # Honeywell does not print a total-liabilities row. Its balance sheet
        # equation includes a tiny redeemable noncontrolling interest; deriving
        # liabilities this way matches the reference ratio after rounding.
        total_liabilities = asset_rows[0][1][0] - total_equity
        previous_total_liabilities = (
            asset_rows[0][1][1] - previous_total_equity
            if len(asset_rows[0][1]) > 1 and previous_total_equity is not None
            else None
        )
    else:
        raise ValueError("missing total-liabilities and total-assets rows")
    return BalanceResult(
        current_assets=current_assets,
        total_liabilities=total_liabilities,
        total_equity=total_equity,
        liability_to_equity=total_liabilities / total_equity,
        unit=statement_unit(text),
        previous_current_assets=previous_current_assets,
        previous_total_liabilities=previous_total_liabilities,
        previous_total_equity=previous_total_equity,
        previous_liability_to_equity=(
            previous_total_liabilities / previous_total_equity
            if previous_total_liabilities is not None and previous_total_equity
            else None
        ),
    )


def _amount_values(values: list[float]) -> list[float]:
    """Remove interleaved percentage columns used in J&J income statements."""
    if any(abs(value) > 1_000 for value in values):
        filtered = [value for value in values if abs(value) > 100]
        if filtered:
            return filtered
    return values


def extract_revenue(text: str) -> ComparisonResult:
    tables = parse_markdown_tables(text)
    rows = _metric_rows(
        tables,
        lambda label: label in {
            "total revenue", "net sales", "sales to customers", "sales to customers (note 9)",
            "sales and other operating revenues",
        },
    )
    if not rows:
        raise ValueError("missing revenue row")
    values = _amount_values(rows[0][1])
    current = values[0]
    previous = values[1] if len(values) > 1 else None
    change = ((current / previous) - 1) * 100 if previous else None
    return ComparisonResult(current, previous, change, statement_unit(text))


def extract_rd_ratio(text: str) -> RDRatioResult | None:
    tables = parse_markdown_tables(text)
    rd_rows = _metric_rows(
        tables,
        lambda label: label in {
            "research and development", "research and development expense",
            "research and development expenses",
        },
    )
    if not rd_rows:
        return None
    revenue = extract_revenue(text)
    rd_values = _amount_values(rd_rows[0][1])
    revenue_rows = _metric_rows(
        tables,
        lambda label: label in {
            "total revenue", "net sales", "sales to customers",
            "sales to customers (note 9)", "sales and other operating revenues",
        },
    )
    revenue_values = _amount_values(revenue_rows[0][1]) if revenue_rows else [revenue.current]
    periods = tuple(
        (rd_value, revenue_value, rd_value / revenue_value * 100)
        for rd_value, revenue_value in zip(rd_values, revenue_values)
        if revenue_value
    )
    rd = rd_values[0]
    return RDRatioResult(
        research_and_development=rd,
        revenue=revenue.current,
        ratio_percent=rd / revenue.current * 100,
        unit=statement_unit(text),
        periods=periods,
    )


def extract_cash_flows(text: str) -> CashFlowResult:
    tables = parse_markdown_tables(text)

    def cash_rows(kind: str):
        return _metric_rows(
            tables,
            lambda label: label.startswith("net cash")
            and kind in label
            and not label.startswith("adjustments"),
        )

    operating_rows = cash_rows("operat")
    investing_rows = cash_rows("invest")
    financing_rows = cash_rows("financ")
    if not financing_rows:
        # Some newer J&J tables omit the printed net-financing subtotal. Sum
        # the component rows following the financing section header instead.
        for table in tables:
            for index, row in enumerate(table.rows):
                if row[0].strip().lower() != "cash flows from financing activities":
                    continue
                components = [numeric_cells(component) for component in table.rows[index + 1:]]
                components = [values for values in components if values]
                if components:
                    # Preserve physical column positions. Some J&J component
                    # rows contain an em dash in the current-period column and
                    # a value only in the prior-period column; compressing each
                    # row to a value list shifts that prior amount into current.
                    columns = sorted({column for values in components for column, _, _ in values})
                    totals = [
                        sum(
                            value
                            for values in components
                            for observed_column, value, _ in values
                            if observed_column == column
                        )
                        for column in columns
                    ]
                    financing_rows = [("derived financing activities", totals)]
                break
            if financing_rows:
                break
    if not operating_rows or not investing_rows or not financing_rows:
        raise ValueError("missing one or more cash-flow totals")
    return CashFlowResult(
        operating=tuple(operating_rows[0][1]),
        investing=tuple(investing_rows[0][1]),
        financing=tuple(financing_rows[0][1]),
        unit=statement_unit(text),
    )


def extract_capex(text: str) -> ComparisonResult:
    """Extract property/equipment capital expenditures across source templates."""
    tables = parse_markdown_tables(text)
    rows = _metric_rows(
        tables,
        lambda label: (
            label == "capital expenditures"
            or "additions to property" in label
            or "expenditures for property" in label
            or "purchases of property" in label
            or "purchase of property" in label
        ) and "acquisition" not in label,
    )
    if not rows:
        raise ValueError("missing capital-expenditure row")
    values = _amount_values(rows[0][1])
    current = abs(values[0])
    previous = abs(values[1]) if len(values) > 1 else None
    change = ((current / previous) - 1) * 100 if previous else None
    return ComparisonResult(current, previous, change, statement_unit(text))
