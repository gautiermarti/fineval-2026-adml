"""Utilities for the OCR-derived Markdown financial statements."""

from dataclasses import dataclass
import re


_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


@dataclass(frozen=True)
class MarkdownTable:
    rows: tuple[tuple[str, ...], ...]

    def find(self, pattern: str) -> list[tuple[str, ...]]:
        expression = re.compile(pattern, re.IGNORECASE)
        return [row for row in self.rows if row and expression.search(row[0])]

    def find_with_index(self, pattern: str) -> list[tuple[int, tuple[str, ...]]]:
        expression = re.compile(pattern, re.IGNORECASE)
        return [
            (index, row)
            for index, row in enumerate(self.rows)
            if row and expression.search(row[0])
        ]


def split_markdown_row(line: str) -> tuple[str, ...]:
    if not line.lstrip().startswith("|"):
        raise ValueError("not a Markdown table row")
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def parse_markdown_tables(text: str) -> list[MarkdownTable]:
    tables: list[MarkdownTable] = []
    current: list[tuple[str, ...]] = []
    for line in text.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            cells = split_markdown_row(line)
            if all(not cell or _SEPARATOR_CELL.match(cell) for cell in cells):
                continue
            current.append(cells)
        elif current:
            tables.append(MarkdownTable(tuple(current)))
            current = []
    return tables


def numeric_cells(row: tuple[str, ...]) -> list[tuple[int, float, str]]:
    """Return numeric cells, joining accounting parentheses split across cells."""
    values = []
    for index, cell in enumerate(row[1:], start=1):
        compact = cell.replace(",", "").replace("$", "").strip()
        match = re.fullmatch(r"\(?\s*(-?\d+(?:\.\d+)?)\s*\)?", compact)
        if not match:
            continue
        value = float(match.group(1))
        left_parenthesis = cell.lstrip().startswith("(")
        right_parenthesis = index + 1 < len(row) and row[index + 1].strip() == ")"
        if left_parenthesis or right_parenthesis:
            value = -abs(value)
        values.append((index, value, cell))
    return values


def column_header_context(table: MarkdownTable, row_index: int, column_index: int) -> tuple[str, ...]:
    """Collect multi-row header fragments applying to a numeric value column."""
    fragments = []
    # Financial tables place column headers near the top. Stop before the first
    # ten rows or the target, whichever comes first, to avoid section labels.
    header_end = row_index
    for index, row in enumerate(table.rows[:row_index]):
        if row[0].strip() and numeric_cells(row):
            header_end = index
            break
    for row in table.rows[:header_end]:
        candidates = [
            (index, cell.strip())
            for index, cell in enumerate(row[1:], start=1)
            if cell.strip() and index <= column_index
        ]
        if not candidates:
            continue
        _, fragment = min(candidates, key=lambda item: column_index - item[0])
        lower = fragment.lower()
        if lower.startswith("column ") or re.fullmatch(r"\(?in (millions|thousands)\)?(?: \(unaudited\))?", lower):
            continue
        if fragment not in fragments:
            fragments.append(fragment)
    return tuple(fragments)
