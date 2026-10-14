from fineval.tables import column_header_context, numeric_cells, parse_markdown_tables


def test_parse_markdown_tables() -> None:
    text = "| A | B |\n|---|---|\n| Revenue | 1,200 |\n\n---\n\n| C | D |\n| X | (50 | ) |"
    tables = parse_markdown_tables(text)
    assert len(tables) == 2
    assert tables[0].find("revenue")[0][1] == "1,200"


def test_numeric_cells_accounting_parentheses() -> None:
    assert numeric_cells(("Cash flow", "(1,234", ")", "500")) == [
        (1, -1234.0, "(1,234"),
        (3, 500.0, "500"),
    ]


def test_column_header_context() -> None:
    table = parse_markdown_tables(
        "| Column 1 | Column 2 | Column 3 | Column 4 |\n|---|---|---|---|\n| | Three Months Ended | | |\n"
        "| | 2021 | | 2020 |\n| Revenue | | 20 | 10 |"
    )[0]
    assert column_header_context(table, 3, 2) == ("Three Months Ended", "2021")
