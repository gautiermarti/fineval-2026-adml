from fineval.expert import realize_expert, retrieve_statement_rows


STATEMENTS = """| Metric | 2021 | 2020 |
|---|---:|---:|
| Total revenue | 120 | 100 |
| Research and development | 18 | 15 |
| Cash used in financing activities | (30) | (20) |
"""


def test_expert_retrieval_expands_revenue_to_sales() -> None:
    rows = retrieve_statement_rows(STATEMENTS, "What drove the change in sales?")
    assert rows[0].startswith("Total revenue")


def test_expert_fallback_quotes_exact_statement_row() -> None:
    answer = realize_expert(STATEMENTS, "How did financing cash use change?")
    assert "Cash used in financing activities | (30) | (20)" in answer
    assert "Financial Statement Evidence:" in answer
