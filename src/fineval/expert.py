"""Conservative retrieval fallback for unseen PolyFiQA-Expert questions."""

import re

from fineval.tables import parse_markdown_tables


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "company", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "which", "with", "its", "any", "please", "provide", "financial",
}
SYNONYMS = {
    "revenue": {"revenue", "sales"},
    "cash": {"cash", "financing", "investing", "operating"},
    "profit": {"profit", "income", "margin", "earnings"},
    "debt": {"debt", "liabilities", "borrowings"},
    "investment": {"investment", "acquisition", "purchase", "capital"},
    "capex": {"capex", "expenditure", "expenditures", "additions", "property", "plant", "equipment"},
    "shareholder": {"shareholder", "stockholder", "dividend", "repurchase"},
}


def tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z][a-z-]+", text.casefold())
        if token not in STOPWORDS and len(token) > 2
    }


def expanded_question_tokens(question: str) -> set[str]:
    result = tokens(question)
    for family in SYNONYMS.values():
        if result & family:
            result |= family
    return result


def retrieve_statement_rows(statements: str, question: str, top_k: int = 4) -> list[str]:
    query = expanded_question_tokens(question)
    scored: list[tuple[float, str]] = []
    for table in parse_markdown_tables(statements):
        for row in table.rows:
            if len(row) < 2 or not any(re.search(r"\d", cell) for cell in row[1:]):
                continue
            label_tokens = tokens(row[0])
            overlap = len(query & label_tokens)
            if not overlap:
                continue
            score = overlap / max(len(label_tokens), 1)
            # Revenue questions need operating revenue lines, not rows that only
            # happen to contain the words "sales" or "revenue" (investment
            # sales, sales expense, unearned revenue, or cost of revenue).
            if {"revenue", "sales"} <= query:
                label = row[0].casefold()
                if any(term in label for term in (
                    "sales of investment", "sales and marketing", "unearned revenue",
                    "cost of revenue",
                )):
                    score -= 2.0
                if any(term in label for term in (
                    "total revenue", "net sales", "product sales", "service sales",
                    "sales to customers", "sales and other operating revenues",
                    "product revenue", "service and other revenue",
                )):
                    score += 2.0
            rendered = " | ".join(cell for cell in row if cell.strip())
            scored.append((score, rendered))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [row for _, row in scored[:top_k]]


def realize_expert(statements: str, question: str) -> str:
    """Return exact statement rows, avoiding unsupported synthesis."""
    evidence = retrieve_statement_rows(statements, question)
    if not evidence:
        return "Answer: The available financial statements do not provide enough evidence.\nFinancial Statement Evidence: None."
    joined = "; ".join(evidence)
    return (
        "Answer: The most relevant reported financial statement items are listed below.\n"
        f"Financial Statement Evidence: {joined}"
    )
