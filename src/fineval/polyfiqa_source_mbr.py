"""Label-free quasi-sources and semantic utilities for PolyFiQA MBR."""

from __future__ import annotations

import numpy as np


REQUESTS = {
    "revenue": (
        "Report the revenue trend over the stated periods, including the current "
        "and prior revenue amounts and the direction or percentage change. Include "
        "only news evidence that directly verifies that revenue finding."
    ),
    "balance": (
        "Give the current-assets amount and the ratio of total liabilities to total "
        "equity as final results, without calculation steps. Quote only aligned news "
        "evidence when available."
    ),
    "cash": (
        "Explain the trends in operating, investing, and financing cash flows over "
        "the stated periods and identify material causes supported by the packet. "
        "Quote only aligned news evidence when available."
    ),
    "rd": (
        "Report research-and-development expense relative to revenue for the stated "
        "periods, including the ratio or trend requested. Quote only aligned news "
        "evidence when available."
    ),
    "revenue_focus": (
        "List the top three revenue focuses described by the supplied news, using "
        "specific business areas or drivers rather than generic finance wording."
    ),
    "allocation": (
        "Explain how the company allocates capital, covering the concrete uses of "
        "cash and priorities supported by the financial statements."
    ),
    "margin": (
        "Explain the company's profit-margin strategy using concrete operational, "
        "pricing, cost, mix, or innovation actions supported by the packet."
    ),
    "capex": (
        "Report capital-expenditure amounts or trends and explain their strategic "
        "purpose using financial-statement evidence."
    ),
}


def request_description(family: str) -> str:
    return REQUESTS.get(family, "Answer the financial question using only supported facts.")


def fact_verbalization(graph: dict) -> str:
    clauses = []
    for fact in graph.get("numeric", []):
        values = ", ".join(f"{float(value):.10g}" for value in fact.get("values", []))
        derived = " derived" if fact.get("derived") else ""
        clauses.append(
            f"{fact.get('metric', 'metric')} has{derived} values {values} "
            f"in {fact.get('unit', 'unspecified units')}"
        )
    if not clauses:
        return request_description(str(graph.get("family", "")))
    return ". ".join(clauses) + "."


def cosine_max(vector: np.ndarray, matrix: np.ndarray) -> float:
    if not len(matrix):
        return 0.0
    return float(np.max(matrix @ vector))


def bidirectional_support(
    candidate_parts: np.ndarray, source_parts: np.ndarray,
) -> tuple[float, float, float]:
    """Return semantic precision, recall, and their harmonic mean."""
    if not len(candidate_parts) or not len(source_parts):
        return 0.0, 0.0, 0.0
    similarity = candidate_parts @ source_parts.T
    precision = float(similarity.max(axis=1).mean())
    recall = float(similarity.max(axis=0).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

