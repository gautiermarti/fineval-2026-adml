"""Evidence-entailment utilities for grounded PolyFiQA candidate selection.

The neural NLI model is deliberately kept out of this module.  These helpers
construct auditable hypotheses and symbolic support signals that can be tested
without downloading or loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


LABEL_RE = re.compile(
    r"^(?:Answer|News Evidence|Financial Statement Evidence|"
    r"English news|Chinese news|Japanese news|Spanish news|Greek news)\s*:\s*",
    re.IGNORECASE,
)
QUOTE_RE = re.compile(r'["“](.+?)["”](?=(?:\s|;|$))', re.DOTALL)
NUMBER_RE = re.compile(
    r"(?<![\w.])(?:[$€£]\s*)?"
    r"(?P<number>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent|percentage\s+points?|billion|million|bn|mn|[BM])?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericMention:
    """A normalized numeric claim.

    Monetary magnitudes are represented in millions, percentages in percentage
    points, and unitless ratios in their original scale.
    """

    value: float
    kind: str


def clean_claim(text: str) -> str:
    """Remove answer/evidence labels while preserving the actual claim."""
    value = text.strip()
    while True:
        updated = LABEL_RE.sub("", value, count=1).strip()
        if updated == value:
            return value
        value = updated


def normalized_text(text: str) -> str:
    """Normalize source quotations for conservative containment matching."""
    value = unicodedata.normalize("NFKC", text).casefold()
    value = value.replace("**", "").replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n\"'“”‘’.;:")


def evidence_quotes(evidence: str) -> list[str]:
    """Extract quoted evidence spans, with a safe fallback for unquoted text."""
    if not evidence.strip() or re.search(r"\bNone\s*\.?$", evidence, re.IGNORECASE):
        return []
    quotes = [match.strip() for match in QUOTE_RE.findall(evidence) if match.strip()]
    if quotes:
        return list(dict.fromkeys(quotes))
    fallback = clean_claim(evidence)
    return [fallback] if fallback and fallback.casefold() != "none." else []


def quotation_support(evidence: str, sources: list[str]) -> tuple[float, float]:
    """Return exact quote support rate and mean conservative containment score."""
    quotes = evidence_quotes(evidence)
    if not quotes:
        explicitly_empty = re.search(r"\bNone\s*\.?$", evidence, re.IGNORECASE)
        return (1.0, 1.0) if not evidence.strip() or explicitly_empty else (0.0, 0.0)
    normalized_sources = [normalized_text(source) for source in sources]
    exact = []
    containment = []
    for quote in quotes:
        needle = normalized_text(quote)
        if not needle:
            exact.append(0.0)
            containment.append(0.0)
            continue
        supported = any(
            needle in source or (len(source) >= 24 and source in needle)
            for source in normalized_sources
        )
        exact.append(float(supported))
        # Candidate quotations are normally copied verbatim.  The token score
        # catches only mild formatting drift; it is not a semantic substitute.
        quote_tokens = set(re.findall(r"\w+", needle))
        best = 0.0
        for source in normalized_sources:
            source_tokens = set(re.findall(r"\w+", source))
            if quote_tokens:
                best = max(best, len(quote_tokens & source_tokens) / len(quote_tokens))
        containment.append(best)
    return sum(exact) / len(exact), sum(containment) / len(containment)


def numeric_mentions(text: str) -> list[NumericMention]:
    """Extract useful financial magnitudes while ignoring years and tiny indices."""
    output = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group("number").replace(",", "")
        value = float(raw)
        unit = (match.group("unit") or "").casefold()
        if unit in {"billion", "bn", "b"}:
            output.append(NumericMention(value * 1000.0, "millions"))
        elif unit in {"million", "mn", "m"}:
            output.append(NumericMention(value, "millions"))
        elif unit in {"%", "percent", "percentage point", "percentage points"}:
            output.append(NumericMention(value, "percent"))
        elif 1900 <= abs(value) <= 2100 and value.is_integer():
            continue
        elif abs(value) >= 10 or "." in raw:
            output.append(NumericMention(value, "raw"))
    return output


def graph_numeric_mentions(graph: dict) -> list[NumericMention]:
    """Convert typed fact-graph values into the same normalized representation."""
    output = []
    for fact in graph.get("numeric", []):
        unit = str(fact.get("unit", "")).casefold()
        if unit == "millions":
            kind = "millions"
        elif unit == "percent":
            kind = "percent"
        else:
            kind = "raw"
        output.extend(NumericMention(float(value), kind) for value in fact.get("values", []))
    return output


def numeric_match(left: NumericMention, right: NumericMention) -> bool:
    """Match rounded financial values with unit-aware tolerances."""
    compatible = left.kind == right.kind or "raw" in {left.kind, right.kind}
    if not compatible:
        return False
    scale = max(abs(left.value), abs(right.value), 1.0)
    tolerance = 0.02 * scale
    if left.kind == right.kind == "percent":
        tolerance = max(tolerance, 0.25)
    elif left.kind == right.kind == "raw":
        tolerance = max(tolerance, 0.02)
    return abs(left.value - right.value) <= tolerance


def numeric_support(text: str, graph: dict) -> tuple[float, int, int]:
    """Return supported fraction, supported count, and unsupported count.

    Support is checked against both typed facts and immutable source evidence.
    This keeps a news-only number from being falsely marked unsupported.
    """
    claims = numeric_mentions(text)
    if not claims:
        return 1.0, 0, 0
    sources = graph_numeric_mentions(graph)
    for item in graph.get("evidence", []):
        sources.extend(numeric_mentions(str(item.get("text", ""))))
    supported = sum(
        any(numeric_match(claim, source) for source in sources) for claim in claims
    )
    return supported / len(claims), supported, len(claims) - supported


def fact_clauses(graph: dict) -> list[str]:
    """Create short NLI premises rather than one overloaded fact paragraph."""
    clauses = []
    for fact in graph.get("numeric", []):
        values = ", ".join(f"{float(value):.10g}" for value in fact.get("values", []))
        qualifier = "Derived " if fact.get("derived") else ""
        clauses.append(
            f"{qualifier}{fact.get('metric', 'metric')} values are {values} "
            f"in {fact.get('unit', 'unspecified units')}."
        )
    return clauses
