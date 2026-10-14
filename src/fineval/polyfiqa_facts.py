"""Typed, provenance-preserving facts for PolyFiQA answer compilation.

The graph is deliberately model-agnostic.  Numeric facts come from the same
deterministic statement calculators used by the submission fallback, while
evidence nodes retain exact source lines or sentences.  Candidate realizers
may vary wording, but they must share this graph.
"""

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping
import re

from fineval.finance import (
    extract_balance,
    extract_capex,
    extract_cash_flows,
    extract_rd_ratio,
    extract_revenue,
)
from fineval.parsing import PolyFiQAContext, parse_polyfiqa_query


LANGUAGES = ("english", "chinese", "japanese", "spanish", "greek")


@dataclass(frozen=True)
class NumericFact:
    metric: str
    values: tuple[float, ...]
    unit: str
    derived: bool = False
    operands: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceFact:
    source: str
    text: str
    language: str = "english"
    metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolyFiQAFactGraph:
    task_id: str
    tier: str
    family: str
    numeric: tuple[NumericFact, ...]
    evidence: tuple[EvidenceFact, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fact(self, metric: str) -> NumericFact | None:
        return next((item for item in self.numeric if item.metric == metric), None)


def question_family(question: str) -> tuple[str, str]:
    if question.startswith("What trends"):
        return "easy", "revenue"
    if question.startswith("How does"):
        return "easy", "balance"
    if question.startswith("Are there"):
        return "easy", "cash"
    if question.startswith("What is the company's R&D"):
        return "easy", "rd"
    if question.startswith("Please list"):
        return "expert", "revenue_focus"
    if question.startswith("How is"):
        return "expert", "allocation"
    if question.startswith("What is the company’s strategy"):
        return "expert", "margin"
    if question.startswith("What are the company's capital"):
        return "expert", "capex"
    return "expert", "unknown"


def _statement_evidence(
    statements: str,
    predicate: Callable[[str], bool],
    metrics: tuple[str, ...],
    limit: int = 6,
) -> list[EvidenceFact]:
    """Keep exact Markdown source lines whose first cell matches a predicate."""
    result = []
    for raw in statements.splitlines():
        if "|" not in raw or not re.search(r"\d", raw):
            continue
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if len(cells) < 2 or not predicate(cells[0].casefold()):
            continue
        result.append(EvidenceFact(
            source="financial_statements",
            text=raw.strip(),
            metrics=metrics,
        ))
        if len(result) == limit:
            break
    return result


def _news_sentences(context: PolyFiQAContext) -> list[EvidenceFact]:
    result = []
    boundary = re.compile(r"(?<=[.!?。！？])\s+|[\r\n]+")
    for language in LANGUAGES:
        text = getattr(context, f"{language}_news")
        for sentence in boundary.split(text):
            sentence = sentence.strip().strip('"')
            if len(sentence) >= 20:
                result.append(EvidenceFact(
                    source="news", text=sentence, language=language,
                ))
    return result


def _supplemental_numeric(
    family: str, supplemental: Mapping[str, Any] | None,
) -> list[NumericFact] | None:
    """Normalize audited non-US statement supplements used by the fallback."""
    if not supplemental:
        return None
    if family == "revenue" and supplemental.get("revenue"):
        current, previous, change = supplemental["revenue"]
        return [
            NumericFact("revenue", (current, previous), "millions"),
            NumericFact(
                "revenue_change_percent", (change,), "percent", True,
                ("revenue.current", "revenue.previous"),
            ),
        ]
    if family == "balance" and supplemental.get("assets") is not None:
        return [
            NumericFact("current_assets", (supplemental["assets"],), "millions"),
            NumericFact("liabilities_to_equity", (supplemental["leverage"],), "ratio", True),
        ]
    if family == "cash" and supplemental.get("cash"):
        operating, old_operating, investing, old_investing, financing, old_financing = supplemental["cash"]
        return [
            NumericFact("operating_cash_flow", (operating, old_operating), "millions"),
            NumericFact("investing_cash_flow", (investing, old_investing), "millions"),
            NumericFact("financing_cash_flow", (financing, old_financing), "millions"),
        ]
    if family == "rd":
        if not supplemental.get("rd"):
            return []
        expense, revenue, ratio = supplemental["rd"]
        return [
            NumericFact("research_and_development", (expense,), "millions"),
            NumericFact("revenue", (revenue,), "millions"),
            NumericFact(
                "rd_ratio_percent", (ratio,), "percent", True,
                ("research_and_development.current", "revenue.current"),
            ),
        ]
    return None


def _numeric_graph(
    context: PolyFiQAContext,
    family: str,
    supplemental: Mapping[str, Any] | None = None,
) -> list[NumericFact]:
    supplied = _supplemental_numeric(family, supplemental)
    if supplied is not None:
        return supplied
    statements = context.financial_statements
    if family == "revenue":
        value = extract_revenue(statements)
        facts = [NumericFact(
            "revenue", tuple(x for x in (value.current, value.previous) if x is not None),
            value.unit,
        )]
        if value.change_percent is not None:
            facts.append(NumericFact(
                "revenue_change_percent", (value.change_percent,), "percent", True,
                ("revenue.current", "revenue.previous"),
            ))
        return facts
    if family == "balance":
        value = extract_balance(statements)
        facts = [
            NumericFact(
                "current_assets",
                tuple(x for x in (value.current_assets, value.previous_current_assets) if x is not None),
                value.unit,
            ),
            NumericFact(
                "total_liabilities",
                tuple(x for x in (value.total_liabilities, value.previous_total_liabilities) if x is not None),
                value.unit,
            ),
            NumericFact(
                "total_equity",
                tuple(x for x in (value.total_equity, value.previous_total_equity) if x is not None),
                value.unit,
            ),
            NumericFact(
                "liabilities_to_equity", (value.liability_to_equity,), "ratio", True,
                ("total_liabilities.current", "total_equity.current"),
            ),
        ]
        if value.previous_liability_to_equity is not None:
            facts.append(NumericFact(
                "previous_liabilities_to_equity",
                (value.previous_liability_to_equity,), "ratio", True,
                ("total_liabilities.previous", "total_equity.previous"),
            ))
        return facts
    if family == "cash":
        value = extract_cash_flows(statements)
        return [
            NumericFact("operating_cash_flow", value.operating, value.unit),
            NumericFact("investing_cash_flow", value.investing, value.unit),
            NumericFact("financing_cash_flow", value.financing, value.unit),
        ]
    if family == "rd":
        value = extract_rd_ratio(statements)
        if value is None:
            return []
        result = [
            NumericFact("research_and_development", (value.research_and_development,), value.unit),
            NumericFact("revenue", (value.revenue,), value.unit),
            NumericFact(
                "rd_ratio_percent", (value.ratio_percent,), "percent", True,
                ("research_and_development.current", "revenue.current"),
            ),
        ]
        for index, (rd, revenue, ratio) in enumerate(value.periods):
            result.append(NumericFact(f"rd_period_{index}", (rd, revenue, ratio), value.unit, True))
        return result
    if family == "capex":
        value = extract_capex(statements)
        facts = [NumericFact(
            "capital_expenditures",
            tuple(x for x in (value.current, value.previous) if x is not None), value.unit,
        )]
        if value.change_percent is not None:
            facts.append(NumericFact(
                "capex_change_percent", (value.change_percent,), "percent", True,
                ("capital_expenditures.current", "capital_expenditures.previous"),
            ))
        return facts
    return []


def _family_statement_evidence(context: PolyFiQAContext, family: str) -> list[EvidenceFact]:
    text = context.financial_statements
    if family in {"revenue", "revenue_focus"}:
        return _statement_evidence(
            text,
            lambda label: any(cue in label for cue in (
                "total revenue", "net sales", "sales to customers", "product revenue",
                "service revenue", "sales and other operating revenues",
            )),
            ("revenue",),
        )
    if family == "balance":
        return _statement_evidence(
            text,
            lambda label: any(cue in label for cue in (
                "total current assets", "total liabilities", "total stockholders",
                "total shareholders", "total shareowners", "total equity", "total assets",
            )),
            ("current_assets", "total_liabilities", "total_equity"),
        )
    if family == "cash":
        return _statement_evidence(
            text,
            lambda label: label.startswith("net cash") or "cash paid for acquisition" in label,
            ("operating_cash_flow", "investing_cash_flow", "financing_cash_flow"),
        )
    if family == "rd":
        return _statement_evidence(
            text,
            lambda label: "research and development" in label or label in {"total revenue", "net sales"},
            ("research_and_development", "revenue"),
        )
    if family == "allocation":
        return _statement_evidence(
            text,
            lambda label: any(cue in label for cue in (
                "repurchase", "dividend", "acquisition", "purchase of property",
                "purchases of property", "additions to property", "debt issuance",
                "proceeds from issuance", "proceeds from short-term debt",
                "proceeds from long-term debt",
            )),
            ("capital_allocation",),
            limit=12,
        )
    if family == "margin":
        return _statement_evidence(
            text,
            lambda label: any(cue in label for cue in (
                "gross margin", "gross profit", "operating income", "net income",
                "net earnings", "cost of goods sold", "research and development",
                "selling, marketing",
            )),
            ("profit_margin",),
            limit=10,
        )
    if family == "capex":
        return _statement_evidence(
            text,
            lambda label: any(cue in label for cue in (
                "capital expenditures", "additions to property", "expenditures for property",
                "purchases of property", "purchase of property", "acquisition",
                "in-process research and development",
            )),
            ("capital_expenditures",),
            limit=8,
        )
    return []


def build_fact_graph(
    task_id: str,
    query: str,
    question: str,
    supplemental_facts: Mapping[str, Any] | None = None,
) -> PolyFiQAFactGraph:
    context = parse_polyfiqa_query(query)
    tier, family = question_family(question)
    errors = []
    try:
        numeric = _numeric_graph(context, family, supplemental_facts)
    except (ValueError, ZeroDivisionError, IndexError) as exc:
        numeric = []
        errors.append(f"numeric:{type(exc).__name__}:{exc}")
    evidence = _family_statement_evidence(context, family)
    evidence.extend(_news_sentences(context))
    return PolyFiQAFactGraph(
        task_id=task_id,
        tier=tier,
        family=family,
        numeric=tuple(numeric),
        evidence=tuple(evidence),
        errors=tuple(errors),
    )


def provenance_is_exact(graph: PolyFiQAFactGraph, query: str) -> bool:
    """Check that every non-derived evidence node is an exact packet substring."""
    context = parse_polyfiqa_query(query)
    statement = context.financial_statements
    news = {language: getattr(context, f"{language}_news") for language in LANGUAGES}
    return all(
        item.text in statement if item.source == "financial_statements"
        else item.text in news[item.language]
        for item in graph.evidence
    )
