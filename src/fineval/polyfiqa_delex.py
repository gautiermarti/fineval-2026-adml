"""Strict delexicalization for provenance-constrained PolyFiQA realization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from fineval.finance import statement_unit
from fineval.parsing import parse_polyfiqa_query
from fineval.polyfiqa_facts import PolyFiQAFactGraph
from fineval.polyfiqa_programs import (
    COMPANY_NAMES,
    NUMBER_RE,
    YEAR_MARKERS,
    _align_slot,
    _fact_value,
    _mention_value,
    _render,
    _style,
    answer_core,
    task_company,
    task_year,
)


@dataclass(frozen=True)
class DelexSlot:
    marker: str
    source: str
    metric: str | None
    index: int | None
    kind: str | None
    style: dict | None
    copy_value: str | None
    copy_context: str | None
    support_value: float | None = None
    support_surface: str | None = None
    support_occurrences: int | None = None


def _render_planned_value(value: float, style_data: dict) -> str:
    """Render a fact while preserving minus versus accounting parentheses."""
    from fineval.polyfiqa_programs import SlotStyle

    style_data = dict(style_data)
    negative_form = style_data.pop(
        "negative_form", "parentheses" if style_data.get("negative") else "none",
    )
    style_data["negative"] = False
    style = SlotStyle(**style_data)
    rendered = _render(value, style)
    if negative_form == "parentheses":
        return f"({rendered})"
    if negative_form == "minus":
        return (
            rendered[0] + "-" + rendered[1:]
            if style.prefix and rendered.startswith(style.prefix) else "-" + rendered
        )
    return rendered


def _query_copy(match_text: str, query: str) -> tuple[str, str] | None:
    """Return exact query value/context; no fuzzy numeric transformation."""
    found = query.find(match_text)
    if found < 0:
        return None
    return match_text, query[max(0, found - 100):found + len(match_text) + 100]


_FINANCIAL_CUE_GROUPS = {
    "revenue": ("revenue", "sales", "turnover"),
    "assets": ("asset", "liquidity"),
    "leverage": ("liabilit", "equity", "leverage"),
    "operating_cash": ("operating cash", "operating activities"),
    "investing_cash": ("investing cash", "investing activities"),
    "financing_cash": ("financing cash", "financing activities"),
    "acquisition": ("acquisition", "acquired", "purchase of business"),
    "capex": (
        "capex", "capital expenditure", "additions to property",
        "purchase of property", "purchases of property", "property and equipment",
        "invested", "investment",
    ),
    "dividend": ("dividend",),
    "repurchase": ("repurchase", "buyback"),
    "debt": ("debt", "bond", "commercial paper"),
    "rd": ("research and development", "r&d"),
    "profit": ("profit", "margin", "income", "earnings"),
    "segment": ("segment", "sector", "division", "business unit"),
}


def _financial_cues(text: str) -> set[str]:
    folded = text.casefold()
    return {
        group for group, cues in _FINANCIAL_CUE_GROUPS.items()
        if any(cue in folded for cue in cues)
    }


def _stable_discourse_number(match: re.Match[str], shown: float) -> bool:
    """Recognize narrow non-financial bare integers inside dates/product names."""
    if match.group("prefix") or match.group("suffix") or shown != int(shown):
        return False
    left = match.string[max(0, match.start() - 32):match.start()]
    right = match.string[match.end():match.end() + 32]
    if re.search(
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+$", left, flags=re.IGNORECASE,
    ) and re.match(r"__(?:YEAR)_", right):
        return True
    return bool(re.search(r"\bOffice\s+$", left, flags=re.IGNORECASE))


def _query_transform(
    match: re.Match[str], raw: str, query: str, graph: PolyFiQAFactGraph,
    style: dict,
) -> dict | None:
    """Return a uniquely supported input value that exactly renders as ``raw``.

    This admits deterministic formatting transformations such as a statement
    value of 1,287 million rendered as ``$1.3B``.  It is not nearest-number
    matching: if two distinct input magnitudes render to the same target, the
    reference remains rejected.  Bare values are considered only in statement
    tables; news needs an explicit currency, unit, or percent marker.
    """
    try:
        context = parse_polyfiqa_query(query)
    except ValueError:
        return None
    _, target_kind = _mention_value(match)
    target_context = match.string[
        max(0, match.start() - 120):match.end() + 120
    ]
    target_cues = _financial_cues(target_context)
    candidates: list[tuple[float, str, str]] = []

    statements = context.financial_statements
    unit = statement_unit(statements)
    statement_offset = 0
    for line in statements.splitlines(keepends=True):
        is_table = "|" in line
        row_has_currency = bool(re.search(r"[$€£¥]", line))
        for source_match in NUMBER_RE.finditer(line):
            explicit = bool(source_match.group("prefix") or source_match.group("suffix"))
            value, source_kind = _mention_value(source_match)
            if not explicit:
                if not is_table or target_kind != "money":
                    continue
                source_kind = "money"
                # OCR tables often put ``$`` in its own cell.  Small decimals
                # on such rows are literal per-share dollars; large UVV values
                # remain table-thousands despite the separated currency cell.
                separated_literal_currency = row_has_currency and abs(value) < 10
                if not separated_literal_currency and (
                    unit == "thousands" or (
                    unit == "units" and graph.task_id.startswith("UVV")
                    )
                ):
                    value /= 1_000
            if source_kind != target_kind:
                continue
            source_cues = _financial_cues(line)
            if target_cues and not (target_cues & source_cues):
                continue
            if _render_planned_value(value, style) != raw:
                continue
            surface = source_match.group(0)
            absolute = statement_offset + source_match.start()
            context_text = statements[
                max(0, absolute - 100):absolute + len(surface) + 100
            ]
            candidates.append((value, surface, context_text))
        statement_offset += len(line)

    for news in (
        context.english_news, context.chinese_news, context.japanese_news,
        context.spanish_news, context.greek_news,
    ):
        for source_match in NUMBER_RE.finditer(news):
            if not (source_match.group("prefix") or source_match.group("suffix")):
                continue
            value, source_kind = _mention_value(source_match)
            if source_kind != target_kind:
                continue
            if _render_planned_value(value, style) != raw:
                continue
            surface = source_match.group(0)
            candidates.append((
                value, surface,
                news[max(0, source_match.start() - 100):source_match.end() + 100],
            ))

    by_value: dict[float, list[tuple[float, str, str]]] = {}
    for candidate in candidates:
        by_value.setdefault(round(candidate[0], 10), []).append(candidate)
    if len(by_value) != 1:
        return None
    occurrences = next(iter(by_value.values()))
    value, surface, context_text = occurrences[0]
    return {
        "value": value, "surface": surface, "context": context_text,
        "occurrences": len(occurrences), "kind": target_kind,
    }


def delexicalize_reference(
    task_id: str, answer: str, query: str, graph: PolyFiQAFactGraph,
) -> tuple[dict | None, str]:
    """Mask every variable numeric mention or reject the training example.

    Graph-aligned values receive typed fact slots.  A value not represented by
    the current graph is accepted only when the exact rendered literal occurs
    in the input packet; this becomes an oracle-plan copy slot.  The latter is
    useful for training the realizer but must be predicted by a separate plan
    model at inference.
    """
    company_code = task_company(task_id)
    company = COMPANY_NAMES.get(company_code)
    if company is None:
        return None, "unknown_company"
    core = answer_core(answer)
    if not core:
        return None, "empty_core"
    possessive_match = re.search(
        re.escape(company) + r"(?:'s|’s)", core, flags=re.IGNORECASE,
    )
    possessive_surface = possessive_match.group(0) if possessive_match else company + "'s"
    template = re.sub(
        re.escape(company) + r"(?:'s|’s)", "__COMPANY_POSSESSIVE__", core,
        flags=re.IGNORECASE,
    )
    template = re.sub(re.escape(company), "__COMPANY__", template, flags=re.IGNORECASE)
    year = task_year(task_id)
    if year is not None:
        for offset in range(-4, 2):
            template = re.sub(
                rf"(?<!\d){year + offset}(?!\d)", YEAR_MARKERS[offset], template,
            )

    replacements = []
    slots = []
    for match in NUMBER_RE.finditer(template):
        # NUMBER_RE's currency order is ``$`` then ``(``, while common prose is
        # ``($5)``.  In that case the regex starts at ``$`` and consumes only
        # the trailing parenthesis.  Leave that unmatched outer punctuation in
        # the template instead of silently deleting it during slot replacement.
        replacement_start = match.start()
        replacement_end = match.end()
        if match.group("open") and not match.group("close"):
            replacement_start += len(match.group("open"))
        if match.group("close") and not match.group("open"):
            replacement_end -= len(match.group("close"))
        raw = template[replacement_start:replacement_end]
        shown = float(match.group("number").replace(",", ""))
        if match.group("sign") or match.group("open"):
            shown = -shown
        # Quarter/list ordinals are stable discourse tokens rather than values.
        if (
            not match.group("prefix") and not match.group("suffix")
            and shown in {1, 2, 3, 4}
        ) or _stable_discourse_number(match, shown):
            continue
        marker = f"__VALUE_{len(slots)}__"
        style = asdict(_style(match))
        style["negative_form"] = (
            "parentheses" if match.group("open") and match.group("close") else
            "minus" if match.group("sign") else "none"
        )
        aligned = _align_slot(match, graph)
        if aligned is not None:
            fact, index, kind = aligned
            # A loose nearest-fact match can map a quoted acquisition amount to
            # a numerically close cash-flow total.  Exact re-rendering is a hard
            # provenance check; otherwise fall back to an exact query copy.
            if _render_planned_value(
                _fact_value(fact, index, kind, graph.task_id), style,
            ) != raw:
                aligned = None
        if aligned is not None:
            fact, index, kind = aligned
            slot = DelexSlot(
                marker=marker, source="fact", metric=fact.metric, index=index,
                kind=kind, style=style, copy_value=None,
                copy_context=None, support_value=None, support_surface=None,
                support_occurrences=None,
            )
        else:
            copied = _query_copy(raw, query)
            if copied is not None:
                copy_value, copy_context = copied
                slot = DelexSlot(
                    marker=marker, source="query_copy", metric=None, index=None,
                    kind=None, style=None, copy_value=copy_value,
                    copy_context=copy_context, support_value=None,
                    support_surface=None, support_occurrences=None,
                )
            else:
                transformed = _query_transform(match, raw, query, graph, style)
                if transformed is None:
                    return None, f"unverifiable_number:{raw}"
                slot = DelexSlot(
                    marker=marker, source="query_transform", metric=None,
                    index=None, kind=str(transformed["kind"]), style=style,
                    copy_value=None, copy_context=str(transformed["context"]),
                    support_value=float(transformed["value"]),
                    support_surface=str(transformed["surface"]),
                    support_occurrences=int(transformed["occurrences"]),
                )
        replacements.append((replacement_start, replacement_end, marker))
        slots.append(slot)
    for left, right, marker in reversed(replacements):
        template = template[:left] + marker + template[right:]
    return {
        "task_id": task_id,
        "company": company_code,
        "family": graph.family,
        "tier": graph.tier,
        "template": template,
        "company_possessive_surface": possessive_surface,
        "slots": [asdict(slot) for slot in slots],
        "core": core,
    }, "aligned" if slots else "aligned_qualitative"


def restore_reference(record: dict, graph: PolyFiQAFactGraph, task_id: str) -> str:
    """Mechanically restore a delexicalized reference from its typed plan."""
    result = str(record["template"])
    company = COMPANY_NAMES[task_company(task_id)]
    result = result.replace(
        "__COMPANY_POSSESSIVE__",
        str(record.get("company_possessive_surface", company + "'s")),
    )
    result = result.replace("__COMPANY__", company)
    year = task_year(task_id)
    if year is not None:
        for offset, marker in YEAR_MARKERS.items():
            result = result.replace(marker, str(year + offset))
    for raw_slot in record["slots"]:
        if raw_slot["source"] == "query_copy":
            value = raw_slot["copy_value"]
        elif raw_slot["source"] == "query_transform":
            value = _render_planned_value(
                float(raw_slot["support_value"]), raw_slot["style"],
            )
        else:
            fact = graph.fact(raw_slot["metric"])
            if fact is None or int(raw_slot["index"]) >= len(fact.values):
                raise ValueError(f"missing fact slot {raw_slot['metric']}[{raw_slot['index']}]")
            value = _render_planned_value(
                _fact_value(
                    fact, int(raw_slot["index"]), str(raw_slot["kind"]), task_id,
                ),
                raw_slot["style"],
            )
        result = result.replace(raw_slot["marker"], str(value))
    if re.search(r"__(?:VALUE|COMPANY|YEAR)_", result):
        raise ValueError("unresolved delexicalization marker")
    return result


def _oracle_evidence(record: dict, graph: PolyFiQAFactGraph, limit: int = 6):
    """Compact historical teacher plan; deployment must learn this selection."""
    stop = {
        "the", "and", "for", "from", "with", "was", "were", "has", "had",
        "this", "that", "company", "answer", "million", "billion",
    }

    def words(text: str):
        return {
            word for word in re.findall(r"[A-Za-zΑ-Ωα-ω]{3,}", text.casefold())
            if word not in stop
        }

    target_words = words(record["core"])
    used_metrics = {
        slot["metric"] for slot in record["slots"] if slot["source"] == "fact"
    }
    ranked = []
    for index, item in enumerate(graph.evidence):
        overlap = len(target_words & words(item.text))
        metric_match = bool(used_metrics & set(item.metrics))
        ranked.append((metric_match, overlap, -index, item))
    statement = [row for row in ranked if row[3].source == "financial_statements" and row[0]]
    news = [row for row in ranked if row[3].source == "news" and row[1] > 0]
    selected = sorted(statement, reverse=True)[:4]
    selected.extend(sorted(news, reverse=True)[:max(0, limit - len(selected))])
    if not selected and not used_metrics:
        selected = sorted(ranked, reverse=True)[:min(3, limit)]
    return [row[3] for row in selected[:limit]]


def realization_plan(
    record: dict, graph: PolyFiQAFactGraph, *, include_oracle_evidence: bool = True,
) -> dict:
    """Build the compact graph/slot input consumed by a future realizer."""
    used_copy_slots = [
        {
            "marker": slot["marker"], "source": slot["source"],
            "value": slot["copy_value"], "context": slot["copy_context"],
            "support_value": slot.get("support_value"),
            "support_surface": slot.get("support_surface"),
            "support_occurrences": slot.get("support_occurrences"),
        }
        for slot in record["slots"] if slot["source"] != "fact"
    ]
    selected_evidence = _oracle_evidence(record, graph) if include_oracle_evidence else []
    return {
        "task_id": graph.task_id,
        "tier": graph.tier,
        "family": graph.family,
        "numeric": [
            {
                "metric": fact.metric,
                "unit": fact.unit,
                "derived": fact.derived,
                "values": [
                    {"marker": f"__FACT_{fact.metric}_{index}__", "value": value}
                    for index, value in enumerate(fact.values)
                ],
            }
            for fact in graph.numeric
        ],
        "requested_slots": [
            {
                "marker": slot["marker"], "source": slot["source"],
                "metric": slot["metric"], "index": slot["index"],
                "kind": slot["kind"], "style": slot["style"],
            }
            for slot in record["slots"]
        ],
        "evidence": [
            {
                "source": item.source, "language": item.language,
                "metrics": list(item.metrics), "text": item.text,
            }
            for item in selected_evidence
        ],
        "oracle_copy_slots": used_copy_slots,
        "evidence_selection": (
            "historical_reference_oracle" if include_oracle_evidence else "none_fact_swap"
        ),
    }
