"""Strict typed reference programs for grounded PolyFiQA realization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from fineval.polyfiqa_facts import NumericFact, PolyFiQAFactGraph


COMPANY_NAMES = {
    "MSFT": "Microsoft",
    "JNJ": "Johnson & Johnson",
    "HON": "Honeywell",
    "UVV": "Universal Corporation",
    "BABA": "Alibaba",
    "TCEHY": "Tencent",
}

YEAR_MARKERS = {
    -4: "__YEAR_MINUS_FOUR__",
    -3: "__YEAR_MINUS_THREE__",
    -2: "__YEAR_MINUS_TWO__",
    -1: "__YEAR_MINUS_ONE__",
    0: "__YEAR_CURRENT__",
    1: "__YEAR_PLUS_ONE__",
}

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<prefix>[$€£¥])?"
    r"(?P<open>\()?(?P<sign>-)?(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"(?P<space>\s*)(?P<suffix>%|％|billion|million|thousand|[BM])?"
    r"(?P<close>\))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SlotStyle:
    prefix: str
    suffix: str
    decimals: int
    grouped: bool
    space: str
    negative: bool


@dataclass(frozen=True)
class ProgramSlot:
    metric: str
    index: int
    kind: str
    style: SlotStyle


@dataclass(frozen=True)
class ReferenceProgram:
    source_task_id: str
    source_company: str
    tier: str
    family: str
    state: tuple[tuple[str, str], ...]
    template: str
    slots: tuple[ProgramSlot, ...]
    core: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def task_company(task_id: str) -> str:
    return str(task_id).split("_", 1)[0]


def task_year(task_id: str) -> int | None:
    match = re.search(r"_(20\d{2})\d{4}$", str(task_id))
    return int(match.group(1)) if match else None


def answer_core(answer: str) -> str:
    """Keep the answer realization while excluding separately headed evidence."""
    lines = []
    for line in str(answer).splitlines():
        folded = line.strip().casefold()
        if re.match(
            r"^(?:news evidence|evidence from news|financial statements? evidence|"
            r"english news|japanese news|chinese news|spanish news|greek news)\s*:",
            folded,
        ):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _slot_kind(fact: NumericFact, index: int) -> str:
    """Return the semantic kind from the fact schema, not its metric spelling.

    ``liabilities_to_equity`` is a ratio even though its metric name does not
    contain the literal word ``ratio``.  Relying only on the name silently
    removed almost every balance reference from the strict training corpus.
    """
    metric = fact.metric
    if metric.startswith("rd_period_"):
        return "percent" if index == 2 else "money"
    if fact.unit == "percent" or "percent" in metric:
        return "percent"
    if fact.unit == "ratio" or "ratio" in metric:
        return "ratio"
    return "money"


def _fact_value(
    fact: NumericFact, index: int, kind: str, task_id: str | None = None,
) -> float:
    value = float(fact.values[index])
    if kind == "ratio_from_percent":
        return value / 100
    if kind == "percent_from_ratio":
        return value * 100
    if kind != "money":
        return value
    if fact.unit == "thousands":
        return value / 1_000
    if fact.unit == "units" and str(task_id or "").startswith("UVV"):
        # Universal's unlabelled tables are in thousands.  J&J's unlabelled
        # tables are in millions; treating every ``units`` fact as thousands
        # double-converted J&J amounts and rejected otherwise valid references.
        return value / 1_000
    return value


def _mention_value(match: re.Match[str]) -> tuple[float, str]:
    raw = match.group("number").replace(",", "")
    value = float(raw)
    suffix = (match.group("suffix") or "").casefold()
    if suffix in {"b", "billion"}:
        value *= 1_000
        kind = "money"
    elif suffix in {"m", "million"}:
        kind = "money"
    elif suffix == "thousand":
        value /= 1_000
        kind = "money"
    elif suffix in {"%", "％"}:
        kind = "percent"
    elif match.group("prefix"):
        kind = "money"
    else:
        kind = "ratio" if value < 10 else "money"
    if match.group("sign") or match.group("open"):
        value = -value
    return value, kind


def _slot_candidates(
    graph: PolyFiQAFactGraph, kind: str,
) -> list[tuple[float, NumericFact, int, str]]:
    result = []
    for fact in graph.numeric:
        for index in range(len(fact.values)):
            slot_kind = _slot_kind(fact, index)
            candidate_kind = kind
            if slot_kind == kind:
                pass
            elif slot_kind == "percent" and kind == "ratio":
                candidate_kind = "ratio_from_percent"
            elif slot_kind == "ratio" and kind == "percent":
                candidate_kind = "percent_from_ratio"
            else:
                continue
            result.append((
                _fact_value(fact, index, candidate_kind, graph.task_id), fact, index,
                candidate_kind,
            ))
    return result


def _align_slot(
    match: re.Match[str], graph: PolyFiQAFactGraph,
) -> tuple[NumericFact, int, str] | None:
    shown, kind = _mention_value(match)
    candidates = _slot_candidates(graph, kind)
    if not candidates:
        return None
    ranked = sorted(
        (
            abs(abs(shown) - abs(value)) / max(abs(value), 1),
            abs(abs(shown) - abs(value)), fact.metric, index, fact, candidate_kind,
        )
        for value, fact, index, candidate_kind in candidates
    )
    relative, absolute, _, index, fact, candidate_kind = ranked[0]
    suffix = (match.group("suffix") or "").casefold()
    if kind == "percent":
        valid = absolute <= max(0.6, abs(shown) * 0.05)
    elif kind == "ratio":
        valid = absolute <= 0.04
    elif suffix in {"b", "billion"}:
        valid = absolute <= max(60, abs(shown) * 0.01)
    else:
        valid = absolute <= max(1.0, abs(shown) * 0.006)
    return (fact, index, candidate_kind) if valid and relative <= 0.08 else None


def _style(match: re.Match[str]) -> SlotStyle:
    number = match.group("number").replace(",", "")
    decimals = len(number.rsplit(".", 1)[1]) if "." in number else 0
    return SlotStyle(
        prefix=match.group("prefix") or "",
        suffix=match.group("suffix") or "",
        decimals=decimals,
        grouped="," in match.group("number"),
        space=match.group("space") or "",
        negative=bool(match.group("sign") or match.group("open")),
    )


def semantic_state(graph: PolyFiQAFactGraph) -> tuple[tuple[str, str], ...]:
    """Issuer-independent trend signature guarding literal direction words."""
    relevant = {
        "revenue": ("revenue",),
        "balance": ("current_assets", "liabilities_to_equity"),
        "cash": ("operating_cash_flow", "investing_cash_flow", "financing_cash_flow"),
        "rd": ("rd_period_0", "rd_period_1"),
        "capex": ("capital_expenditures",),
    }.get(graph.family, ())
    result = []
    for metric in relevant:
        fact = graph.fact(metric)
        if fact is None or len(fact.values) < 2:
            continue
        current, previous = fact.values[:2]
        relative = (current - previous) / max(abs(previous), 1)
        direction = "flat" if abs(relative) <= 0.02 else "up" if relative > 0 else "down"
        result.append((metric, direction))
    return tuple(result)


def induce_program(
    task_id: str, answer: str, graph: PolyFiQAFactGraph,
) -> tuple[ReferenceProgram | None, str]:
    """Induce a program only when every non-period number maps to a typed fact."""
    company_code = task_company(task_id)
    company = COMPANY_NAMES.get(company_code)
    if company is None or graph.family not in {"revenue", "balance", "cash", "rd"}:
        return None, "unsupported_company_or_family"
    core = answer_core(answer)
    if not core:
        return None, "empty_core"
    template = core
    # Possessive must be replaced before the plain company name.
    template = re.sub(
        re.escape(company) + r"(?:'s|’s)", "__COMPANY_POSSESSIVE__", template,
        flags=re.IGNORECASE,
    )
    template = re.sub(re.escape(company), "__COMPANY__", template, flags=re.IGNORECASE)
    year = task_year(task_id)
    if year is not None:
        for offset in range(-4, 2):
            template = re.sub(
                rf"(?<!\d){year + offset}(?!\d)", YEAR_MARKERS[offset], template,
            )

    matches = list(NUMBER_RE.finditer(template))
    replacements: list[tuple[int, int, str]] = []
    slots: list[ProgramSlot] = []
    for match in matches:
        shown, kind = _mention_value(match)
        # Question/fiscal ordinals such as Q3 are excluded by the letter
        # lookbehind. Remaining small integer list indices are not financial.
        if not match.group("prefix") and not match.group("suffix") and shown in {1, 2, 3, 4}:
            continue
        aligned = _align_slot(match, graph)
        if aligned is None:
            return None, f"unaligned_number:{match.group(0)}"
        fact, index, kind = aligned
        marker = f"__SLOT_{len(slots)}__"
        replacements.append((match.start(), match.end(), marker))
        slots.append(ProgramSlot(fact.metric, index, kind, _style(match)))
    if not slots:
        return None, "no_typed_slots"
    for left, right, marker in reversed(replacements):
        template = template[:left] + marker + template[right:]
    return ReferenceProgram(
        source_task_id=task_id,
        source_company=company_code,
        tier=graph.tier,
        family=graph.family,
        state=semantic_state(graph),
        template=template,
        slots=tuple(slots),
        core=core,
    ), "aligned"


def _render(value: float, style: SlotStyle) -> str:
    suffix = style.suffix.casefold()
    shown = abs(value)
    if suffix in {"b", "billion"}:
        shown /= 1_000
    elif suffix == "thousand":
        shown *= 1_000
    if style.decimals:
        number = f"{shown:,.{style.decimals}f}" if style.grouped else f"{shown:.{style.decimals}f}"
    else:
        number = f"{shown:,.0f}" if style.grouped else f"{shown:.0f}"
    if style.negative:
        number = f"({number})"
    return f"{style.prefix}{number}{style.space}{style.suffix}"


def execute_program(
    program: ReferenceProgram, graph: PolyFiQAFactGraph, target_task_id: str,
) -> tuple[str | None, str]:
    if program.family != graph.family or program.tier != graph.tier:
        return None, "family_or_tier_mismatch"
    if program.state != semantic_state(graph):
        return None, "semantic_state_mismatch"
    company_code = task_company(target_task_id)
    company = COMPANY_NAMES.get(company_code)
    if company is None:
        return None, "unknown_target_company"
    result = program.template.replace("__COMPANY__", company)
    result = result.replace("__COMPANY_POSSESSIVE__", company + "'s")
    year = task_year(target_task_id)
    if year is None and "__YEAR_" in result:
        return None, "missing_target_year"
    if year is not None:
        for offset in range(-4, 2):
            result = result.replace(YEAR_MARKERS[offset], str(year + offset))
    for slot_index, slot in enumerate(program.slots):
        fact = graph.fact(slot.metric)
        if fact is None or slot.index >= len(fact.values):
            return None, "missing_target_slot"
        value = _fact_value(fact, slot.index, slot.kind, target_task_id)
        result = result.replace(f"__SLOT_{slot_index}__", _render(value, slot.style))
    if "__" in result:
        return None, "unresolved_marker"
    return result, "executed"
