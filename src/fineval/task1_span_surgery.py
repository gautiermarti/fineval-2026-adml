"""Counterfactual boundary alternatives for Greek financial NER."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


ENTITY_TYPES = ("ΠΡΟΣΩΠΟ", "ΟΡΓΑΝΙΣΜΟΣ", "ΤΟΠΟΘΕΣΙΑ")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
ARTICLE_RE = re.compile(r"^(?:η|ο|οι|the)$", flags=re.IGNORECASE)
LEGAL_SUFFIX_RE = re.compile(
    r"(?i)(?:\s+(?:Α\s*\.\s*Ε\s*\.?|A\s*\.\s*E\s*\.?|"
    r"S\s*\.\s*A\s*\.?|LTD\.?|LIMITED|PLC|INC\.?|CORP\.?|"
    r"COMPANY|HOLDINGS?))\s*$"
)


@dataclass(frozen=True)
class SpanAlternative:
    start: int
    end: int
    entity_type: str
    surface: str
    operations: tuple[str, ...]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def span_surgeries(
    text: str, start: int, end: int, entity_type: str, width: int = 3,
) -> list[SpanAlternative]:
    """Return exact-substring local alternatives around one gold span."""
    if not (0 <= start < end <= len(text)):
        raise ValueError("invalid gold span")
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"unknown entity type {entity_type}")
    tokens = [match.span() for match in TOKEN_RE.finditer(text)]
    inside = [span for span in tokens if span[0] >= start and span[1] <= end]
    before = [span for span in tokens if span[1] <= start]
    after = [span for span in tokens if span[0] >= end]
    candidates: dict[tuple[int, int, str], set[str]] = {}

    def add(left: int, right: int, kind: str, operation: str) -> None:
        left, right = _trim(text, left, right)
        if left >= right or (left, right, kind) == (start, end, entity_type):
            return
        if not any(character.isalnum() for character in text[left:right]):
            return
        candidates.setdefault((left, right, kind), set()).add(operation)

    for amount in range(1, width + 1):
        if len(inside) > amount:
            add(inside[amount][0], end, entity_type, f"contract_left_{amount}")
            add(start, inside[-amount - 1][1], entity_type, f"contract_right_{amount}")
        if len(before) >= amount:
            add(before[-amount][0], end, entity_type, f"expand_left_{amount}")
        if len(after) >= amount:
            add(start, after[amount - 1][1], entity_type, f"expand_right_{amount}")
    if before and after:
        add(before[-1][0], after[0][1], entity_type, "expand_both_1")

    surface = text[start:end]
    suffix = LEGAL_SUFFIX_RE.search(surface)
    if suffix and suffix.start() > 0:
        add(start, start + suffix.start(), entity_type, "drop_legal_suffix")
    if inside and ARTICLE_RE.match(text[inside[0][0]:inside[0][1]]) and len(inside) > 1:
        add(inside[1][0], end, entity_type, "drop_leading_article")
    if before and ARTICLE_RE.match(text[before[-1][0]:before[-1][1]]):
        add(before[-1][0], end, entity_type, "add_leading_article")

    for match in re.finditer(r"\(([^()]{2,})\)", surface):
        inner_start, inner_end = match.span(1)
        add(start + inner_start, start + inner_end, entity_type, "parenthetical_alias_only")
        if match.start() > 0:
            add(start, start + match.start(), entity_type, "drop_parenthetical_alias")

    for competing_type in ENTITY_TYPES:
        if competing_type != entity_type:
            add(start, end, competing_type, f"change_type_to_{competing_type}")

    return [
        SpanAlternative(
            start=left, end=right, entity_type=kind, surface=text[left:right],
            operations=tuple(sorted(operations)),
        )
        for (left, right, kind), operations in sorted(candidates.items())
    ]


def alternative_asdict(alternative: SpanAlternative) -> dict:
    return asdict(alternative)
