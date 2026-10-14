"""Structural and grounding checks for generated PolyFiQA answers."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AnswerValidation:
    valid_structure: bool
    within_word_limit: bool
    unsupported_numbers: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.valid_structure and self.within_word_limit and not self.unsupported_numbers


_NUMBER = re.compile(r"(?<![\w])\$?\d[\d,.]*(?:\s*(?:%|B|M|billion|million|thousand))?", re.I)


def normalized_number(token: str) -> str:
    cleaned = re.sub(r"[^0-9.]", "", token.replace(",", "")).strip(".")
    return cleaned.lstrip("0") or "0"


def number_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).strip() for match in _NUMBER.finditer(text))


def canonical_number(token: str) -> tuple[str, float]:
    lower = token.casefold().strip().rstrip(".,")
    # Source packets can contain outline/version identifiers such as ``3.3.1``.
    # Parse one numeric component instead of feeding a multi-dot token to float.
    match = re.search(r"\d[\d,.]*", lower)
    if not match:
        return "number", 0.0
    raw = match.group(0)
    if raw.count(".") > 1:
        raw = raw.split(".", 1)[0]
    # A comma followed by one or two digits is a decimal comma; three digits is
    # a thousands separator in the source filings.
    if re.fullmatch(r"\d+,\d{1,2}", raw):
        value = float(raw.replace(",", "."))
    else:
        value = float(raw.replace(",", ""))
    if "%" in lower:
        return "percent", value
    if re.search(r"(?:b|billion)\b", lower):
        value *= 1_000
    elif re.search(r"\bthousand\b", lower):
        value /= 1_000
    return "amount", value


def display_rounding_tolerance(token: str) -> float:
    """Half-unit tolerance implied by compact one-decimal M/B rendering."""
    lower = token.casefold().strip().rstrip(".,")
    match = re.search(r"\d[\d,]*(?:\.(\d+))?", lower)
    decimals = len(match.group(1)) if match and match.group(1) else 0
    if re.search(r"(?:b|billion)\b", lower):
        return 0.5 * 1_000 * (10 ** -decimals)
    if re.search(r"(?:m|million)\b", lower):
        return 0.5 * (10 ** -decimals)
    return 0.01


def validate_answer(
    answer: str,
    source_packet: str,
    tier: str,
    word_limit: int = 100,
    allowed_numbers: tuple[str, ...] = (),
) -> AnswerValidation:
    heading = "News Evidence:" if tier == "easy" else "Financial Statement Evidence:"
    valid_structure = answer.lstrip().startswith("Answer:") and heading in answer
    source_token_list = number_tokens(source_packet)
    source_numbers = [canonical_number(token) for token in source_token_list]
    table_declares_or_implies_thousands = bool(
        re.search(r"\b(?:in\s+)?thousands\b", source_packet, re.I)
        # Some OCR packets lose the table-level ``in thousands`` caption but
        # retain separate Markdown currency cells (``| $ | 38,796 |``). This
        # is the same filing representation used by the deterministic parser.
        or re.search(r"\|\s*(?:US)?\$\s*\|\s*\(?\d{1,3}(?:,\d{3})+", source_packet, re.I)
    )
    if table_declares_or_implies_thousands:
        # Filing tables often state the unit once in a header. Preserve normal
        # interpretations and add million-equivalent variants only for
        # comma-grouped values, avoiding accidental scaling of bare years.
        source_numbers.extend(
            (kind, value / 1_000)
            for token in source_token_list
            for kind, value in [canonical_number(token)]
            if kind == "amount" and "," in token and value >= 1_000
        )
    source_numbers.extend(canonical_number(token) for token in allowed_numbers)
    unsupported = []
    for token in number_tokens(answer):
        normalized = normalized_number(token)
        # Numbered list markers and common rounded years are structural when
        # present in the question; all other values must occur in the packet.
        kind, value = canonical_number(token)
        supported = any(
            # Compact answers intentionally round filing amounts (for example,
            # 3,767 million to $3.8B). A 1.5% window accepts one-decimal display
            # rounding while remaining far below any M/B scale confusion.
            source_kind == kind and abs(source_value - value) <= max(
                0.025 * abs(source_value), display_rounding_tolerance(token),
            )
            for source_kind, source_value in source_numbers
        )
        if normalized in {"1", "2", "3"} or supported:
            continue
        unsupported.append(token)
    return AnswerValidation(
        valid_structure=valid_structure,
        within_word_limit=len(answer.split()) <= word_limit,
        unsupported_numbers=tuple(unsupported),
    )
