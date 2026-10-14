import re

IDENTIFIER = re.compile(
    r"(?:Α\.?\s*Μ\.?\s*Σ\.?\s*Ο\.?\s*Ε\.?\s*Λ\.?|"
    r"Γ\.?\s*Ε\.?\s*Μ\.?\s*Η\.?|Α\.?\s*Φ\.?\s*Μ\.?)"
    r"\D{0,20}(\d{4,})",
    re.IGNORECASE,
)

GREEK_MONTH_STEM = (
    r"(?:Ιανουαρ|Φεβρουαρ|Μαρτ|Απριλ|Μαΐ|Μαι|Ιουν|Ιουλ|"
    r"Α(?:υγου|ύγου|υγού)στ|Σεπτεμβρ|Οκτωβρ|Νοεμβρ|Δεκεμβρ)"
)

DATE_DAY = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:η|ης)?\s+{GREEK_MONTH_STEM}",
    re.IGNORECASE,
)

INFLECTED_DATE_DAY = re.compile(
    rf"(?<!\d)(?P<surface>(?P<day>\d{{1,2}})(?:η|ης))\s+{GREEK_MONTH_STEM}",
    re.IGNORECASE,
)

ZONE_DESIGNATOR = re.compile(
    r"(?<!\w)Ζώνη\s+(\d+)(?!\d|[.,]\d)",
    re.IGNORECASE,
)

LEGAL_DESIGNATORS = (
    re.compile(
        r"(?<!\w)(?:άρθρ(?:ο|ου)|άρ\.)\s*(\d+)(?!\d|[.,]\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\w)(?:παράγραφος|παραγράφου|παρ\.)\s*(\d+)(?!\d|[.,]\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\w)(?:σημείωση|σημειώσεις)\s*(\d+)(?!\d|[.,]\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\w)(?:ν(?:όμος|όμου)?\.?|κ\.?\s*ν\.?)\s*(\d{3,})(?!\d)",
        re.IGNORECASE,
    ),
)

ATHENS_EXCHANGE = re.compile(r"Χρη(?:μ|µ)ατιστήριο\s+Αθηνών", re.IGNORECASE)
HIGH_SUPPORT_NESTED_COUNTRIES = (
    (re.compile(r"(?<!\w)Ελλάδας(?!\w)", re.IGNORECASE), "Ελλάδας"),
    (re.compile(r"(?<!\w)Ρουμανίας(?!\w)", re.IGNORECASE), "Ρουμανίας"),
)


def add_high_support_nested_locations(
    entities: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Recover exact train-consistent nested locations in financial names.

    The released ontology labels ``Αθηνών`` inside the Athens Stock Exchange
    name, and the country genitives ``Ελλάδας`` and ``Ρουμανίας`` inside
    organization names. Each convention is repeated without a training
    counterexample. Exact surface gates avoid the broad nested-location rule
    that failed the separate public test.
    """
    output = list(entities)
    if any(
        kind == "ΟΡΓΑΝΙΣΜΟΣ" and ATHENS_EXCHANGE.search(entity)
        for entity, kind in entities
    ):
        item = ("Αθηνών", "ΤΟΠΟΘΕΣΙΑ")
        if item not in output:
            output.append(item)
    for pattern, canonical in HIGH_SUPPORT_NESTED_COUNTRIES:
        if any(
            kind == "ΟΡΓΑΝΙΣΜΟΣ" and pattern.search(entity)
            for entity, kind in entities
        ):
            item = (canonical, "ΤΟΠΟΘΕΣΙΑ")
            if item not in output:
                output.append(item)
    return output


def normalize_calendar_day_entities(
    text: str, entities: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Use the GRFinNUM numeric-core convention for inflected calendar days.

    A prediction has no occurrence offsets, so an inflected surface is removed
    only when every exact occurrence of that surface is a calendar day.  The
    numeric core is then recovered from every explicit day-month expression.
    """
    calendar_matches = list(DATE_DAY.finditer(text))
    inflected_calendar_spans: dict[str, set[tuple[int, int]]] = {}
    for match in INFLECTED_DATE_DAY.finditer(text):
        inflected_calendar_spans.setdefault(match.group("surface"), set()).add(
            match.span("surface")
        )

    def is_inflected_calendar_duplicate(item: tuple[str, str]) -> bool:
        value, label = item
        if label != "ΧΡΟΝΙΚΑ" or value not in inflected_calendar_spans:
            return False
        # Prediction records do not carry occurrence offsets.  If the same
        # surface also appears outside calendar syntax, retain it rather than
        # guessing which occurrence the classifier intended.
        occurrences = {
            match.span() for match in re.finditer(re.escape(value), text)
        }
        return bool(occurrences) and occurrences.issubset(
            inflected_calendar_spans[value]
        )

    out = [item for item in entities if not is_inflected_calendar_duplicate(item)]
    existing = set(out)
    for match in calendar_matches:
        item = (match.group(1), "ΧΡΟΝΙΚΑ")
        if item not in existing:
            out.append(item)
            existing.add(item)
    return out


def add_explicit_identifiers(text: str, entities: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Apply audited Greek identifier and explicit calendar-day recovery."""
    matches = list(IDENTIFIER.finditer(text))
    identifier_spans = [match.span(1) for match in matches]
    identifier_values = [match.group(1) for match in matches]

    # Token classifiers sometimes emit a suffix of a long identifier (for
    # example ``91`` inside ``26911``). Remove it only when every occurrence
    # in the text lies inside an explicitly cued identifier span. This keeps a
    # genuine repeated number elsewhere in the sentence.
    def is_identifier_fragment(item: tuple[str, str]) -> bool:
        value = item[0]
        if not value.isdigit() or not any(
            value != identifier and value in identifier for identifier in identifier_values
        ):
            return False
        occurrences = [match.span() for match in re.finditer(re.escape(value), text)]
        return bool(occurrences) and all(
            any(start >= left and end <= right for left, right in identifier_spans)
            for start, end in occurrences
        )

    out = [item for item in entities if not is_identifier_fragment(item)]
    existing = set(out)
    for match in matches:
        item=(match.group(1), 'ΑΛΛΑ')
        if item not in existing:
            out.append(item); existing.add(item)
    out = normalize_calendar_day_entities(text, out)
    existing = set(out)
    for pattern in LEGAL_DESIGNATORS:
        for match in pattern.finditer(text):
            item = (match.group(1), "ΑΛΛΑ")
            if item not in existing:
                out.append(item)
                existing.add(item)
    return out


def add_zone_designators(
    text: str, entities: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Optional higher-risk recovery for numbered project/geographic zones."""
    out = list(entities)
    existing = set(out)
    for match in ZONE_DESIGNATOR.finditer(text):
        item = (match.group(1), "ΑΛΛΑ")
        if item not in existing:
            out.append(item)
            existing.add(item)
    return out
