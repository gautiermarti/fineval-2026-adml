"""Tsutsuji-backed semantic signatures for Japanese financial responses.

The public Tsutsuji resource groups surface forms of Japanese functional
expressions into semantic equivalence classes.  This module deliberately
extracts exact response-side matches: it does not use JF-ICR labels and it does
not treat a substring hit as a decision rule.  The resulting signatures are
features for source-held-out models or auditable diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class FunctionalExpression:
    surface: str
    semantic_class: str
    identifier: str


@dataclass(frozen=True)
class FunctionalMatch:
    start: int
    end: int
    expression: FunctionalExpression


def load_tsutsuji(path: str | Path) -> list[FunctionalExpression]:
    """Load L9 forms while inheriting MCLASS from their XML ancestors."""

    root = ET.parse(path).getroot()
    entries: list[FunctionalExpression] = []

    def walk(node: ET.Element, inherited_class: str = "") -> None:
        semantic_class = node.attrib.get("MCLASS", inherited_class)
        if node.tag.rsplit("}", 1)[-1] == "L9":
            dotted = "".join((node.text or "").split())
            surface = dotted.replace(".", "")
            if surface and semantic_class:
                entries.append(FunctionalExpression(
                    surface=surface,
                    semantic_class=semantic_class,
                    identifier=node.attrib.get("ID", ""),
                ))
        for child in node:
            walk(child, semantic_class)

    walk(root)
    # XML aliases can produce duplicate surface/class pairs.  Preserve the
    # first stable identifier and prevent duplicated feature counts.
    unique: dict[tuple[str, str], FunctionalExpression] = {}
    for entry in entries:
        unique.setdefault((entry.surface, entry.semantic_class), entry)
    return list(unique.values())


def informative_entries(
    entries: list[FunctionalExpression], minimum_characters: int = 3,
) -> list[FunctionalExpression]:
    """Drop one- and two-character forms that are unsafe substring matches."""

    return [entry for entry in entries if len(entry.surface) >= minimum_characters]


def longest_nonoverlapping_matches(
    text: str, entries: list[FunctionalExpression],
) -> list[FunctionalMatch]:
    """Return exact matches, preferring the most specific surface at overlaps."""

    candidates: list[FunctionalMatch] = []
    for entry in entries:
        start = text.find(entry.surface)
        while start >= 0:
            candidates.append(FunctionalMatch(
                start=start, end=start + len(entry.surface), expression=entry,
            ))
            start = text.find(entry.surface, start + 1)
    candidates.sort(key=lambda row: (-(row.end - row.start), row.start,
                                     row.expression.semantic_class))
    occupied = [False] * len(text)
    selected: list[FunctionalMatch] = []
    for row in candidates:
        if any(occupied[row.start:row.end]):
            continue
        selected.append(row)
        occupied[row.start:row.end] = [True] * (row.end - row.start)
    return sorted(selected, key=lambda row: (row.start, row.end))


def semantic_signature(
    text: str, entries: list[FunctionalExpression], tail_fraction: float = 0.35,
) -> dict[str, float]:
    """Aggregate exact matches by full and top-level semantic class."""

    matches = longest_nonoverlapping_matches(text, entries)
    features: dict[str, float] = {
        "global:matches": float(len(matches)),
        "global:coverage": (
            sum(row.end - row.start for row in matches) / max(len(text), 1)
        ),
    }
    tail_start = int(len(text) * (1.0 - tail_fraction))
    for match in matches:
        semantic_class = match.expression.semantic_class
        top_class = semantic_class[:1]
        relative_end = match.end / max(len(text), 1)
        for namespace, value in (("class", semantic_class), ("top", top_class)):
            prefix = f"{namespace}:{value}"
            features[f"{prefix}:count"] = features.get(f"{prefix}:count", 0.0) + 1.0
            features[f"{prefix}:last"] = max(
                features.get(f"{prefix}:last", 0.0), relative_end,
            )
            if match.start >= tail_start:
                features[f"{prefix}:tail"] = features.get(f"{prefix}:tail", 0.0) + 1.0
    return features
