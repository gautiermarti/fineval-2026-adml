"""Utilities for recovering literal rationale spans from EBISU annotations."""

from __future__ import annotations

import re
import unicodedata


LABELS = {"-2", "-1", "0", "+1", "+2"}


def compact_with_map(text: str) -> tuple[str, list[int]]:
    """NFKC-normalize and remove whitespace while retaining source offsets."""
    compact = []
    offsets = []
    for index, character in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", character):
            if not normalized.isspace():
                compact.append(normalized)
                offsets.append(index)
    return "".join(compact), offsets


def compact(text: str) -> str:
    return compact_with_map(text)[0]


def choice_label(annotation: dict) -> str | None:
    for result in annotation.get("result", []):
        if result.get("type") != "choices":
            continue
        choices = result.get("value", {}).get("choices", [])
        if choices:
            label = str(choices[0]).split()[0]
            return label if label in LABELS else None
    return None


def note_fragments(annotation: dict) -> list[str]:
    """Split annotator cue notes at their tilde/ellipsis separators."""
    fragments = []
    for result in annotation.get("result", []):
        if result.get("type") != "textarea":
            continue
        for note in result.get("value", {}).get("text", []):
            for fragment in re.split(r"[〜~…]+", str(note)):
                fragment = fragment.strip(" \t\r\n〜~…・。、")
                if fragment:
                    fragments.append(fragment)
    return fragments


def locate_unique_fragment(text: str, fragment: str) -> tuple[int, int] | None:
    """Locate a note fragment after NFKC/space normalization if unambiguous."""
    normalized_text, offsets = compact_with_map(text)
    normalized_fragment = compact(fragment)
    if len(normalized_fragment) < 2:
        return None
    starts = [
        match.start()
        for match in re.finditer(re.escape(normalized_fragment), normalized_text)
    ]
    if len(starts) != 1:
        return None
    start = starts[0]
    end = start + len(normalized_fragment)
    return offsets[start], offsets[end - 1] + 1

