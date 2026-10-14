"""Immutable-value masking and auditing for PolyFiQA evidence translation."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping


SUPPORTED_LANGUAGES = ("chinese", "japanese", "spanish", "greek")

IMMUTABLE_RE = re.compile(
    r"(?:[$€£¥]\s*)?[-+]?\d[\d.,]*(?:\s*(?:%|％|bps|million|billion|thousand|"
    r"亿美元|亿\d*万(?:美)?元|億\d*万ドル|億ドル|万ドル|兆円|億円|万円))?"
    r"|\b(?:[A-Z]{2,}[A-Z0-9./+-]*|[A-Za-z]+\d+[A-Za-z0-9./+-]*)\b"
)

EVIDENCE_RE = re.compile(
    r"(?P<label>Chinese|Japanese|Spanish|Greek)"
    r"(?:\s+(?:news|News|Evidence))?\s*:\s*\"(?P<text>[^\"]+)\""
)

TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
        "source_claim_count": {"type": "integer", "minimum": 1},
        "translation_claim_count": {"type": "integer", "minimum": 1},
        "notes": {"type": "string"},
    },
    "required": ["translation", "source_claim_count", "translation_claim_count", "notes"],
    "additionalProperties": False,
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "all_material_claims_preserved": {"type": "boolean"},
        "no_unsupported_claims": {"type": "boolean"},
        "polarity_preserved": {"type": "boolean"},
        "temporal_scope_preserved": {"type": "boolean"},
        "source_support_quote": {"type": "string"},
        "translation_support_quote": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": [
        "faithful", "all_material_claims_preserved", "no_unsupported_claims",
        "polarity_preserved", "temporal_scope_preserved", "source_support_quote",
        "translation_support_quote", "confidence", "rationale",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class MaskedEvidence:
    original: str
    masked: str
    mapping: tuple[tuple[str, str], ...]


def mask_immutables(text: str) -> MaskedEvidence:
    """Replace values/identifiers with deterministic, nonnumeric sentinels."""
    mapping = []
    output = []
    cursor = 0
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for index, match in enumerate(IMMUTABLE_RE.finditer(str(text))):
        output.append(text[cursor:match.start()])
        # Base-26 alphabetic index avoids digits that a translator could alter
        # or hallucinate around. The unusual wrapper is easy to audit exactly.
        value = index
        code = ""
        while True:
            code = alphabet[value % 26] + code
            value = value // 26 - 1
            if value < 0:
                break
        placeholder = f"ZXQIMM{code}QXZ"
        output.append(placeholder)
        mapping.append((placeholder, match.group(0)))
        cursor = match.end()
    output.append(text[cursor:])
    return MaskedEvidence(str(text), "".join(output), tuple(mapping))


def restore_immutables(masked_translation: str, item: MaskedEvidence) -> tuple[str | None, str]:
    value = str(masked_translation)
    known = {placeholder for placeholder, _ in item.mapping}
    seen = set(re.findall(r"ZXQIMM[A-Z]+QXZ", value))
    if seen != known:
        return None, "placeholder_set_mismatch"
    if any(value.count(placeholder) != 1 for placeholder in known):
        return None, "placeholder_multiplicity_mismatch"
    # A translation may not introduce unprotected digits.
    without_placeholders = re.sub(r"ZXQIMM[A-Z]+QXZ", "", value)
    if re.search(r"\d", without_placeholders):
        return None, "unprotected_digit_introduced"
    for placeholder, original in item.mapping:
        value = value.replace(placeholder, original)
    return value, "restored"


def validate_translation(
    result: Mapping[str, Any], item: MaskedEvidence, *, maximum_words: int = 45,
) -> tuple[bool, str, str | None]:
    translation = str(result.get("translation", "")).strip()
    if not translation:
        return False, "empty_translation", None
    if len(translation.split()) > maximum_words:
        return False, "translation_too_long", None
    if not re.search(r"[A-Za-z]", translation):
        return False, "translation_not_english", None
    try:
        source_claims = int(result.get("source_claim_count"))
        translated_claims = int(result.get("translation_claim_count"))
    except (TypeError, ValueError):
        return False, "invalid_claim_count", None
    if source_claims < 1 or translated_claims != source_claims:
        return False, "claim_count_mismatch", None
    restored, reason = restore_immutables(translation, item)
    if restored is None:
        return False, reason, None
    for _, immutable in item.mapping:
        if restored.count(immutable) != item.original.count(immutable):
            return False, "immutable_multiplicity_changed", None
    return True, "valid", restored


def extract_translatable_evidence(answer: str) -> list[dict[str, Any]]:
    result = []
    for match in EVIDENCE_RE.finditer(str(answer)):
        language = match.group("label").casefold()
        if language not in SUPPORTED_LANGUAGES:
            continue
        result.append({
            "language": language,
            "text": match.group("text"),
            "start": match.start(),
            "end": match.end(),
            "label": match.group("label"),
        })
    return result


def replace_evidence_translations(
    answer: str, translations: list[Mapping[str, Any]],
) -> tuple[str | None, str]:
    """Replace exact audited spans from right to left; never append evidence."""
    result = str(answer)
    occupied = []
    for record in sorted(translations, key=lambda row: int(row["start"]), reverse=True):
        left, right = int(record["start"]), int(record["end"])
        if left < 0 or right > len(answer) or left >= right:
            return None, "invalid_source_offsets"
        if any(left < old_right and right > old_left for old_left, old_right in occupied):
            return None, "overlapping_source_offsets"
        original = answer[left:right]
        expected = f'{record["label"]}'
        if not original.startswith(expected):
            return None, "source_offset_content_mismatch"
        replacement = (
            f'{record["label"]} news (verified English translation): '
            f'"{record["translation"]}"'
        )
        result = result[:left] + replacement + result[right:]
        occupied.append((left, right))
    if len(result.split()) > 100:
        return None, "answer_word_limit_exceeded"
    return unicodedata.normalize("NFKC", result), "translated"


def verification_is_acceptable(
    verification: Mapping[str, Any], source: str, translation: str,
    *, minimum_confidence: float = 0.8,
) -> tuple[bool, str]:
    required_true = (
        "faithful", "all_material_claims_preserved", "no_unsupported_claims",
        "polarity_preserved", "temporal_scope_preserved",
    )
    if not all(verification.get(field) is True for field in required_true):
        return False, "semantic_verifier_rejected"
    try:
        confidence = float(verification.get("confidence"))
    except (TypeError, ValueError):
        return False, "invalid_verifier_confidence"
    if confidence < minimum_confidence:
        return False, "low_verifier_confidence"
    source_quote = re.sub(r"\s+", "", str(verification.get("source_support_quote", "")))
    target_quote = re.sub(r"\s+", "", str(verification.get("translation_support_quote", "")))
    if not source_quote or source_quote not in re.sub(r"\s+", "", source):
        return False, "ungrounded_source_verifier_quote"
    if not target_quote or target_quote not in re.sub(r"\s+", "", translation):
        return False, "ungrounded_translation_verifier_quote"
    return True, "verified"

